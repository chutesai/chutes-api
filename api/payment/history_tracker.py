import traceback
import uuid
import backoff
from fastapi import FastAPI, status, HTTPException
from contextlib import asynccontextmanager
from sqlalchemy import (
    select,
    update,
    func,
    text,
)
from sqlalchemy.exc import IntegrityError
from async_substrate_interface.sync_substrate import SubstrateInterface
from async_substrate_interface.types import ss58_encode
import asyncio
from datetime import timedelta, datetime
from loguru import logger
from typing import Optional
from api.config import settings
from api.database import get_session
from api.payment.schemas import BTTransferMonitorState, BTTxHistory

RECOVERY_BLOCKS = 100000


class BTTransferMonitor:
    def __init__(self):
        self.substrate = SubstrateInterface(url=settings.subtensor, ss58_format=42)
        self.max_recovery_blocks = RECOVERY_BLOCKS
        self.lock_timeout = timedelta(minutes=5)
        self._is_running = False
        self.instance_id = str(uuid.uuid4())

    async def initialize(self):
        """
        Initialize the monitor state in the database.
        """
        async with get_session() as session:
            result = await session.execute(select(BTTransferMonitorState))
            if not result.scalar_one_or_none():
                current_block = self.get_latest_block()
                start_block = max(0, current_block - self.max_recovery_blocks)
                block_hash = self.get_block_hash(start_block)
                state = BTTransferMonitorState(
                    instance_id=self.instance_id,
                    block_number=start_block,
                    block_hash=block_hash,
                )
                session.add(state)
                await session.commit()
                logger.info(
                    f"Initialized state at block {start_block} (current: {current_block}, recovery: {self.max_recovery_blocks} blocks)"
                )

    def _reconnect(self):
        """
        Reconnect to substrate if connection is lost.
        """
        try:
            substrate = SubstrateInterface(url=settings.subtensor, ss58_format=42)
            self.substrate = substrate
            logger.info("Reconnected to substrate")
        except Exception as exc:
            logger.error(f"Error reconnecting to substrate: {exc}")

    @backoff.on_exception(
        backoff.constant,
        Exception,
        jitter=None,
        interval=3,
        max_tries=7,
    )
    def get_latest_block(self):
        """
        Get the latest block number from the chain.
        """
        try:
            return self.substrate.get_block_number(self.substrate.get_chain_head())
        except Exception:
            self._reconnect()
            raise

    @backoff.on_exception(
        backoff.constant,
        Exception,
        jitter=None,
        interval=3,
        max_tries=7,
    )
    def get_block_hash(self, block_number):
        """
        Get the block hash for a given block number.
        """
        try:
            return self.substrate.get_block_hash(block_number)
        except Exception:
            self._reconnect()
            raise

    @backoff.on_exception(
        backoff.constant,
        Exception,
        jitter=None,
        interval=3,
        max_tries=7,
    )
    def get_events(self, block_hash):
        """
        Get all events for a given block hash.
        """
        try:
            return self.substrate.get_events(block_hash)
        except Exception:
            self._reconnect()
            raise

    async def _lock(self) -> bool:
        """
        Acquire a process lock to prevent duplicate processing.
        """
        async with get_session() as session:
            result = await session.execute(
                update(BTTransferMonitorState)
                .where(
                    (BTTransferMonitorState.is_locked.is_(False))
                    | (BTTransferMonitorState.last_updated_at <= func.now() - self.lock_timeout)
                )
                .values(
                    is_locked=True,
                    lock_holder=self.instance_id,
                    locked_at=func.now(),
                    last_updated_at=func.now(),
                )
                .returning(BTTransferMonitorState.instance_id)
            )
            acquired = bool(result.scalar_one_or_none())
            if acquired:
                await session.commit()
                logger.info(f"Acquired lock: {self.instance_id}")
            return acquired

    async def _unlock(self):
        """
        Release the process lock.
        """
        async with get_session() as session:
            await session.execute(
                update(BTTransferMonitorState)
                .where(BTTransferMonitorState.lock_holder == self.instance_id)
                .values(
                    is_locked=False,
                    lock_holder=None,
                    locked_at=None,
                )
            )
            await session.commit()
            logger.info(f"Released lock: {self.instance_id}")

    async def _get_state(self) -> tuple[int, str]:
        """
        Get the current processing state from the database.
        """
        async with get_session() as session:
            result = await session.execute(select(BTTransferMonitorState))
            state = result.scalar_one()
            block, hash_ = state.block_number, state.block_hash
            current_block = self.get_latest_block()
            if (delta := current_block - block) > self.max_recovery_blocks:
                block = current_block - self.max_recovery_blocks
                logger.warning(f"Monitor is {delta} blocks behind, skipping to block {block}")
                hash_ = self.get_block_hash(block)

            return block, hash_

    async def _save_state(self, block_number: int, block_hash: str):
        """
        Save the current processing state to the database.
        """
        async with get_session() as session:
            try:
                await session.execute(
                    update(BTTransferMonitorState).values(
                        block_number=block_number,
                        block_hash=block_hash,
                        last_updated_at=func.now(),
                    )
                )
                await session.commit()
            except Exception as e:
                logger.error(f"Error saving state: {e}")
                await session.rollback()

    async def _record_transfer(
        self,
        block: int,
        block_hash: str,
        extrinsic_idx: Optional[int],
        from_address: str,
        to_address: str,
        amount: int,
    ):
        """
        Record a transfer in the bt_tx_history table.
        """
        extrinsic_id = f"{block}-{extrinsic_idx if extrinsic_idx is not None else 'event'}"
        async with get_session() as session:
            transfer = BTTxHistory(
                extrinsic_id=extrinsic_id,
                block=block,
                rao_amount=amount,
                transaction_hash=block_hash,
                source=from_address,
                dest=to_address,
            )
            session.add(transfer)
            try:
                await session.commit()
                logger.info(
                    f"Recorded transfer: {extrinsic_id} | {from_address[:8]}...{from_address[-4:]} -> "
                    f"{to_address[:8]}...{to_address[-4:]} | {amount} rao"
                )
            except IntegrityError:
                await session.rollback()
                logger.debug(f"Skipping duplicate transfer: {extrinsic_id}")

    async def monitor_transfers(self):
        """
        Main monitoring loop that processes all transfers on the chain.
        """
        logger.info("Starting BT transfer monitoring loop...")
        self._is_running = True

        try:
            while self._is_running:
                if not await self._lock():
                    logger.debug("Failed to acquire lock, waiting...")
                    await asyncio.sleep(10)
                    continue

                current_block_number, current_block_hash = await self._get_state()
                latest_block_number = self.get_latest_block()

                while self._is_running:
                    if current_block_number >= latest_block_number:
                        while (
                            self._is_running
                            and (latest_block_number := self.get_latest_block())
                            == current_block_number
                        ):
                            logger.debug(
                                f"Waiting for next block (current: {current_block_number})..."
                            )
                            await asyncio.sleep(3)

                    current_block_hash = self.get_block_hash(current_block_number)
                    events = self.get_events(current_block_hash)

                    transfers_count = 0
                    logger.info(f"Processing block {current_block_number}...")

                    for raw_event in events:
                        event = raw_event.get("event")
                        if not event:
                            continue

                        if (
                            event.get("module_id") == "Balances"
                            and event.get("event_id") == "Transfer"
                            and event.get("attributes")
                            and not ({"from", "to", "amount"} - set(event["attributes"]))
                        ):
                            from_address = event["attributes"]["from"]
                            to_address = event["attributes"]["to"]

                            if isinstance(from_address, (list, tuple)):
                                from_address = ss58_encode(
                                    bytes(from_address[0]).hex(), ss58_format=42
                                )
                                to_address = ss58_encode(bytes(to_address[0]).hex(), ss58_format=42)

                            amount = event["attributes"]["amount"]
                            extrinsic_idx = raw_event.get("extrinsic_idx")

                            # Record the transfer
                            await self._record_transfer(
                                block=current_block_number,
                                block_hash=current_block_hash,
                                extrinsic_idx=extrinsic_idx,
                                from_address=from_address,
                                to_address=to_address,
                                amount=amount,
                            )
                            transfers_count += 1

                    if transfers_count > 0:
                        logger.success(
                            f"Processed {transfers_count} transfers in block {current_block_number}"
                        )

                    await self._save_state(current_block_number, current_block_hash)
                    current_block_number += 1

        except Exception as exc:
            logger.error(f"Unexpected error in monitor loop: {exc}\n{traceback.format_exc()}")
            raise
        finally:
            logger.info("Shutting down transfer monitor...")
            await self._unlock()

    async def stop(self):
        """
        Stop the monitoring loop gracefully.
        """
        self._is_running = False


monitor = BTTransferMonitor()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manage application lifecycle.
    """
    logger.info("Starting BT Transfer Monitor application...")
    await monitor.initialize()
    monitor_task = asyncio.create_task(monitor.monitor_transfers())
    yield
    await monitor.stop()
    await monitor_task
    logger.info("BT Transfer Monitor application stopped")


app = FastAPI(lifespan=lifespan, title="BT Transfer Monitor")


@app.get("/status")
async def get_status():
    """
    Health check endpoint for the transfer monitor.
    """
    try:
        async with get_session() as session:
            result = await session.execute(select(BTTransferMonitorState))
            state = result.scalar_one_or_none()
            if not state:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail={"status": "unhealthy", "error": "No monitor state found"},
                )

            latest_block = monitor.get_latest_block()
            block_lag = latest_block - state.block_number

            stats_result = await session.execute(
                text("""
                    SELECT
                        COUNT(*) as total_transfers,
                        MAX(block) as last_transfer_block,
                        MIN(created_at) as first_transfer_time,
                        MAX(created_at) as last_transfer_time
                    FROM bt_tx_history
                """)
            )
            stats = stats_result.one()

            failures = []
            if not state.is_locked:
                failures.append("Process is not locked")
            if state.is_locked and state.lock_holder != monitor.instance_id:
                failures.append("Lock held by different instance")
            if state.last_updated_at < datetime.now() - monitor.lock_timeout:
                failures.append("Updates have ceased")
            if block_lag > 10:
                failures.append(f"Monitor is {block_lag} blocks behind")

            response_data = {
                "status": "healthy" if not failures else "unhealthy",
                "monitor": {
                    "current_block": state.block_number,
                    "latest_network_block": latest_block,
                    "block_lag": block_lag,
                    "last_updated_at": state.last_updated_at.isoformat()
                    if state.last_updated_at
                    else None,
                    "is_locked": state.is_locked,
                    "lock_holder": state.lock_holder,
                    "instance_id": monitor.instance_id,
                },
                "transfers": {
                    "total_count": stats.total_transfers,
                    "last_transfer_block": stats.last_transfer_block,
                    "first_transfer_time": stats.first_transfer_time.isoformat()
                    if stats.first_transfer_time
                    else None,
                    "last_transfer_time": stats.last_transfer_time.isoformat()
                    if stats.last_transfer_time
                    else None,
                },
                "failures": failures,
            }

            if failures:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=response_data,
                )

            return response_data

    except HTTPException:
        raise
    except Exception as e:
        error_response = {
            "status": "unhealthy",
            "error": str(e),
        }
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=error_response
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8077)
