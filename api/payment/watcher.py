import traceback
import uuid
import backoff
from fastapi import FastAPI, status, HTTPException
from contextlib import asynccontextmanager
from sqlalchemy import (
    case,
    select,
    update,
    and_,
    or_,
    func,
    text,
)
from sqlalchemy.exc import IntegrityError
from async_substrate_interface.sync_substrate import SubstrateInterface
import asyncio
from datetime import timedelta
from loguru import logger
from typing import Tuple
from api.fmv.fetcher import get_fetcher
import api.database.orms  # noqa: F401
from api.user.schemas import User
from api.payment.schemas import Payment, PaymentMonitorState
from api.permissions import Permissioning
from api.config import settings
from api.database import get_session, engine, Base
from api.autostake import stake


class PaymentMonitor:
    def __init__(self):
        self.substrate = SubstrateInterface(url=settings.subtensor, ss58_format=42)
        self.max_recovery_blocks = settings.payment_recovery_blocks
        self.lock_timeout = timedelta(minutes=5)
        self.max_recover_blocks = 32
        self._payment_addresses = set()
        self._developer_payment_addresses = set()
        self._is_running = False
        self.instance_id = str(uuid.uuid4())
        self._user_refresh_timestamp = None

    async def initialize(self):
        """
        Load state from the database and lock the process.
        """
        logger.info("Inside initialize...")
        async with get_session() as session:
            result = await session.execute(select(PaymentMonitorState))
            if not result.scalar_one_or_none():
                current_block = self.get_latest_block()
                block_hash = self.get_block_hash(current_block)
                state = PaymentMonitorState(
                    instance_id=self.instance_id,
                    block_number=current_block,
                    block_hash=block_hash,
                )
                session.add(state)
                await session.commit()

    def _reconnect(self):
        """
        Substrate reconnect helper.
        """
        try:
            substrate = SubstrateInterface(url=settings.subtensor, ss58_format=42)
            self.substrate = substrate
        except Exception as exc:
            logger.error(f"Error (re)connecting to substrate @ {settings.subtensor}: {exc}")

    @backoff.on_exception(
        backoff.constant,
        Exception,
        jitter=None,
        interval=3,
        max_tries=7,
    )
    def get_latest_block(self):
        """
        Get the latest block number.
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
        Get the hash for a block number.
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
        Get events for a block hash.
        """
        try:
            return self.substrate.get_events(block_hash)
        except Exception:
            self._reconnect()
            raise

    async def _lock(self) -> bool:
        """
        Attempt acquiring a lock to ensure we aren't double tracking/crediting accounts.
        """
        acquired = False
        async with get_session() as session:
            result = await session.execute(
                update(PaymentMonitorState)
                .where(
                    or_(
                        PaymentMonitorState.is_locked.is_(False),
                        PaymentMonitorState.last_updated_at <= func.now() - self.lock_timeout,
                    )
                )
                .values(
                    is_locked=True,
                    lock_holder=self.instance_id,
                    locked_at=func.now(),
                    last_updated_at=func.now(),
                )
                .returning(PaymentMonitorState.instance_id)
            )
            acquired = bool(result.scalar_one_or_none())
            if acquired:
                await session.commit()
        return acquired

    async def _unlock(self) -> bool:
        """
        Unlock (e.g. release the lock after a shutdown).
        """
        async with get_session() as session:
            await session.execute(
                update(PaymentMonitorState).values(
                    is_locked=False,
                    lock_holder=None,
                    locked_at=None,
                )
            )
            await session.commit()

    async def _refresh_addresses(self):
        """
        Refresh the set of payment addresses from database.
        """
        async with get_session() as session:
            query = select(User.payment_address, User.developer_payment_address, User.updated_at)
            # if self._user_refresh_timestamp:
            #    query = query.where(User.updated_at > self._user_refresh_timestamp)
            query = query.order_by(User.updated_at.asc())
            result = await session.execute(query)
            for payment_address, developer_payment_address, updated_at in result:
                self._payment_addresses.add(payment_address)
                if developer_payment_address:
                    self._developer_payment_addresses.add(developer_payment_address)
                # logger.info(f"Addresses: {payment_address=} {developer_payment_address=}")
                self._user_refresh_timestamp = updated_at

    async def _handle_payment(
        self,
        to_address: str,
        from_address: str,
        amount: int,
        block: int,
        block_hash: str,
        fmv: float,
    ):
        """
        Process an incoming transfer.
        """
        async with get_session() as session:
            user = (
                await session.execute(select(User).where(User.payment_address == to_address))
            ).scalar_one_or_none()
            if not user:
                logger.warning(f"Failed to find user with payment address {to_address}")
                return

            # Sum payments prior to this one.
            total_query = select(func.sum(Payment.usd_amount)).where(
                Payment.user_id == user.user_id, Payment.purpose == "credits"
            )
            total_payments = (await session.execute(total_query)).scalar() or 0

            # Store the payment record.
            payment_id = str(
                uuid.uuid5(uuid.NAMESPACE_OID, f"{block}:{to_address}:{from_address}:{amount}")
            )
            delta = amount * fmv / 1e9
            payment = Payment(
                payment_id=payment_id,
                user_id=user.user_id,
                block=block,
                rao_amount=amount,
                usd_amount=delta,
                fmv=fmv,
                transaction_hash=block_hash,
            )
            session.add(payment)

            # Increase user balance: fair market value * amount of rao / 1e9
            user.balance += delta

            # Add in the first payment bonus, if applicable.
            new_total = total_payments + delta
            if (
                not user.bonus_used
                and settings.first_payment_bonus > 0
                and new_total >= settings.first_payment_bonus_threshold
            ):
                logger.success(
                    f"User {user.user_id} total payments has reached ${new_total}, applying first payment bonus!"
                )
                user.balance += settings.first_payment_bonus
                user.bonus_used = True

            # Track new balance for the payment_address.
            await session.execute(
                text(
                    "INSERT INTO wallet_balances (wallet_id, balance) VALUES (:wallet_id, :balance) ON CONFLICT (wallet_id) DO UPDATE SET balance = wallet_balances.balance + EXCLUDED.balance"
                ),
                {"wallet_id": user.payment_address, "balance": amount},
            )
            try:
                await session.commit()
            except IntegrityError as exc:
                if "UniqueViolationError" in str(exc):
                    logger.warning(f"Skipping (apparent) duplicate transaction: {payment_id=}")
                    await session.rollback()
                    return
                else:
                    raise
            logger.success(
                f"Received payment [user_id={user.user_id} username={user.username}]: {amount} rao @ ${fmv} FMV = ${delta} balance increase, updated balance: ${user.balance}"
            )

            # Autostake the payment tao to chutes.
            await stake.kiq(user.user_id)

    async def _handle_developer_deposit(
        self,
        to_address: str,
        from_address: str,
        amount: int,
        block: int,
        block_hash: str,
        fmv: float,
    ):
        """
        Process an incoming transfer to enable developement (create images/chutes).
        """
        async with get_session() as session:
            user = (
                await session.execute(
                    select(User).where(User.developer_payment_address == to_address)
                )
            ).scalar_one_or_none()
            if not user:
                logger.warning(f"Failed to find user with payment address {to_address}")
                return

            # Sum payments prior to this one.
            total_query = select(func.sum(Payment.usd_amount)).where(
                Payment.user_id == user.user_id, Payment.purpose == "developer"
            )
            total_payments = (await session.execute(total_query)).scalar() or 0

            # Store the payment record.
            payment_id = str(
                uuid.uuid5(uuid.NAMESPACE_OID, f"{block}:{to_address}:{from_address}:{amount}")
            )
            delta = amount * fmv / 1e9
            payment = Payment(
                payment_id=payment_id,
                user_id=user.user_id,
                block=block,
                rao_amount=amount,
                usd_amount=delta,
                fmv=fmv,
                transaction_hash=block_hash,
                purpose="developer",
            )
            session.add(payment)

            new_total = total_payments + delta
            if new_total >= settings.developer_deposit:
                logger.success(
                    f"User {user.user_id} total developer deposits has reached ${new_total}, enabling development!"
                )
                Permissioning.enable(user, Permissioning.developer)

            try:
                await session.commit()
            except IntegrityError as exc:
                if "UniqueViolationError" in str(exc):
                    logger.warning(f"Skipping (apparent) duplicate transaction: {payment_id=}")
                    await session.rollback()
                    return
                else:
                    raise
            logger.success(
                f"Received developer deposit [user_id={user.user_id} username={user.username}]: {amount} rao @ ${fmv} FMV = ${delta} deposit, total deposit ${new_total}"
            )

    async def _get_state(self) -> Tuple[int, str]:
        """
        Get current state from database.
        """
        async with get_session() as session:
            result = await session.execute(select(PaymentMonitorState))
            state = result.scalar_one()
            block, hash_ = state.block_number, state.block_hash
            current_block = self.get_latest_block()
            if (delta := current_block - block) > self.max_recovery_blocks:
                block = current_block - self.max_recovery_blocks
                logger.warning(
                    f"Payment watcher is {delta} blocks behind, skipping ahead to {block}..."
                )
                hash_ = self.get_block_hash(block)
            return block, hash_

    async def _save_state(self, block_number: int, block_hash: str):
        """
        Save current state to database.
        """
        async with get_session() as session:
            try:
                await session.execute(
                    update(PaymentMonitorState).values(
                        block_number=block_number,
                        block_hash=block_hash,
                        last_updated_at=func.now(),
                    )
                )
                await session.commit()
            except Exception as e:
                logger.error(f"Error saving state: {e}")
                await session.rollback()

    async def monitor_transfers(self):
        """
        Main monitoring loop.
        """
        logger.info("Starting monitor_transfers loop...")
        self.is_running = True
        try:
            while self.is_running:
                # Make sure we have a process lock.
                if not await self._lock():
                    logger.error("Failed to acquire lock, waiting...")
                    await asyncio.sleep(10)
                    continue

                # Load state.
                current_block_number, current_block_hash = await self._get_state()
                latest_block_number = self.get_latest_block()
                await self._refresh_addresses()
                fetcher = get_fetcher()
                fmv = await fetcher.get_price("tao")

                while self.is_running:
                    # Wait for the block to advance.
                    if current_block_number == latest_block_number:
                        while (
                            self.is_running
                            and (latest_block_number := self.get_latest_block())
                            == current_block_number
                        ):
                            logger.debug("Waiting for next block...")
                            await asyncio.sleep(3)

                        # Update current fair-market value (tao price in USD).
                        fmv = await fetcher.get_price("tao")

                        # Refresh known addresses.
                        await self._refresh_addresses()

                    # Process events.
                    current_block_hash = self.get_block_hash(current_block_number)
                    events = self.get_events(current_block_hash)
                    payments = 0
                    developer_deposits = 0
                    logger.info(f"Processing block {current_block_number}...")
                    for event in events:
                        # event = event.value or {}
                        if (
                            event.get("module_id") != "Balances"
                            or event.get("event_id") != "Transfer"
                            or not event.get("attributes")
                            or {"from", "to", "amount"} - set(event["attributes"])
                        ):
                            continue
                        from_address = event["attributes"]["from"]
                        to_address = event["attributes"]["to"]
                        amount = event["attributes"]["amount"]
                        if to_address in self._payment_addresses:
                            await self._handle_payment(
                                to_address,
                                from_address,
                                amount,
                                current_block_number,
                                current_block_hash,
                                fmv,
                            )
                            payments += 1
                        if to_address in self._developer_payment_addresses:
                            await self._handle_developer_deposit(
                                to_address,
                                from_address,
                                amount,
                                current_block_number,
                                current_block_hash,
                                fmv,
                            )
                            developer_deposits += 1
                    if payments or developer_deposits:
                        logger.success(
                            f"Processed {payments} payment(s), {developer_deposits} deposit(s) in block: {current_block_number}"
                        )

                    # Update state and continue to next block.
                    await self._save_state(current_block_number, current_block_hash)
                    current_block_number += 1
        except Exception as exc:
            logger.error(f"Unexpected error encountered: {exc} -- {traceback.format_exc()}")
            raise
        finally:
            logger.info(f"Releasing process lock: instance_id={self.instance_id}")
            await self._unlock()

    async def stop(self):
        """
        Graceful shutdown.
        """
        self.is_running = False


monitor = PaymentMonitor()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Inside the lifespan...")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Initialized the database...")
    await monitor.initialize()
    monitor_task = asyncio.create_task(monitor.monitor_transfers())
    yield
    await monitor.stop()
    await monitor_task


monitor = PaymentMonitor()
app = FastAPI(lifespan=lifespan)


@app.get("/status")
async def get_status():
    """
    Health check/status endpoint for the payment monitor.
    """
    try:
        async with get_session() as session:
            query = select(
                PaymentMonitorState,
                case(
                    (PaymentMonitorState.is_locked.is_(False), "Process is not locked"),
                    else_=None,
                ).label("lock_status"),
                case(
                    (
                        and_(
                            PaymentMonitorState.is_locked,
                            PaymentMonitorState.lock_holder != monitor.instance_id,
                        ),
                        "Lock held by different instance",
                    ),
                    else_=None,
                ).label("lock_holder_status"),
                case(
                    (
                        PaymentMonitorState.last_updated_at < func.now() - monitor.lock_timeout,
                        "Updates have ceased",
                    ),
                    else_=None,
                ).label("update_status"),
                func.now().label("current_time"),
            )
            result = await session.execute(query)
            row = result.one()
            state = row.PaymentMonitorState
            current_time = row.current_time

            # Check any healthcheck failure conditions.
            failures = [
                status
                for status in [
                    row.lock_status,
                    row.lock_holder_status,
                    row.update_status,
                ]
                if status is not None
            ]

            # Get latest network block for lag calculation
            latest_block = monitor.get_latest_block()
            block_lag = latest_block - state.block_number
            if block_lag > 10:
                failures.append(f"Payment monitor is {block_lag} blocks behind!")
            response_data = {
                "status": "healthy" if not failures else "unhealthy",
                "current_block": state.block_number,
                "latest_network_block": latest_block,
                "block_lag": block_lag,
                "last_updated_at": state.last_updated_at,
                "is_locked": state.is_locked,
                "lock_holder": state.lock_holder,
                "locked_at": state.locked_at,
                "current_time": current_time,
                "failures": failures,
            }
            print(response_data)
            if failures:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=response_data,
                )
            return response_data
    except Exception as e:
        error_response = {
            "status": "unhealthy",
            "error": str(e),
        }
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=error_response
        )
