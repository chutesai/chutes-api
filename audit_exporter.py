"""
Export audit information from our validator.

Information collected:
  - deployment audit information (instance_audit), so we can cross-check against miner self-reports
  - compute multiplier history (instance_compute_history), for time-weighted scoring

The information is then hashed, the hash is committed to chain via set_commitment,
and the full payload is uploaded to blob store.
"""

import io
import json
import uuid
import backoff
import hashlib
import asyncio
from loguru import logger
from sqlalchemy import text
from datetime import UTC, datetime, timedelta
from substrateinterface import SubstrateInterface
from api.config import settings
from api.database import get_session
from api.audit.schemas import AuditEntry
import api.database.orms  # noqa


COMPUTE_HISTORY_QUERY = text(
    """
SELECT
    instance_id,
    compute_multiplier,
    started_at,
    ended_at
FROM instance_compute_history
WHERE ended_at IS NULL
   OR (ended_at >= :start_time AND ended_at <= :end_time)
   OR (started_at >= :start_time AND started_at <= :end_time)
   OR (started_at < :start_time AND (ended_at IS NULL OR ended_at > :end_time))
ORDER BY instance_id, started_at
"""
)


async def get_compute_history(start_time, end_time) -> list:
    """
    Get compute multiplier history for instances.

    This captures all history records that overlap with the time window,
    allowing accurate time-weighted scoring calculations.
    """
    async with get_session() as session:
        result = await session.execute(
            COMPUTE_HISTORY_QUERY,
            {
                "start_time": start_time.replace(tzinfo=None),
                "end_time": end_time.replace(tzinfo=None),
            },
        )
        results = [dict(row._mapping) for row in result]
        for item in results:
            for key in item:
                if isinstance(item[key], datetime):
                    item[key] = item[key].isoformat()
        return results


async def get_instance_audit(start_time, end_time) -> list:
    """
    Get deployment/instance audit information.

    Filtering here is just based on having a deleted_at timestamp of null or within our start/end time.
    - if the deployment is not deleted, then it should be included since it's either running
      or pending validation.
    - if it's deleted, we only need to include it in the audit result for this time bucket, otherwise
      it's part of a different audit entry.
    """
    async with get_session() as session:
        query = text(
            """
           SELECT * FROM instance_audit
            WHERE deleted_at IS NULL OR (deleted_at >= :start_time AND deleted_at <= :end_time)
        """
        )
        result = await session.execute(
            query,
            {
                "start_time": start_time.replace(tzinfo=None),
                "end_time": end_time.replace(tzinfo=None),
            },
        )
        results = [dict(row._mapping) for row in result]
        for item in results:
            for key in item:
                if isinstance(item[key], datetime):
                    item[key] = item[key].isoformat()
        return results


@backoff.on_exception(
    backoff.constant,
    Exception,
    jitter=None,
    interval=10,
    max_tries=6,
)
async def upload_report(report_data, block_number):
    """
    Upload the combined audit data to blob storage.
    """
    destination = f"audit/validator/{settings.validator_ss58}/{block_number}.json"
    async with settings.s3_client() as s3:
        await s3.upload_fileobj(
            io.BytesIO(report_data),
            settings.storage_bucket,
            destination,
            ExtraArgs={"ContentType": "application/json"},
        )
        logger.success(f"Uploaded audit data to: {destination}")
        return destination


@backoff.on_exception(
    backoff.constant,
    Exception,
    jitter=None,
    interval=10,
    max_tries=12,
)
def commit(sha256) -> int:
    """
    Commit this bucket of audit data to chain.
    """
    substrate = SubstrateInterface(url=settings.subtensor)
    call = substrate.compose_call(
        call_module="Commitments",
        call_function="set_commitment",
        call_params={"netuid": settings.netuid, "info": {"fields": [[{"Sha256": f"0x{sha256}"}]]}},
    )
    extrinsic = substrate.create_signed_extrinsic(
        call=call,
        keypair=settings.validator_keypair,
    )
    response = substrate.submit_extrinsic(
        extrinsic=extrinsic,
        wait_for_inclusion=True,
        wait_for_finalization=False,
    )
    response.process_events()
    assert response.is_success
    block_hash = response.block_hash
    block_number = substrate.get_block_number(block_hash)
    logger.success(f"Committed checksum {sha256} in block {block_number}")
    return block_number


async def main():
    """
    Do all the exporty things.
    """
    end_time = datetime.now(UTC).replace(minute=0, second=0, microsecond=0)
    start_time = end_time - timedelta(hours=1)
    logger.info(
        f"Generating audit information for time range: {start_time.isoformat()} through {end_time.isoformat()}"
    )

    instance_audit = await get_instance_audit(start_time, end_time)
    compute_history = await get_compute_history(start_time, end_time)
    report_data = json.dumps(
        {
            "instance_audit": instance_audit,
            "compute_history": compute_history,
        }
    ).encode()
    sha256 = hashlib.sha256(report_data).hexdigest()

    block_number = commit(sha256)
    report_path = await upload_report(report_data, block_number)

    entry_id = str(uuid.uuid5(uuid.NAMESPACE_OID, f"{settings.validator_ss58}.{block_number}"))
    async with get_session() as session:
        audit_entry = AuditEntry(
            entry_id=entry_id,
            hotkey=settings.validator_ss58,
            block=block_number,
            path=report_path,
            start_time=start_time.replace(tzinfo=None),
            end_time=end_time.replace(tzinfo=None),
        )
        session.add(audit_entry)
        await session.commit()
        await session.refresh(audit_entry)
    logger.success(f"Completed audit report generation/commit: {block_number=} {entry_id=}")


if __name__ == "__main__":
    asyncio.run(main())
