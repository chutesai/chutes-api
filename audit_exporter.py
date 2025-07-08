"""
Export audit/invocation information from our validator.

Information collected:
  - invocation counts/duration
  - user-generated reports
  - deployment audit information, so we can cross-check against miner self-reports
  - etc.

The information is then signed with our hotkey and uploaded to blob store, and a sha256 of the report data is committed to chain.
"""

import io
import csv
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


INVOCATION_QUERY = text(
    """
SELECT
     parent_invocation_id,
     invocation_id,
     chute_id,
     chute_user_id,
     function_name,
     image_id,
     image_user_id,
     instance_id,
     miner_uid,
     miner_hotkey,
     started_at,
     completed_at,
     error_message,
     compute_multiplier,
     bounty,
     metrics
 FROM invocations
WHERE (started_at >= :start_time AND started_at < :end_time) OR (started_at >= :start_time - INTERVAL '1 day' AND completed_at >= :start_time AND completed_at < :end_time)
 ORDER BY started_at ASC
"""
)
REPORT_QUERY = text(
    """
SELECT *
  FROM reports
 WHERE (timestamp >= :start_time AND timestamp < :end_time) OR (confirmed_at >= :start_time AND confirmed_at < :end_time)
 ORDER BY timestamp ASC
"""
)


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


def get_sha256(path):
    """
    Calculate sha256 of file.
    """
    sha256_hash = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256_hash.update(chunk)
    return sha256_hash.hexdigest()


@backoff.on_exception(
    backoff.constant,
    Exception,
    jitter=None,
    interval=10,
    max_tries=6,
)
async def generate_invocation_report_data(start_time, end_time) -> dict:
    """
    Export all invocation and report data for this time slice to CSV file,
    tracking the blob storage paths and checksums.
    """
    async with get_session() as session:
        for type_, query in (("invocations", INVOCATION_QUERY), ("reports", REPORT_QUERY)):
            result = await session.stream(
                query,
                {
                    "start_time": start_time.replace(tzinfo=None),
                    "end_time": end_time.replace(tzinfo=None),
                },
            )
            with open(f"/tmp/{type_}.csv", "w") as outfile:
                writer = csv.writer(outfile)
                writer.writerow(result.keys())
                async for row in result:
                    writer.writerow(row)

    # Upload both to S3.
    year = end_time.strftime("%Y")
    month = end_time.strftime("%m")
    day = end_time.strftime("%d")
    hour = end_time.strftime("%H")
    base_path = f"invocations/{year}/{month}/{day}/{hour}"
    paths = {
        "invocations": f"{base_path}.csv",
        "reports": f"{base_path}-reports.csv",
    }
    async with settings.s3_client() as s3:
        for type_, destination in paths.items():
            await s3.upload_file(
                f"/tmp/{type_}.csv",
                settings.storage_bucket,
                destination,
                ExtraArgs={"ContentType": "text/csv"},
            )
            logger.success(f"Uploaded {type_} CSV to: {destination}")
    result = {
        type_: {
            "path": path,
            "sha256": get_sha256(f"/tmp/{type_}.csv"),
        }
        for type_, path in paths.items()
    }
    return result


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
    invocation_reports = await generate_invocation_report_data(start_time, end_time)
    report_data = json.dumps(
        {
            "instance_audit": instance_audit,
            "csv_exports": invocation_reports,
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
