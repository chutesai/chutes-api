"""
Partner integration endpoints.

Currently only Hugging Face, which requires providers to expose a billing
callback so HF can charge its users the provider's real per-request cost:
https://huggingface.co/docs/inference-providers/en/register-as-a-provider#4-billing
"""

import math
from typing import Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from api.database import get_db_session
from api.user.schemas import User
from api.user.service import get_current_user

router = APIRouter()

# HF sends batches of up to 10,000 request IDs per call.
MAX_REQUEST_IDS = 10000

# HF gives up on a request roughly 30 minutes after it was served, so there is no
# point scanning further back than that (and it keeps the index range tight).
LOOKBACK_INTERVAL = "2 hours"

NANO_USD = 1_000_000_000


class BillingRequest(BaseModel):
    requestIds: list[str] = Field(default_factory=list)


class BillingEntry(BaseModel):
    requestId: str
    costNanoUsd: int


class BillingResponse(BaseModel):
    requests: Optional[list[BillingEntry]]


@router.post("/huggingface/billing", response_model=BillingResponse)
async def huggingface_billing(
    body: BillingRequest,
    db: AsyncSession = Depends(get_db_session),
    current_user: User = Depends(get_current_user()),
):
    """
    Return the cost, in nano-USD, of previously served invocations.

    Hugging Face polls this once a minute with the IDs it routed to us, using the
    same bearer auth as inference. Costs are read from the per-invocation billed
    balance already persisted in `invocations.metrics->>'b'` (USD), so this
    reports exactly what the user was charged — no separate pricing path to keep
    in sync.

    Two behaviours are load-bearing and come straight from HF's spec:

    * Unknown IDs are OMITTED, never returned as 0. HF replaces its placeholder
      cost with whatever we return and never asks again, so answering 0 for an
      invocation that simply hasn't been written yet would bill it as free,
      permanently. Omitted IDs get retried every minute for ~30 minutes.
    * A batch we know nothing about returns `{"requests": null}`, HF's documented
      "no data yet, retry later" signal.

    Scoped to the authenticated user so one partner key can never read another
    account's costs.
    """
    request_ids = list(dict.fromkeys(body.requestIds or []))[:MAX_REQUEST_IDS]
    if not request_ids:
        return BillingResponse(requests=None)

    result = await db.execute(
        text(f"""
SELECT
    invocation_id,
    (metrics->>'b')::float AS billed_usd
FROM invocations
WHERE
    invocation_id = ANY(:request_ids)
    AND user_id = :user_id
    AND completed_at >= NOW() - INTERVAL '{LOOKBACK_INTERVAL}'
    AND metrics->>'b' IS NOT NULL
"""),
        {"request_ids": request_ids, "user_id": current_user.user_id},
    )

    entries = []
    for row in result.fetchall():
        if row.billed_usd is None or row.billed_usd < 0:
            # Unparseable or negative: omit so HF retries rather than locking in
            # a value it would otherwise treat as final.
            continue
        # HF requires a non-negative integer and rounds non-integers up, so do it
        # here explicitly. 0 is valid and means "served for free".
        entries.append(
            BillingEntry(
                requestId=row.invocation_id,
                costNanoUsd=math.ceil(row.billed_usd * NANO_USD),
            )
        )

    return BillingResponse(requests=entries or None)
