"""
Helper functions for instances.
"""

from api.instance.schemas import Instance
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy import func, case
from sqlalchemy.orm import aliased
from datetime import datetime

# Define an alias for the Instance model to use in a subquery
InstanceAlias = aliased(Instance)


async def discover_chute_targets(session: AsyncSession, chute_id: str):
    """
    Fancy query to attempt evenly distributing queries based on coldkey and
    last invocation timestamp.
    """
    subquery = (
        select(InstanceAlias.miner_coldkey, func.count().label("instance_count"))
        .where(InstanceAlias.active.is_(True), InstanceAlias.chute_id == chute_id)
        .group_by(InstanceAlias.miner_coldkey)
        .subquery()
    )

    # Main query to fetch instances filtered by chute_id and ordered as needed
    query = (
        select(Instance)
        .join(subquery, Instance.miner_coldkey == subquery.c.miner_coldkey)
        .where(Instance.active.is_(True), Instance.chute_id == chute_id)
        .order_by(
            subquery.c.instance_count,
            case(
                (Instance.last_queried_at.is_(None), datetime.min),
                else_=Instance.last_queried_at,
            ).asc(),
        )
        .limit(3)
    )

    # Execute the query asynchronously
    result = await session.execute(query)
    instances = result.scalars().all()
    return instances
