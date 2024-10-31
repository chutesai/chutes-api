import re
from sqlalchemy import or_
from sqlalchemy.future import select
from run_api.chute.schemas import Chute
from run_api.user.schemas import User


async def get_chute_by_id_or_name(chute_id_or_name, db, current_user):
    """
    Helper to load a chute by ID or full chute name (optional username/chute name)
    """
    name_match = re.match(
        r"(?:([a-z0-9][a-z0-9_-]*)/)?([a-z0-9][a-z0-9_-]*)$",
        chute_id_or_name.lstrip("/"),
        re.I,
    )
    query = (
        select(Chute)
        .join(User, Chute.user_id == User.user_id)
        .where(or_(Chute.public.is_(True), Chute.user_id == current_user.user_id))
    )
    if name_match:
        username = name_match.group(1) or current_user.username
        chute_name = name_match.group(2)
        query = query.where(User.username == username).where(Chute.name == chute_name)
    else:
        query = query.where(Chute.chute_id == chute_id_or_name)
    result = await db.execute(query)
    return result.scalar_one_or_none()
