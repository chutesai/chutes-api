import uuid
from sqlalchemy import event
from run_api.chute.schemas import Chute


@event.listens_for(Chute, "before_insert")
def generate_uid(_, __, chute):
    """
    Set the chute_id deterministically.
    """
    chute.chute_id = str(
        uuid.uuid5(uuid.NAMESPACE_OID, f"{chute.user_id}::chute::{chute.name}")
    )
