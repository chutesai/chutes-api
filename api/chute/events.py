"""
Event listeners for chutes.
"""

import uuid
import hashlib
from sqlalchemy import event
from api.chute.schemas import Chute


@event.listens_for(Chute, "before_insert")
def generate_uuid(_, __, chute):
    """
    Set chute_id deterministically.
    """
    chute.chute_id = str(uuid.uuid5(uuid.NAMESPACE_OID, f"{chute.user_id}::chute::{chute.name}"))


@event.listens_for(Chute, "before_insert")
def set_version(_, __, chute):
    """
    Set the version (sha256 of the code).
    """
    chute.version = hashlib.sha256(chute.code.encode()).hexdigest()
