"""
User event listeners.
"""

import uuid
from sqlalchemy import event
from run_api.user.schemas import User


@event.listens_for(User, "before_insert")
def generate_uid(_, __, user):
    """
    Set the user_id based on hotkey.
    """
    user.user_id = str(uuid.uuid5(uuid.NAMESPACE_OID, user.hotkey))
