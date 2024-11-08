"""
Node event listeners.
"""

from sqlalchemy import event
from api.node.schemas import Node


@event.listens_for(Node, "before_insert")
def validate_before_insert(mapper, connection, node):
    """
    Verify GPU specs.
    """
    node._validate_all()
