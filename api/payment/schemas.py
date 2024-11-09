"""
ORM definitions for payments.
"""

from sqlalchemy.sql import func
from sqlalchemy import (
    Column,
    Enum,
    BigInteger,
    String,
    DateTime,
    Index,
    UniqueConstraint,
)
from api.database import Base


class PayoutReason(Enum):
    MINER = "miner"
    MAINTAINER = "maintainer"
    MODERATOR = "moderator"
    CONTRIBUTOR = "contributor"
    IMAGE_CREATOR = "image_creator"
    CHUTE_CREATOR = "chute_creator"


class PayoutStatus(Enum):
    PENDING = "pending"
    PAID = "paid"
    CREDITED = "credited"


class Payment(Base):
    __tablename__ = "payments"
    payment_id = Column(String, nullable=False, primary_key=True)
    payment_block = Column(BigInteger, nullable=False, index=True)
    recipient_address = Column(String, nullable=False, index=True)
    sending_address = Column(String, nullable=False, index=True)
    reason = Column(Enum(PayoutReason), nullable=False, index=True)
    status = Column(Enum(PayoutStatus), nullable=False, index=True)
    amount = Column(BigInteger, nullable=False)  # RAO
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        Index(
            "idx_reason_status_block_created_at",
            "reason",
            "status",
            "block",
            "created_at",
        ),
        UniqueConstraint("name", "tag", name="constraint_image_name_tag"),
    )
