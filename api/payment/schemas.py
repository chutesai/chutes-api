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
)
from api.database import Base


class PayoutReason(Enum):
    MINER = "miner"
    MAINTAINER = "maintainer"
    MODERATOR = "moderator"
    CONTRIBUTOR = "contributor"
    IMAGE_CREATOR = "image_creator"
    CHUTE_CREATOR = "chute_creator"


class WalletPurpose(Enum):
    CONTRIBUTOR = "contributor"
    GENERAL = "general"


class EpochStatus(Enum):
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class PaymentEpoch(Base):
    __tablename__ = "payment_epochs"
    epoch_id = Column(String, primary_key=True)
    last_processed_timestamp = Column(DateTime(timezone=True), nullable=False)
    start_block = Column(BigInteger, nullable=False)
    end_block = Column(BigInteger, nullable=False)
    processed_at = Column(DateTime(timezone=True), server_default=func.now())
    status = Column(Enum(EpochStatus), nullable=False)

    __table_args__ = (Index("idx_last_processed", "last_processed_timestamp"),)


class Payment(Base):
    __tablename__ = "payments"
    payment_id = Column(String, nullable=False, primary_key=True)
    payment_block = Column(BigInteger, nullable=False)
    recipient_address = Column(String, nullable=False)
    sending_address = Column(String, nullable=False)
    amount = Column(BigInteger, nullable=False)  # RAO
    transaction_hash = Column(String, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        Index(
            "idx_recipient_date",
            "recipient_address",
            "created_at",
        ),
    )


class Transaction(Base):
    __tablename__ = "transactions"
    transaction_id = Column(String, nullable=False, primary_key=True)
    recipient_address = Column(String, nullable=False)
    sending_address = Column(String, nullable=False)
    reason = Column(Enum(PayoutReason), nullable=False)
    amount = Column(BigInteger, nullable=False)  # RAO
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class WalletBalance(Base):
    __tablename__ = "wallet_balances"
    wallet_id = Column(String, nullable=False, primary_key=True)
    purpose = Column(Enum(WalletPurpose), default=None)
    balance = Column(BigInteger, default=0)
