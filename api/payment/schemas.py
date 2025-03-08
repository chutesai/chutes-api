"""
ORM definitions for payments.
"""

from sqlalchemy.sql import func
from sqlalchemy import (
    Column,
    Boolean,
    BigInteger,
    String,
    DateTime,
    ForeignKey,
    Double,
    Index,
)
from api.database import Base


class Payment(Base):
    __tablename__ = "payments"
    payment_id = Column(String, nullable=False, primary_key=True)
    user_id = Column(String, ForeignKey("users.user_id"), nullable=False)
    block = Column(BigInteger, nullable=False)
    rao_amount = Column(BigInteger, nullable=False)
    fmv = Column(Double, nullable=False)
    usd_amount = Column(Double, nullable=False)
    transaction_hash = Column(String, nullable=False)
    purpose = Column(String, default="credits")
    source_address = Column(String)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        Index(
            "idx_user_id_date_block",
            "user_id",
            "created_at",
            "block",
        ),
    )


class WalletBalance(Base):
    __tablename__ = "wallet_balances"
    wallet_id = Column(String, nullable=False, primary_key=True)
    balance = Column(BigInteger, default=0)


class PaymentMonitorState(Base):
    __tablename__ = "payment_monitor_state"
    instance_id = Column(String, primary_key=True)
    block_number = Column(BigInteger, nullable=False)
    block_hash = Column(String, nullable=False)
    is_locked = Column(Boolean, default=False)
    lock_holder = Column(String)
    locked_at = Column(DateTime)
    last_updated_at = Column(DateTime, default=func.now())


class UsageData(Base):
    __tablename__ = "usage_data"
    user_id = Column(String, ForeignKey("users.user_id"), primary_key=True)
    bucket = Column(DateTime, primary_key=True)
    chute_id = Column(String, primary_key=True)
    amount = Column(Double, nullable=False)
    count = Column(BigInteger, nullable=False)
