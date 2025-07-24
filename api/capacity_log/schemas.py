from sqlalchemy import Column, String, Float, Integer, DateTime, Index
from api.database import Base


class CapacityLog(Base):
    __tablename__ = "capacity_log"

    entry_id = Column(String, primary_key=True)
    chute_id = Column(String, nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, index=True)

    # Utilization metrics
    utilization_current = Column(Float, nullable=True)
    utilization_5m = Column(Float, nullable=True)
    utilization_15m = Column(Float, nullable=True)
    utilization_1h = Column(Float, nullable=True)

    # Rate limiting ratios
    rate_limit_ratio_5m = Column(Float, nullable=True)
    rate_limit_ratio_15m = Column(Float, nullable=True)
    rate_limit_ratio_1h = Column(Float, nullable=True)

    # Request counts
    total_requests_5m = Column(Float, nullable=True)
    total_requests_15m = Column(Float, nullable=True)
    total_requests_1h = Column(Float, nullable=True)
    completed_requests_5m = Column(Float, nullable=True)
    completed_requests_15m = Column(Float, nullable=True)
    completed_requests_1h = Column(Float, nullable=True)
    rate_limited_requests_5m = Column(Float, nullable=True)
    rate_limited_requests_15m = Column(Float, nullable=True)
    rate_limited_requests_1h = Column(Float, nullable=True)

    # Instance info
    instance_count = Column(Integer, nullable=True)

    # Action taken
    action_taken = Column(String, nullable=True)

    # Number of additional instances allowed to scale.
    num_to_add = Column(Integer, nullable=True)

    # Index on the timestamp and chute_id.
    __table_args__ = (Index("idx_capacity_log_chute_timestamp", "chute_id", "timestamp"),)
