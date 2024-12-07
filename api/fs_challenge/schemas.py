"""
Filesystem challenge schemas.
"""

from sqlalchemy import (
    Column,
    String,
    ForeignKey,
    BigInteger,
    Index,
    Integer,
)
from sqlalchemy.orm import relationship
from api.database import Base


class FSChallenge(Base):
    __tablename__ = "fs_challenges"
    challenge_id = Column(String, primary_key=True)
    image_id = Column(String, ForeignKey("images.image_id", ondelete="CASCADE"), nullable=False)
    challenge_type = Column(String, nullable=False)
    filename = Column(String, nullable=False)
    offset = Column(BigInteger, nullable=False)
    length = Column(Integer, nullable=False)
    expected = Column(String, nullable=False)

    image = relationship("Image", back_populates="fs_challenges")

    _table_args__ = (Index("idx_image_id_type", "image_id", "challenge_type"),)
