"""
ORM definitions for images.
"""

import re
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship, validates
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy import (
    Column,
    String,
    DateTime,
    Boolean,
    Index,
    ForeignKey,
    UniqueConstraint,
    event,
)
from api.database import Base
from api.log import image_logger, LifecycleEvent


class Image(Base):
    __tablename__ = "images"
    image_id = Column(String, primary_key=True, default="replaceme")
    user_id = Column(String, ForeignKey("users.user_id"), nullable=False)
    name = Column(String, nullable=False)
    tag = Column(String, nullable=False)
    readme = Column(String, nullable=True, default="")
    logo_id = Column(String, ForeignKey("logos.logo_id", ondelete="SET NULL"), nullable=True)
    public = Column(Boolean, default=False)
    status = Column(String, default="pending build")
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    chutes_version = Column(String, nullable=True)
    patch_version = Column(String, nullable=True, default="initial")
    build_started_at = Column(DateTime(timezone=True))
    build_completed_at = Column(DateTime(timezone=True))
    inspecto = Column(String, nullable=True)
    package_hashes = Column(JSONB, nullable=True)

    chutes = relationship("Chute", back_populates="image")
    logo = relationship("Logo", back_populates="images", lazy="joined")
    user = relationship("User", back_populates="images", lazy="joined")

    __table_args__ = (
        Index("idx_image_name_tag", "name", "tag"),
        Index("idx_name_public", "name", "public"),
        Index("idx_name_created_at", "name", "created_at"),
        UniqueConstraint("user_id", "name", "tag", name="constraint_user_id_image_name_tag"),
    )

    @validates("name")
    def validate_name(self, _, name):
        """
        Basic validation on image name.
        """
        if not isinstance(name, str) or not re.match(r"^[a-z0-9][a-z0-9_\.\/-]{2,64}$", name, re.I):
            raise ValueError(f"Invalid image name: {name}")
        return name

    @validates("tag")
    def validate_tag(self, _, tag):
        """
        Basic validation on image tag.
        """
        if not isinstance(tag, str) or not re.match(r"^[a-z0-9][a-z0-9_\.\/-]{1,32}$", tag, re.I):
            raise ValueError(f"Invalid image tag: {tag}")
        return tag


class ImageHistory(Base):
    __tablename__ = "image_history"
    entry_id = Column(String, primary_key=True)
    image_id = Column(String, nullable=False)
    user_id = Column(String, nullable=False)
    name = Column(String, nullable=False)
    tag = Column(String, nullable=False)
    readme = Column(String, nullable=True, default="")
    logo_id = Column(String)
    public = Column(Boolean, default=False)
    status = Column(String, default="pending build")
    created_at = Column(DateTime)
    deleted_at = Column(DateTime)
    chutes_version = Column(String, nullable=True)
    build_started_at = Column(DateTime(timezone=True))
    build_completed_at = Column(DateTime(timezone=True))


# ---------------------------------------------------------------------------
# Image build (forge) lifecycle logging -- same ORM-event approach as instances/launch configs, so
# the build stage of a chute's lifecycle is searchable in OpenSearch by image_id (a chute ties to
# its build via chute.image_id). Build state is driven by the `status` column, set via ORM in both
# api/image/forge.py and api/image/remote_forge.py:
#   pending build -> building -> "built and pushed" | "error: <detail>"
# The "pending build" default is applied at INSERT (not an attribute 'set'), so image creation is
# captured by after_insert; subsequent transitions by the status 'set' event.
# ---------------------------------------------------------------------------

_BUILD_STATUS_EVENTS = {
    "building": LifecycleEvent.IMAGE_BUILD_START,
    "built and pushed": LifecycleEvent.IMAGE_BUILD_COMPLETE,
}


def _build_event_for_status(status: str) -> LifecycleEvent:
    if status in _BUILD_STATUS_EVENTS:
        return _BUILD_STATUS_EVENTS[status]
    if status and status.lower().startswith("error"):
        return LifecycleEvent.IMAGE_BUILD_FAIL
    return LifecycleEvent.IMAGE_BUILD_STATUS


@event.listens_for(Image, "after_insert")
def _on_image_created(mapper, connection, target):
    image_logger(target, event=LifecycleEvent.IMAGE_CREATE).info(
        f"image created (build queued): {target.image_id} "
        f"({target.name}:{target.tag}, user {target.user_id})"
    )


@event.listens_for(Image.status, "set")
def _on_image_status_changed(target, value, oldvalue, initiator):
    # Fires on ORM status assignment (not on load), covering both forge and remote_forge builders.
    if not value or value == oldvalue:
        return
    build_event = _build_event_for_status(value)
    log = image_logger(target, event=build_event, build_status=value)
    message = f"image build status -> {value}: {target.image_id} ({target.name}:{target.tag})"
    if build_event == LifecycleEvent.IMAGE_BUILD_FAIL:
        log.warning(message)
    elif build_event == LifecycleEvent.IMAGE_BUILD_COMPLETE:
        log.success(message)
    else:
        log.info(message)
