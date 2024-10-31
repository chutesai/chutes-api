import asyncio
import uuid
from sqlalchemy import event
from run_api.image.schemas import Image
from run_api.image.forge import forge


@event.listens_for(Image, "before_insert")
def generate_uid(_, __, image):
    """
    Set the image_id deterministically.
    """
    image.image_id = str(
        uuid.uuid5(uuid.NAMESPACE_OID, f"{image.user_id}/{image.name}:{image.tag}")
    )


@event.listens_for(Image, "after_insert")
def after_insert(_, __, image):
    """
    Trigger image build after insert.
    """
    asyncio.create_task(forge.kiq(image.image_id))
