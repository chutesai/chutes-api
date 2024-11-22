import asyncio
from sqlalchemy import event
from api.image.schemas import Image


@event.listens_for(Image, "after_insert")
def after_insert(_, __, image):
    """
    Trigger image build after insert.
    """
    from api.image.forge import forge

    asyncio.create_task(forge.kiq(image.image_id))
