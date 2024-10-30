import uuid
from sqlalchemy import event
from run_api.image.schemas import Image


@event.listens_for(Image, "before_insert")
def generate_uid(_, __, image):
    """
    Set the image_id deterministically.
    """
    image.image_id = str(
        uuid.uuid5(uuid.NAMESPACE_OID, f"{image.user_id}/{image.name}:{image.tag}")
    )
