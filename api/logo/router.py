"""
Routes for logos.
"""

import io
import uuid
from loguru import logger
from fastapi import APIRouter, Depends, HTTPException, File, UploadFile, status, Response
from fastapi.responses import RedirectResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from api.config import settings
from api.database import get_db_session
from api.user.schemas import User
from api.user.service import get_current_user
from api.logo.schemas import Logo
from api.logo.util import validate_and_convert_image

router = APIRouter()


@router.post("/")
async def create_logo(
    logo: UploadFile = File(...),
    db: AsyncSession = Depends(get_db_session),
    current_user: User = Depends(get_current_user()),
):
    """
    Create/upload a new logo.
    """
    if not logo.content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded logo is not an image!",
        )

    # Resize, crop, compress, convert to webp.
    image_data, content_type = await validate_and_convert_image(logo)

    # Upload the build context to our S3-compatible storage backend.
    logo_id = str(uuid.uuid4())
    destination = f"logos/{current_user.user_id}/{logo_id}.webp"
    try:
        async with settings.s3_client() as s3:
            await s3.upload_fileobj(
                io.BytesIO(image_data),
                settings.storage_bucket,
                destination,
                ExtraArgs={"ContentType": content_type},
            )
        logger.success(f"Uploaded {logo_id=} to {settings.storage_bucket}/{destination}")
    except HTTPException:
        raise
    except Exception as exc:
        error_message = f"Error processing logo upload: {exc}"
        logger.error(error_message)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=error_message,
        )

    # Store it.
    logo = Logo(logo_id=logo_id, path=destination, user_id=current_user.user_id)
    db.add(logo)
    await db.commit()
    await db.refresh(logo)
    return {
        "logo_id": logo_id,
        "path": f"/logos/{logo_id}.webp",
    }


@router.get("/{logo_id}.{extension}")
async def render_logo(
    logo_id: str, extension: str, db: AsyncSession = Depends(get_db_session)
) -> Response:
    """
    Logo image response.
    """
    if (
        logo := (await db.execute(select(Logo).where(Logo.logo_id == logo_id))).scalar_one_or_none()
    ) is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Logo not found: {logo_id}",
        )

    # Handle legacy png logos also.
    webp_path = f"logos/{logo.user_id}/{logo_id}.webp"
    if logo.path.endswith(".png"):
        data = io.BytesIO()
        async with settings.s3_client() as s3:
            await s3.download_fileobj(settings.storage_bucket, logo.path, data)
        data.seek(0)
        virt_file = UploadFile(filename=f"{logo_id}.png", file=data)
        image_data, content_type = await validate_and_convert_image(virt_file)
        try:
            async with settings.s3_client() as s3:
                await s3.upload_fileobj(
                    io.BytesIO(image_data),
                    settings.storage_bucket,
                    webp_path,
                    ExtraArgs={"ContentType": content_type},
                )
            logger.success(f"Recreated {logo_id=} as webp {settings.storage_bucket}/{webp_path}")
            logo.path = webp_path
            await db.commit()
            await db.refresh(logo)
        except Exception as exc:
            logger.error(f"Error re-creating logo as webp: {exc}")

    # Now download the logo, whether it was png or webp.
    data = io.BytesIO()
    async with settings.s3_client() as s3:
        await s3.download_fileobj(settings.storage_bucket, logo.path, data)
    if not logo.path.endswith(f".{extension}") and extension == "png":
        return RedirectResponse(url=f"/logos/{logo_id}.webp")
    actual_ext = "png" if logo.path.endswith(".png") else "webp"
    return Response(content=data.getvalue(), media_type=f"image/{actual_ext}")
