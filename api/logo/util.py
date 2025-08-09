"""
Helpers/utility functions for logos.
"""

import io
from PIL import Image
from typing import Tuple
from fastapi import HTTPException, UploadFile, status


async def validate_and_convert_image(file: UploadFile) -> Tuple[bytes, str]:
    """
    Validates that the uploaded file is an image and converts it to WebP format.
    """
    content = await file.read()
    try:
        image = Image.open(io.BytesIO(content))

        # Handle transparency
        if image.mode in ("RGBA", "LA"):
            background = Image.new("RGB", image.size, (255, 255, 255))
            background.paste(image, mask=image.split()[-1])
            image = background
        elif image.mode != "RGB":
            image = image.convert("RGB")

        # Center crop to square.
        width, height = image.size
        size = min(width, height)
        left = (width - size) // 2
        top = (height - size) // 2
        right = left + size
        bottom = top + size
        image = image.crop((left, top, right, bottom))
        image = image.resize((512, 512), Image.Resampling.LANCZOS)

        # Save as WebP
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format="WEBP", quality=85, method=6, lossless=False, exact=False)
        img_byte_arr.seek(0)
        return img_byte_arr.getvalue(), "image/webp"

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid image file: {str(e)}"
        )
