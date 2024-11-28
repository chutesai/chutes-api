"""
Helpers/utility functions for logos.
"""

import io
from PIL import Image
from typing import Tuple
from fastapi import HTTPException, UploadFile, status


async def validate_and_convert_image(file: UploadFile) -> Tuple[bytes, str]:
    """
    Validates that the uploaded file is an image and converts it to PNG format.
    """
    content = await file.read()
    try:
        image = Image.open(io.BytesIO(content))
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
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format="PNG", optimize=True)
        img_byte_arr.seek(0)
        return img_byte_arr.getvalue(), "image/png"
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid image file: {str(e)}"
        )
