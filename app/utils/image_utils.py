"""
SMARTZI - Image Validation Utilities
Pre-flight checks before passing files to the AI pipeline.
"""

import logging
from typing import Tuple
from PIL import Image, UnidentifiedImageError

logger = logging.getLogger("smartzi.utils.image")

# Minimum dimensions for meaningful ELA analysis
MIN_WIDTH = 64
MIN_HEIGHT = 64

# Maximum dimensions before mandatory resize (avoids OOM on free tier)
MAX_DIMENSION = 4096


def validate_image(file_path: str) -> Tuple[bool, str]:
    """
    Validate that the file is a real, readable image of acceptable size.

    Returns:
        (True, "") on success
        (False, reason) on failure
    """
    try:
        img = Image.open(file_path)
        img.verify()  # Check file integrity (does not decode pixels)
    except UnidentifiedImageError:
        return False, "File is not a recognisable image format."
    except Exception as e:
        return False, f"Image validation failed: {e}"

    # Re-open after verify (verify() invalidates the object)
    try:
        img = Image.open(file_path)
        w, h = img.size
    except Exception as e:
        return False, f"Cannot read image dimensions: {e}"

    if w < MIN_WIDTH or h < MIN_HEIGHT:
        return False, f"Image too small ({w}×{h}). Minimum: {MIN_WIDTH}×{MIN_HEIGHT}."

    if w > MAX_DIMENSION or h > MAX_DIMENSION:
        logger.warning("Large image %dx%d — will be processed but may be slow", w, h)

    return True, ""
