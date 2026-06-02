"""
SMARTZI — Strict Image Input Validator
Validates file integrity, format, size, and color mode before any
downstream PyTorch or forensic processing touches the image.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Tuple

from PIL import Image

logger = logging.getLogger("smartzi.image_validator")

# ── Constants ─────────────────────────────────────────────────────────────────
ALLOWED_EXTENSIONS: frozenset[str] = frozenset({"jpg", "jpeg", "png"})
MAX_FILE_SIZE_BYTES: int = 10 * 1024 * 1024  # 10 MB
MIN_DIMENSION: int = 16  # reject absurdly small images
MAX_DIMENSION: int = 16_384  # reject absurdly large images (16k px)


class ImageValidationError(ValueError):
    """Raised when an image fails any validation check."""
    pass


def validate_input_image(image_path: str) -> Tuple[Image.Image, dict]:
    """
    Validate that *image_path* is a real, uncorrupted JPEG/PNG image within
    safe size limits, and return a 3-channel RGB PIL Image ready for inference.

    Returns
    -------
    (pil_image, info_dict)
        pil_image : PIL.Image.Image in RGB mode
        info_dict : dict with keys width, height, original_mode, file_size_bytes

    Raises
    ------
    ImageValidationError
        On any validation failure (clear, actionable message).
    FileNotFoundError
        If the file does not exist on disk.
    """
    path = Path(image_path)

    # ── 1. Existence ──────────────────────────────────────────────────────
    if not path.is_file():
        raise FileNotFoundError(
            f"Image file not found: {path}. "
            "Ensure the file exists and the path is correct."
        )

    # ── 2. Extension ──────────────────────────────────────────────────────
    ext = path.suffix.lstrip(".").lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise ImageValidationError("Invalid image format")

    # ── 3. File size ──────────────────────────────────────────────────────
    file_size = path.stat().st_size
    if file_size == 0:
        raise ImageValidationError("Invalid image format")
    if file_size > MAX_FILE_SIZE_BYTES:
        raise ImageValidationError("Image exceeds 10MB limit")

    # ── 4. PIL integrity check (verify pass) ──────────────────────────────
    try:
        with Image.open(path) as img_verify:
            img_verify.verify()  # checks for corruption without full decode
    except Exception as exc:
        raise ImageValidationError("Corrupted image") from exc

    # ── 5. Full open + dimension check ────────────────────────────────────
    #    PIL.verify() consumes the file pointer, so we re-open.
    try:
        img = Image.open(path)
        img.load()  # force full decode to surface late corruption
    except Exception as exc:
        raise ImageValidationError("Image decoding failed") from exc

    width, height = img.size
    if width < MIN_DIMENSION or height < MIN_DIMENSION:
        raise ImageValidationError(
            f"Image dimensions ({width}×{height}) are too small. "
            f"Minimum is {MIN_DIMENSION}×{MIN_DIMENSION} pixels."
        )
    if width > MAX_DIMENSION or height > MAX_DIMENSION:
        raise ImageValidationError(
            f"Image dimensions ({width}×{height}) exceed the "
            f"{MAX_DIMENSION}×{MAX_DIMENSION} pixel limit."
        )

    # ── 6. Color mode normalization ───────────────────────────────────────
    original_mode = img.mode

    if original_mode == "RGBA":
        # Alpha-composite onto white background before converting
        background = Image.new("RGB", img.size, (255, 255, 255))
        background.paste(img, mask=img.split()[3])  # alpha channel as mask
        img = background
        logger.info("Converted RGBA → RGB (alpha composited onto white).")
    elif original_mode in ("L", "1"):
        # Grayscale / binary → 3-channel RGB
        img = img.convert("RGB")
        logger.info("Converted %s → RGB.", original_mode)
    elif original_mode == "P":
        # Palette mode → RGB
        img = img.convert("RGB")
        logger.info("Converted palette (P) → RGB.")
    elif original_mode == "CMYK":
        img = img.convert("RGB")
        logger.info("Converted CMYK → RGB.")
    elif original_mode != "RGB":
        # Catch-all for LA, I, F, etc.
        try:
            img = img.convert("RGB")
            logger.info("Converted %s → RGB.", original_mode)
        except Exception as exc:
            raise ImageValidationError(
                f"Cannot convert color mode '{original_mode}' to RGB: {exc}"
            ) from exc

    info = {
        "width": width,
        "height": height,
        "original_mode": original_mode,
        "file_size_bytes": file_size,
    }

    logger.info(
        "Image validated: %s | %d×%d | %s → RGB | %.1f KB",
        path.name, width, height, original_mode, file_size / 1024,
    )

    return img, info
