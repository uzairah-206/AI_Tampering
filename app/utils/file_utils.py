"""
SMARTZI - File Utility Helpers
Safe file operations used across the application.
"""

import hashlib
import os
import logging
from typing import Optional

logger = logging.getLogger("smartzi.utils.file")


def compute_md5(file_path: str) -> str:
    """Compute MD5 hash of a file (used for deduplication)."""
    h = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_delete(file_path: str) -> bool:
    """Delete a file silently; returns True if deleted, False otherwise."""
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.debug("Deleted temp file: %s", file_path)
            return True
    except OSError as e:
        logger.warning("Could not delete %s: %s", file_path, e)
    return False


def get_extension(filename: str) -> str:
    """Return the lowercase file extension without leading dot."""
    return filename.rsplit(".", 1)[-1].lower() if "." in filename else ""


def human_readable_size(size_bytes: int) -> str:
    """Convert bytes to a human-readable string."""
    for unit in ("B", "KB", "MB", "GB"):
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024  # type: ignore
    return f"{size_bytes:.1f} TB"
