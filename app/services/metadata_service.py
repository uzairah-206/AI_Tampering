"""
SMARTZI - Metadata Extraction Service
Extracts EXIF and file-level metadata from uploaded images.
"""

import logging
import os
from typing import Dict, Any, Optional, Tuple
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
import piexif

from app.schemas.analysis import ImageMetadata

logger = logging.getLogger("smartzi.metadata")


class MetadataExtractor:
    """
    Responsible for extracting rich metadata from image files.
    Handles EXIF, GPS, and basic file-level attributes.
    """

    # ── Public API ────────────────────────────────────────────────────────────
    def extract(self, image_path: str) -> Any:
        """
        Entry point: extract all metadata from a local image file.
        Returns an ImageMetadata dataclass populated with available fields.
        """
        filename = os.path.basename(image_path)

        if not os.path.exists(image_path):
            class SafeErrorResponse(dict):
                def __init__(self, fname):
                    super().__init__({"status": "error", "message": "file not found"})
                    self.filename = fname
                    self.file_size_kb = 0.0
                    self.width = None
                    self.height = None
                    self.mode = None
                    self.camera_make = None
                    self.camera_model = None
                    self.date_taken = None
                    self.gps_latitude = None
                    self.gps_longitude = None
                    self.software = None
                    self.orientation = None
                    self.iso = None
                    self.shutter_speed = None
                    self.aperture = None
                    self.focal_length = None
                    self.flash = None
                    self.color_space = None
                    self.has_exif = False
                    self.raw_exif = {}
                def __getattr__(self, name):
                    try:
                        return self[name]
                    except KeyError:
                        return getattr(super(), name)
            return SafeErrorResponse(filename)

        file_size_kb = round(os.path.getsize(image_path) / 1024, 2)

        try:
            with Image.open(image_path) as img:
                # Basic image attributes
                base_meta: Dict[str, Any] = {
                    "filename": filename,
                    "file_size_kb": file_size_kb,
                    "format": img.format,
                    "width": img.width,
                    "height": img.height,
                    "mode": img.mode,
                }

                # EXIF extraction
                exif_data = self._extract_exif(img)
                gps_lat, gps_lon = self._extract_gps(exif_data.get("GPSInfo", {}))

        except Exception as e:
            logger.error("Cannot open image %s: %s", image_path, e)
            return ImageMetadata(filename=filename, file_size_kb=file_size_kb)

        return ImageMetadata(
            **base_meta,
            camera_make=exif_data.get("Make"),
            camera_model=exif_data.get("Model"),
            date_taken=exif_data.get("DateTimeOriginal") or exif_data.get("DateTime"),
            gps_latitude=gps_lat,
            gps_longitude=gps_lon,
            software=exif_data.get("Software"),
            orientation=exif_data.get("Orientation"),
            iso=self._safe_int(exif_data.get("ISOSpeedRatings")),
            shutter_speed=self._ratio_to_str(exif_data.get("ExposureTime")),
            aperture=self._ratio_to_str(exif_data.get("FNumber")),
            focal_length=self._ratio_to_str(exif_data.get("FocalLength")),
            flash=str(exif_data.get("Flash")) if exif_data.get("Flash") is not None else None,
            color_space=str(exif_data.get("ColorSpace")),
            has_exif=bool(exif_data),
            raw_exif={k: str(v) for k, v in exif_data.items() if k != "GPSInfo"},
        )

    # ── Private Helpers ───────────────────────────────────────────────────────
    def _extract_exif(self, img: Image.Image) -> Dict[str, Any]:
        """Parse raw EXIF bytes into a human-readable dictionary."""
        result: Dict[str, Any] = {}
        try:
            raw = img._getexif()  # type: ignore
            if raw is None:
                return result
            for tag_id, value in raw.items():
                tag = TAGS.get(tag_id, tag_id)
                result[tag] = value
        except Exception as e:
            logger.debug("EXIF extraction skipped: %s", e)
        return result

    def _extract_gps(self, gps_info: Any) -> Tuple[Optional[float], Optional[float]]:
        """Convert raw GPSInfo into decimal (lat, lon) pair."""
        if not gps_info or not isinstance(gps_info, dict):
            return None, None
        try:
            gps_tags: Dict[str, Any] = {}
            for key, val in gps_info.items():
                name = GPSTAGS.get(key, key)
                gps_tags[name] = val

            lat = self._dms_to_decimal(
                gps_tags.get("GPSLatitude"), gps_tags.get("GPSLatitudeRef", "N")
            )
            lon = self._dms_to_decimal(
                gps_tags.get("GPSLongitude"), gps_tags.get("GPSLongitudeRef", "E")
            )
            return lat, lon
        except Exception as e:
            logger.debug("GPS parsing failed: %s", e)
            return None, None

    @staticmethod
    def _dms_to_decimal(dms: Any, ref: str) -> Optional[float]:
        """Convert degrees/minutes/seconds tuple to decimal degrees."""
        if dms is None:
            return None
        try:
            d, m, s = dms
            decimal = float(d) + float(m) / 60 + float(s) / 3600
            if ref in ("S", "W"):
                decimal *= -1
            return round(decimal, 6)
        except Exception:
            return None

    @staticmethod
    def _ratio_to_str(value: Any) -> Optional[str]:
        """Convert IFDRational or tuple to readable string."""
        if value is None:
            return None
        try:
            if hasattr(value, "numerator") and hasattr(value, "denominator"):
                return f"{value.numerator}/{value.denominator}"
            if isinstance(value, tuple) and len(value) == 2:
                return f"{value[0]}/{value[1]}"
            return str(value)
        except Exception:
            return str(value)

    @staticmethod
    def _safe_int(value: Any) -> Optional[int]:
        """Safely cast to int."""
        try:
            return int(value)
        except (TypeError, ValueError):
            return None


# Module-level singleton
metadata_extractor = MetadataExtractor()
