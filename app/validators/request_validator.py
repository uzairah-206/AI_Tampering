import os
from PIL import Image

class RequestValidator:
    ALLOWED_IMAGE_EXTS = {"jpg", "jpeg", "png", "webp"}
    ALLOWED_TEXT_EXTS = {"txt", "pdf"}
    MAX_SIZE_MB = 10

    @classmethod
    def validate(cls, filepath: str, filename: str, is_text: bool = False):
        if not filepath or not os.path.exists(filepath):
            raise ValueError("File does not exist or null path provided.")
            
        ext = filename.split(".")[-1].lower() if "." in filename else ""
        
        if is_text:
            if ext not in cls.ALLOWED_TEXT_EXTS:
                raise ValueError(f"Unsupported text format: {ext}. Allowed: {cls.ALLOWED_TEXT_EXTS}")
        else:
            if ext not in cls.ALLOWED_IMAGE_EXTS:
                raise ValueError(f"Unsupported image format: {ext}. Allowed: {cls.ALLOWED_IMAGE_EXTS}")
            
            # Validate image integrity
            try:
                with Image.open(filepath) as img:
                    img.verify()
            except Exception as e:
                raise ValueError(f"Invalid or corrupted image file: {e}")
                
        # Validate size
        size_mb = os.path.getsize(filepath) / (1024 * 1024)
        if size_mb > cls.MAX_SIZE_MB:
            raise ValueError(f"File size exceeds limit of {cls.MAX_SIZE_MB}MB.")
            
        return True
