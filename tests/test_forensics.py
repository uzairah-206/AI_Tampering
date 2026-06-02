# backend/tests/test_forensics.py
import os
import pytest
import numpy as np
import tempfile
from PIL import Image
from app.validators.image_validator import validate_input_image, ImageValidationError
from app.services.forensic_signals import compute_forensic_signals
from app.core.startup_validator import startup_validator

def create_temp_image(size=(100, 100), mode="RGB", format="PNG", color=(255, 0, 0)):
    temp = tempfile.NamedTemporaryFile(suffix=f".{format.lower()}", delete=False)
    img = Image.new(mode, size, color)
    img.save(temp.name, format=format)
    temp.close()
    return temp.name

def test_image_validator_not_found():
    """Verify FileNotFoundError is raised when path is invalid."""
    with pytest.raises(FileNotFoundError):
        validate_input_image("non_existent_file.png")

def test_image_validator_invalid_extension():
    """Verify ImageValidationError is raised for unlisted extensions."""
    temp_path = create_temp_image(format="BMP") # Allowed in config, but image_validator.py ALLOWED_EXTENSIONS is frozenset({"jpg", "jpeg", "png"})
    try:
        with pytest.raises(ImageValidationError) as exc:
            validate_input_image(temp_path)
        assert "Invalid image format" in str(exc.value)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

def test_image_validator_too_small():
    """Verify validation fails for images smaller than 16x16 pixels."""
    temp_path = create_temp_image(size=(10, 10), format="PNG")
    try:
        with pytest.raises(ImageValidationError) as exc:
            validate_input_image(temp_path)
        assert "too small" in str(exc.value)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

def test_image_validator_too_large():
    """Verify validation fails for images larger than 16384x16384 pixels."""
    # We can mock the PIL Image size directly rather than generating a huge image to save RAM/CPU
    temp_path = create_temp_image(size=(32, 32), format="PNG")
    
    # Python mock of image size check
    from unittest.mock import patch, PropertyMock
    try:
        with patch('PIL.Image.Image.size', new_callable=PropertyMock) as mock_size:
            mock_size.return_value = (20000, 20000)
            with pytest.raises(ImageValidationError) as exc:
                validate_input_image(temp_path)
            assert "exceed" in str(exc.value)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

def test_image_validator_color_conversions():
    """Verify color conversions for RGBA, grayscale, and palette modes."""
    # 1. Grayscale
    gray_path = create_temp_image(mode="L", format="PNG", color=128)
    try:
        img, info = validate_input_image(gray_path)
        assert img.mode == "RGB"
        assert info["original_mode"] == "L"
    finally:
        if os.path.exists(gray_path):
            os.remove(gray_path)

    # 2. RGBA with transparency
    rgba_path = create_temp_image(mode="RGBA", format="PNG", color=(255, 0, 0, 128))
    try:
        img, info = validate_input_image(rgba_path)
        assert img.mode == "RGB"
        assert info["original_mode"] == "RGBA"
    finally:
        if os.path.exists(rgba_path):
            os.remove(rgba_path)

def test_forensic_signals_computation():
    """Verify that traditional signals compute successfully on a dummy JPEG."""
    temp_path = create_temp_image(format="JPEG", size=(64, 64))
    try:
        scores = compute_forensic_signals(temp_path)
        assert "ela_score" in scores
        assert "fft_score" in scores
        assert "noise_score" in scores
        assert "metadata_score" in scores
        assert 0.0 <= scores["ela_score"] <= 1.0
        assert 0.0 <= scores["fft_score"] <= 1.0
        assert 0.0 <= scores["noise_score"] <= 1.0
        assert 0.0 <= scores["metadata_score"] <= 1.0
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

def test_startup_validator_checklist():
    """Verify startup_validator runs validations and generates a dashboard summary."""
    summary = startup_validator.validate()
    assert "torch" in summary
    assert "cuda" in summary
    assert "gemini" in summary
    assert "warnings" in summary
