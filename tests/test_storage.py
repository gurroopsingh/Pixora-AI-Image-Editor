import io
import tempfile
from pathlib import Path
import pytest
from PIL import Image

from modules.database import init_db, get_image
from modules.image_storage import (
    validate_image_file,
    generate_thumbnail,
    save_uploaded_image,
    save_edited_version,
    delete_physical_files,
    StorageValidationError,
)


def create_test_image_bytes(format="PNG", size=(200, 200), color="red") -> bytes:
    """Helper to create dummy in-memory image bytes."""
    buf = io.BytesIO()
    img = Image.new("RGB", size, color=color)
    img.save(buf, format=format)
    return buf.getvalue()


def test_validate_image_file_valid():
    """Test validation of valid JPG and PNG streams."""
    png_bytes = create_test_image_bytes("PNG", size=(300, 150))
    w, h, mime = validate_image_file("photo.png", len(png_bytes), io.BytesIO(png_bytes))
    assert w == 300
    assert h == 150
    assert mime == "image/png"


def test_validate_image_file_invalid_extension():
    """Test rejection of unsupported file extensions."""
    data = b"dummy content"
    with pytest.raises(StorageValidationError) as exc:
        validate_image_file("document.pdf", len(data), io.BytesIO(data))
    assert "Unsupported file format" in str(exc.value)


def test_validate_image_file_corrupt():
    """Test rejection of corrupted image bytes."""
    data = b"not an image file at all"
    with pytest.raises(StorageValidationError) as exc:
        validate_image_file("corrupt.png", len(data), io.BytesIO(data))
    assert "Invalid or corrupted image file" in str(exc.value)


def test_thumbnail_generation():
    """Test aspect-ratio thumbnail creation."""
    with tempfile.NamedTemporaryFile(suffix=".webp", delete=False) as f:
        thumb_path = Path(f.name)

    src_img = Image.new("RGB", (800, 400), color="green")
    generate_thumbnail(src_img, thumb_path, size=(200, 200))

    assert thumb_path.exists()
    with Image.open(thumb_path) as thumb_img:
        assert thumb_img.size[0] <= 200
        assert thumb_img.size[1] <= 200

    thumb_path.unlink()


def test_save_uploaded_image_and_cleanup():
    """Test full upload save cycle and file cleanup."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = Path(f.name)
    init_db(db_path)

    img_bytes = create_test_image_bytes("JPEG", (400, 400))
    img_data = save_uploaded_image("camera_shot.jpg", img_bytes, db_path=db_path)

    assert img_data["id"] is not None
    orig_path = Path(img_data["original_path"])
    thumb_path = Path(img_data["thumbnail_path"])

    assert orig_path.exists()
    assert thumb_path.exists()

    # Clean up physical files
    delete_physical_files([orig_path, thumb_path])
    assert not orig_path.exists()
    assert not thumb_path.exists()

    if db_path.exists():
        db_path.unlink()
