import io
import uuid
from pathlib import Path
from typing import BinaryIO, Dict, List, Optional, Tuple, Union
from PIL import Image, ImageOps

from config import Config
from modules.database import insert_image, get_next_version_number, insert_version
from modules.utilities import sanitize_filename


class StorageValidationError(Exception):
    """Raised when an uploaded image fails validation."""
    pass


def validate_image_file(file_name: str, file_size: int, file_stream: BinaryIO) -> Tuple[int, int, str]:
    """
    Validate uploaded image type, size limit, and image file integrity.
    Returns (width, height, mime_type).
    """
    # 1. Check extension
    suffix = Path(file_name).suffix.lower()
    if suffix not in Config.SUPPORTED_EXTENSIONS:
        allowed = ", ".join(Config.SUPPORTED_EXTENSIONS)
        raise StorageValidationError(f"Unsupported file format '{suffix}'. Allowed formats: {allowed}")

    # 2. Check file size
    if file_size > Config.MAX_UPLOAD_BYTES:
        raise StorageValidationError(
            f"File size ({file_size / (1024*1024):.1f} MB) exceeds maximum allowed limit of {Config.MAX_UPLOAD_MB} MB."
        )

    if file_size == 0:
        raise StorageValidationError("Uploaded file is empty (0 bytes).")

    # 3. Check integrity with Pillow
    try:
        # Read initial bytes to verify
        pos = file_stream.tell()
        img = Image.open(file_stream)
        img.verify()

        # Reopen after verify to inspect dimensions & format
        file_stream.seek(pos)
        img = Image.open(file_stream)
        width, height = img.size
        img_format = (img.format or "PNG").upper()
        file_stream.seek(pos)

        # Standardize mime type
        mime_map = {
            "JPEG": "image/jpeg",
            "JPG": "image/jpeg",
            "PNG": "image/png",
            "WEBP": "image/webp",
        }
        mime_type = mime_map.get(img_format, f"image/{img_format.lower()}")
        return width, height, mime_type

    except Exception as e:
        raise StorageValidationError(f"Invalid or corrupted image file: {str(e)}")


def generate_thumbnail(
    image_input: Union[Path, str, Image.Image],
    target_path: Path,
    size: Tuple[int, int] = (320, 320),
) -> Path:
    """Generate and save an optimized thumbnail preserving aspect ratio."""
    target_path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(image_input, (str, Path)):
        img = Image.open(image_input)
    else:
        img = image_input.copy()

    # Handle EXIF orientation
    img = ImageOps.exif_transpose(img)

    # Convert to RGB if necessary for WEBP/JPEG
    if img.mode in ("RGBA", "LA", "P"):
        # WEBP supports RGBA
        pass
    else:
        img = img.convert("RGB")

    # Resize with antialiasing thumbnail
    img.thumbnail(size, Image.Resampling.LANCZOS)
    img.save(target_path, format="WEBP", quality=85, method=6)
    return target_path


def save_uploaded_image(
    file_name: str,
    file_bytes: bytes,
    db_path: Optional[Path] = None,
) -> Dict:
    """
    Save original image to disk, create initial thumbnail, insert record into database,
    and initialize version 1 (original version).
    """
    Config.ensure_directories()
    file_size = len(file_bytes)
    stream = io.BytesIO(file_bytes)

    # Validate
    width, height, mime_type = validate_image_file(file_name, file_size, stream)

    # Generate IDs and file paths
    image_id = str(uuid.uuid4())
    clean_name = sanitize_filename(file_name)
    ext = Path(clean_name).suffix.lower() or ".png"
    stored_name = f"{image_id}{ext}"
    original_path = Config.ORIGINALS_DIR / stored_name
    thumbnail_path = Config.THUMBNAILS_DIR / f"{image_id}_v1_thumb.webp"

    # Write original bytes to disk
    with open(original_path, "wb") as f:
        f.write(file_bytes)

    # Generate thumbnail
    generate_thumbnail(original_path, thumbnail_path)

    # Insert main image into DB
    img_data = insert_image(
        image_id=image_id,
        original_filename=clean_name,
        stored_filename=stored_name,
        original_path=str(original_path),
        thumbnail_path=str(thumbnail_path),
        mime_type=mime_type,
        width=width,
        height=height,
        file_size=file_size,
        caption="",
        db_path=db_path,
    )

    # Initialize Version 1 as original version
    version_id = str(uuid.uuid4())
    insert_version(
        version_id=version_id,
        image_id=image_id,
        version_number=1,
        version_path=str(original_path),
        thumbnail_path=str(thumbnail_path),
        prompt="Original Upload",
        provider="System",
        model="Source",
        parent_version_id=None,
        is_original=True,
        db_path=db_path,
    )

    return img_data


def save_edited_version(
    image_id: str,
    pil_image: Image.Image,
    prompt: str,
    provider: str,
    model: str,
    parent_version_id: Optional[str] = None,
    db_path: Optional[Path] = None,
) -> Dict:
    """
    Save edited image to storage/edited, generate thumbnail, and create a new version entry.
    """
    Config.ensure_directories()
    version_id = str(uuid.uuid4())
    version_number = get_next_version_number(image_id, db_path=db_path)

    # Save edited file as PNG to preserve visual quality
    version_filename = f"{image_id}_v{version_number}.png"
    version_path = Config.EDITED_DIR / version_filename
    thumbnail_path = Config.THUMBNAILS_DIR / f"{image_id}_v{version_number}_thumb.webp"

    pil_image.save(version_path, format="PNG")
    generate_thumbnail(pil_image, thumbnail_path)

    ver_data = insert_version(
        version_id=version_id,
        image_id=image_id,
        version_number=version_number,
        version_path=str(version_path),
        thumbnail_path=str(thumbnail_path),
        prompt=prompt,
        provider=provider,
        model=model,
        parent_version_id=parent_version_id,
        is_original=False,
        db_path=db_path,
    )

    return ver_data


def delete_physical_files(file_paths: List[Union[str, Path]]) -> None:
    """Safely unlink physical image and thumbnail files from storage."""
    for fp in file_paths:
        try:
            p = Path(fp)
            if p.exists() and p.is_file():
                p.unlink()
        except Exception:
            pass
