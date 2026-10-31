import base64
from datetime import datetime
import io
import math
import re
from pathlib import Path
from typing import List, Optional, Tuple, Union
from PIL import Image


def format_file_size(size_bytes: int) -> str:
    """Convert bytes to human-readable string (e.g. 1.2 MB, 450 KB)."""
    if size_bytes <= 0:
        return "0 B"
    units = ["B", "KB", "MB", "GB"]
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return f"{s} {units[i]}"


def format_iso_timestamp(iso_str: Optional[str]) -> str:
    """Format an ISO 8601 string into a user-friendly date and time string."""
    if not iso_str:
        return "Unknown"
    try:
        dt = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
        return dt.strftime("%b %d, %Y • %I:%M %p")
    except Exception:
        return iso_str[:16].replace("T", " ")


def sanitize_filename(filename: str) -> str:
    """Sanitize filename to prevent path traversal or special character issues."""
    name = Path(filename).name
    # Remove non-alphanumeric except hyphen, underscore, and dot
    clean = re.sub(r"[^\w\s\.-]", "", name).strip()
    return clean if clean else "image.png"


def pil_to_base64(image: Image.Image, format: str = "PNG") -> str:
    """Encode PIL Image to base64 string."""
    buffered = io.BytesIO()
    image.save(buffered, format=format)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def base64_to_pil(b64_string: str) -> Image.Image:
    """Decode base64 string to PIL Image."""
    if "," in b64_string:
        b64_string = b64_string.split(",", 1)[1]
    image_bytes = base64.b64decode(b64_string)
    return Image.open(io.BytesIO(image_bytes))


def image_to_data_uri(image_path: Union[str, Path]) -> str:
    """Read image from path and return data URI for HTML/CSS rendering."""
    p = Path(image_path)
    if not p.exists():
        return ""
    suffix = p.suffix.lower().lstrip(".")
    mime = f"image/{suffix}" if suffix != "jpg" else "image/jpeg"
    with open(p, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    """Compute cosine similarity between two float vectors."""
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)
