"""
PIXORA AI — Version Manager
Audit-grade version tracking, timeline generation, and non-destructive revert system.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional
import uuid

from modules.database import (
    get_image,
    get_version,
    get_versions_for_image,
    record_revert,
    update_image_active_version,
)
from modules.utilities import format_iso_timestamp


class VersionManagerError(Exception):
    """Raised on version tracking or revert failures."""
    pass


def get_image_version_history(image_id: str, db_path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """
    Retrieve full version history timeline for an image, sorted by version number ASC.
    Enhances records with formatted timestamps and active indicators.
    """
    img = get_image(image_id, db_path=db_path)
    if not img:
        raise VersionManagerError(f"Image '{image_id}' not found.")

    versions = get_versions_for_image(image_id, db_path=db_path)
    active_version_id = img.get("active_version_id")

    for v in versions:
        v["is_active"] = (v["id"] == active_version_id)
        v["formatted_date"] = format_iso_timestamp(v.get("created_at"))

    return versions


def revert_to_version(
    image_id: str,
    target_version_id: str,
    db_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Revert the active working version of an image to a previous version.
    
    CRITICAL BEHAVIOR:
    - Points active_version_id to target_version_id.
    - Records an audit entry in image_reverts.
    - Keeps all subsequent versions intact without deletion.
    - Original image remains untouched.
    """
    img = get_image(image_id, db_path=db_path)
    if not img:
        raise VersionManagerError(f"Image '{image_id}' not found.")

    current_active_id = img.get("active_version_id")
    if current_active_id == target_version_id:
        return {"status": "noop", "message": "Target version is already the active version."}

    target_ver = get_version(target_version_id, db_path=db_path)
    if not target_ver or target_ver.get("image_id") != image_id:
        raise VersionManagerError(f"Target version '{target_version_id}' does not belong to image '{image_id}'.")

    # Ensure target image file exists on disk
    if not Path(target_ver["version_path"]).exists():
        raise VersionManagerError(
            f"Version file not found on disk at {target_ver['version_path']}. Cannot revert."
        )

    # Record revert audit event
    revert_id = str(uuid.uuid4())
    record_revert(
        revert_id=revert_id,
        image_id=image_id,
        from_version_id=current_active_id or "initial",
        to_version_id=target_version_id,
        db_path=db_path,
    )

    return {
        "status": "success",
        "image_id": image_id,
        "from_version_id": current_active_id,
        "to_version_id": target_version_id,
        "reverted_to_version_number": target_ver["version_number"],
        "active_version_path": target_ver["version_path"],
        "active_thumbnail_path": target_ver["thumbnail_path"],
    }
