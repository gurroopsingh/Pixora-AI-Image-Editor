import tempfile
from pathlib import Path
import pytest
from PIL import Image

from modules.database import init_db, insert_image, insert_version, get_image
from modules.version_manager import get_image_version_history, revert_to_version, VersionManagerError


@pytest.fixture
def test_setup():
    """Create temp database and simulated version files."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = Path(f.name)
    init_db(db_path)

    # Create dummy images on disk
    temp_dir = Path(tempfile.mkdtemp())
    img_v1 = temp_dir / "img_v1.png"
    img_v2 = temp_dir / "img_v2.png"
    img_v3 = temp_dir / "img_v3.png"

    dummy = Image.new("RGB", (100, 100), color="blue")
    dummy.save(img_v1)
    dummy.save(img_v2)
    dummy.save(img_v3)

    img_id = "test-history-img"
    insert_image(
        image_id=img_id,
        original_filename="blue.png",
        stored_filename="blue.png",
        original_path=str(img_v1),
        thumbnail_path=str(img_v1),
        mime_type="image/png",
        width=100,
        height=100,
        file_size=500,
        db_path=db_path,
    )

    # Version 1
    insert_version(
        version_id="v1-id",
        image_id=img_id,
        version_number=1,
        version_path=str(img_v1),
        thumbnail_path=str(img_v1),
        prompt="Original Upload",
        provider="System",
        model="Source",
        is_original=True,
        db_path=db_path,
    )

    # Version 2
    insert_version(
        version_id="v2-id",
        image_id=img_id,
        version_number=2,
        version_path=str(img_v2),
        thumbnail_path=str(img_v2),
        prompt="Make it red",
        provider="Gemini",
        model="imagen-3.0-capability-001",
        parent_version_id="v1-id",
        is_original=False,
        db_path=db_path,
    )

    # Version 3
    insert_version(
        version_id="v3-id",
        image_id=img_id,
        version_number=3,
        version_path=str(img_v3),
        thumbnail_path=str(img_v3),
        prompt="Add sunset background",
        provider="Gemini",
        model="imagen-3.0-capability-001",
        parent_version_id="v2-id",
        is_original=False,
        db_path=db_path,
    )

    yield {"db_path": db_path, "image_id": img_id, "temp_dir": temp_dir}

    # Cleanup
    if db_path.exists():
        db_path.unlink()


def test_get_version_history(test_setup):
    """Test retrieving chronological version timeline."""
    history = get_image_version_history(test_setup["image_id"], db_path=test_setup["db_path"])
    assert len(history) == 3
    assert history[0]["version_number"] == 1
    assert history[1]["version_number"] == 2
    assert history[2]["version_number"] == 3

    # Version 3 should currently be active
    assert history[2]["is_active"] is True
    assert history[0]["is_active"] is False


def test_revert_functionality_preserves_subsequent_versions(test_setup):
    """
    Test revert to Version 1:
    - Active version shifts to Version 1
    - Versions 2 and 3 remain intact in history
    - Audit entry is logged
    """
    image_id = test_setup["image_id"]
    db_path = test_setup["db_path"]

    # Revert to version 1
    result = revert_to_version(image_id=image_id, target_version_id="v1-id", db_path=db_path)
    assert result["status"] == "success"
    assert result["reverted_to_version_number"] == 1

    # Verify active pointer in database
    img = get_image(image_id, db_path=db_path)
    assert img["active_version_id"] == "v1-id"

    # Verify ALL 3 versions are still present
    history = get_image_version_history(image_id, db_path=db_path)
    assert len(history) == 3
    assert history[0]["is_active"] is True
    assert history[1]["is_active"] is False
    assert history[2]["is_active"] is False
    assert history[2]["prompt"] == "Add sunset background"  # Future version still exists!
