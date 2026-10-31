import tempfile
from pathlib import Path
import pytest
import sqlite3

from modules.database import (
    init_db,
    insert_image,
    get_image,
    update_image_caption,
    update_image_active_version,
    toggle_image_favorite,
    get_all_images,
    delete_image,
    insert_version,
    get_version,
    get_versions_for_image,
    get_next_version_number,
    record_revert,
    get_platform_metrics,
)


@pytest.fixture
def temp_db():
    """Create a temporary database for test isolation."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        temp_path = Path(f.name)
    init_db(temp_path)
    yield temp_path
    if temp_path.exists():
        temp_path.unlink()


def test_database_init(temp_db):
    """Test table creation and schema integrity."""
    conn = sqlite3.connect(temp_db)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [row[0] for row in cursor.fetchall()]
    conn.close()

    assert "images" in tables
    assert "image_versions" in tables
    assert "image_reverts" in tables
    assert "image_embeddings" in tables


def test_insert_and_get_image(temp_db):
    """Test image insertion and retrieval."""
    img_id = "test-img-001"
    insert_image(
        image_id=img_id,
        original_filename="sample.jpg",
        stored_filename="test-img-001.jpg",
        original_path="/tmp/test-img-001.jpg",
        thumbnail_path="/tmp/thumb-001.webp",
        mime_type="image/jpeg",
        width=1920,
        height=1080,
        file_size=204800,
        caption="A beautiful test mountain landscape",
        db_path=temp_db,
    )

    data = get_image(img_id, db_path=temp_db)
    assert data is not None
    assert data["id"] == img_id
    assert data["original_filename"] == "sample.jpg"
    assert data["width"] == 1920
    assert data["caption"] == "A beautiful test mountain landscape"
    assert data["is_favorite"] == 0


def test_toggle_favorite_and_update_caption(temp_db):
    """Test toggling favorite flag and updating captions."""
    img_id = "test-fav-001"
    insert_image(
        image_id=img_id,
        original_filename="fav.png",
        stored_filename="fav.png",
        original_path="/tmp/fav.png",
        thumbnail_path="/tmp/fav_thumb.webp",
        mime_type="image/png",
        width=800,
        height=600,
        file_size=10240,
        db_path=temp_db,
    )

    # Toggle favorite
    fav_status = toggle_image_favorite(img_id, db_path=temp_db)
    assert fav_status is True
    img = get_image(img_id, db_path=temp_db)
    assert img["is_favorite"] == 1

    # Update caption
    update_image_caption(img_id, "Updated caption for favorite image", db_path=temp_db)
    img = get_image(img_id, db_path=temp_db)
    assert img["caption"] == "Updated caption for favorite image"


def test_version_insertion_and_numbering(temp_db):
    """Test sequential version numbers and active version pointer."""
    img_id = "test-ver-img"
    insert_image(
        image_id=img_id,
        original_filename="tree.png",
        stored_filename="tree.png",
        original_path="/tmp/tree.png",
        thumbnail_path="/tmp/tree_thumb.webp",
        mime_type="image/png",
        width=500,
        height=500,
        file_size=5000,
        db_path=temp_db,
    )

    next_v = get_next_version_number(img_id, db_path=temp_db)
    assert next_v == 1

    # Insert v1 (original)
    v1 = insert_version(
        version_id="ver-1",
        image_id=img_id,
        version_number=1,
        version_path="/tmp/tree.png",
        thumbnail_path="/tmp/tree_thumb.webp",
        prompt="Original Upload",
        provider="System",
        model="Source",
        is_original=True,
        db_path=temp_db,
    )
    assert v1["version_number"] == 1

    # Insert v2 (edit)
    next_v2 = get_next_version_number(img_id, db_path=temp_db)
    assert next_v2 == 2

    v2 = insert_version(
        version_id="ver-2",
        image_id=img_id,
        version_number=2,
        version_path="/tmp/tree_v2.png",
        thumbnail_path="/tmp/tree_v2_thumb.webp",
        prompt="Make the tree look like cherry blossom",
        provider="Gemini",
        model="imagen-3.0-capability-001",
        parent_version_id="ver-1",
        is_original=False,
        db_path=temp_db,
    )
    assert v2["version_number"] == 2

    # Check active version pointer
    img = get_image(img_id, db_path=temp_db)
    assert img["active_version_id"] == "ver-2"
    assert img["active_version"]["version_number"] == 2


def test_delete_cascade(temp_db):
    """Test cascade deletion removes version history and returns file paths."""
    img_id = "test-del-001"
    insert_image(
        image_id=img_id,
        original_filename="delete_me.png",
        stored_filename="delete_me.png",
        original_path="/tmp/delete_me.png",
        thumbnail_path="/tmp/delete_me_thumb.webp",
        mime_type="image/png",
        width=100,
        height=100,
        file_size=1000,
        db_path=temp_db,
    )
    insert_version(
        version_id="del-ver-1",
        image_id=img_id,
        version_number=1,
        version_path="/tmp/delete_me.png",
        thumbnail_path="/tmp/delete_me_thumb.webp",
        prompt="Original",
        provider="System",
        model="Source",
        db_path=temp_db,
    )

    files = delete_image(img_id, db_path=temp_db)
    assert len(files) >= 2  # Original path and thumbnail

    # Database records should be gone
    assert get_image(img_id, db_path=temp_db) is None
    assert len(get_versions_for_image(img_id, db_path=temp_db)) == 0
