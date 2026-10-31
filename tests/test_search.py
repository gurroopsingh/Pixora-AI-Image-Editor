import tempfile
from pathlib import Path
import pytest

from modules.database import init_db, insert_image
from modules.image_search import search_images


@pytest.fixture
def search_test_db():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = Path(f.name)
    init_db(db_path)

    # Insert sample test images
    insert_image(
        image_id="img-beach",
        original_filename="beach_holiday.jpg",
        stored_filename="beach_holiday.jpg",
        original_path="/tmp/beach.jpg",
        thumbnail_path="/tmp/beach_thumb.webp",
        mime_type="image/jpeg",
        width=1920,
        height=1080,
        file_size=50000,
        caption="A sunny tropical beach with golden sand, turquoise ocean waves, and palm trees under a blue sky.",
        db_path=db_path,
    )

    insert_image(
        image_id="img-car",
        original_filename="luxury_sedan.png",
        stored_filename="luxury_sedan.png",
        original_path="/tmp/car.png",
        thumbnail_path="/tmp/car_thumb.webp",
        mime_type="image/png",
        width=1200,
        height=800,
        file_size=80000,
        caption="A sleek modern sports car parked on an asphalt city street at night with neon reflections.",
        db_path=db_path,
    )

    yield db_path

    if db_path.exists():
        db_path.unlink()


def test_search_by_caption_keyword(search_test_db):
    """Test searching images by natural language queries."""
    results, mode = search_images("Find pictures with beach and ocean", db_path=search_test_db)
    assert len(results) >= 1
    assert results[0]["id"] == "img-beach"
    assert "Search" in mode

    results_car, _ = search_images("Show me the car", db_path=search_test_db)
    assert len(results_car) >= 1
    assert results_car[0]["id"] == "img-car"


def test_search_empty_query_returns_all(search_test_db):
    """Test empty search query returns all images."""
    results, mode = search_images("", db_path=search_test_db)
    assert len(results) == 2
    assert mode == "All Library Images"
