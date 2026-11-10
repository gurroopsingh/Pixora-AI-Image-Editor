import tempfile
from pathlib import Path
import pytest

from modules.database import init_db, insert_image
from modules.image_search import search_images
from modules.vector_store import (
    index_image_vector,
    search_vector_database,
    compute_dense_semantic_embedding,
    get_chroma_client,
    get_vector_collection,
)


@pytest.fixture
def search_test_db():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = Path(f.name)
    init_db(db_path)

    # Insert sample test images
    beach_caption = "A sunny tropical beach with golden sand, turquoise ocean waves, and palm trees under a blue sky."
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
        caption=beach_caption,
        db_path=db_path,
    )
    # Index in ChromaDB
    index_image_vector("img-beach", f"beach_holiday.jpg. {beach_caption}")

    car_caption = "A sleek modern sports car parked on an asphalt city street at night with neon reflections."
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
        caption=car_caption,
        db_path=db_path,
    )
    # Index in ChromaDB
    index_image_vector("img-car", f"luxury_sedan.png. {car_caption}")

    yield db_path

    if db_path.exists():
        db_path.unlink()


def test_search_by_caption_keyword(search_test_db):
    """Test searching images by natural language queries."""
    results, mode = search_images("Find pictures with beach and ocean", db_path=search_test_db)
    assert len(results) >= 1
    assert results[0]["id"] == "img-beach"
    assert "Vector" in mode or "Search" in mode
    if "similarity_score" in results[0]:
        assert results[0]["similarity_score"] > 0.0

    results_car, _ = search_images("Show me the car", db_path=search_test_db)
    assert len(results_car) >= 1
    assert results_car[0]["id"] == "img-car"


def test_vector_similarity_ranking(search_test_db):
    """Test that search results are ranked by vector similarity."""
    results, _ = search_images("automobile racing vehicle on street", db_path=search_test_db)
    assert len(results) >= 1
    assert results[0]["id"] == "img-car"
    assert results[0].get("similarity_score", 0) > 0


def test_semantic_meaning_search(search_test_db):
    """Verify semantic search works by meaning, matching coastal / marine to beach."""
    results, mode = search_images("tropical coastline with sand and marine water", db_path=search_test_db)
    assert len(results) >= 1
    assert results[0]["id"] == "img-beach"


def test_dense_semantic_embedding_normalized():
    """Verify generated dense semantic vectors have unit L2 norm."""
    vec = compute_dense_semantic_embedding("A stunning mountain peak during twilight")
    assert len(vec) == 384
    norm = sum(x * x for x in vec)
    assert pytest.approx(norm, rel=1e-3) == 1.0


def test_search_empty_query_returns_all(search_test_db):
    """Test empty search query returns all images."""
    results, mode = search_images("", db_path=search_test_db)
    assert len(results) == 2
    assert mode == "All Library Images"
