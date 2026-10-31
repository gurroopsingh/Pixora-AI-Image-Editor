import tempfile
from pathlib import Path
import pytest
from PIL import Image

from modules.database import init_db
from modules.image_storage import save_uploaded_image, delete_physical_files
from modules.image_editor import execute_image_edit, ImageEditorError
from modules.image_analyzer import analyze_image_caption
from modules.providers import (
    GeminiProvider,
    OpenAIProvider,
    MockProvider,
    get_provider,
    APIKeyMissingError,
)


def test_provider_factory_and_missing_keys():
    """Verify missing API key handling across providers."""
    # Gemini with empty key
    gemini = GeminiProvider(api_key="")
    assert gemini.is_configured() is False
    with pytest.raises(APIKeyMissingError):
        gemini.analyze_image(Path("dummy.png"))

    # OpenAI with empty key
    openai = OpenAIProvider(api_key="")
    assert openai.is_configured() is False
    with pytest.raises(APIKeyMissingError):
        openai.analyze_image(Path("dummy.png"))

    # Mock provider is always configured
    mock = MockProvider()
    assert mock.is_configured() is True


def test_mock_provider_execution_cycle():
    """Test full image editing and analysis workflow using the developer MockProvider."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = Path(f.name)
    init_db(db_path)

    # Create dummy source image
    buf = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    img = Image.new("RGB", (200, 200), color="blue")
    img.save(buf.name, format="PNG")
    with open(buf.name, "rb") as f_img:
        img_bytes = f_img.read()

    # Upload
    img_data = save_uploaded_image("mock_test.png", img_bytes, db_path=db_path)
    image_id = img_data["id"]

    mock_provider = MockProvider()

    # 1. Test image captioning
    success, caption, err = analyze_image_caption(
        image_id=image_id,
        provider=mock_provider,
        db_path=db_path,
    )
    assert success is True
    assert "[MOCK / TEST MODE]" in caption

    # 2. Test editing execution
    result = execute_image_edit(
        image_id=image_id,
        user_prompt="Make it black and white fine art",
        action_type="black_and_white",
        provider=mock_provider,
        db_path=db_path,
    )

    assert result["status"] == "success"
    assert result["new_version_number"] == 2
    assert Path(result["edited_image_path"]).exists()

    # Clean up
    delete_physical_files([
        img_data["original_path"],
        img_data["thumbnail_path"],
        result["edited_image_path"],
        result["thumbnail_path"],
        buf.name,
    ])
    if db_path.exists():
        db_path.unlink()


def test_empty_prompt_rejected():
    """Test that empty prompts are strictly rejected before hitting provider."""
    mock = MockProvider()
    with pytest.raises(ImageEditorError) as exc:
        execute_image_edit("some-id", "   ", provider=mock)
    assert "Editing prompt cannot be empty" in str(exc.value)
