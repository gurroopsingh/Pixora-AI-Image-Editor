"""
PIXORA AI — Image Editor Engine
Coordinates prompt compilation, provider execution, and version creation for AI edits.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from PIL import Image

from modules.database import get_image
from modules.image_storage import save_edited_version
from modules.prompt_templates import build_editing_prompt, validate_prompt
from modules.providers import BaseAIProvider, get_provider, AIProviderError, APIKeyMissingError


class ImageEditorError(Exception):
    """Raised when an editing operation cannot proceed."""
    pass


def execute_image_edit(
    image_id: str,
    user_prompt: str,
    action_type: str = "custom",
    provider: Optional[BaseAIProvider] = None,
    db_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Executes an AI image-to-image edit:
    1. Validates selected image and non-empty prompt.
    2. Loads current active working image.
    3. Builds structured prompt from templates.
    4. Sends image + prompt to provider API.
    5. Saves output as new sequential version.
    6. Updates active version pointer.
    7. Returns before/after comparison metadata.
    """
    # 1. Validate prompt
    try:
        clean_prompt = validate_prompt(user_prompt)
    except ValueError as e:
        raise ImageEditorError(str(e))

    # 2. Validate image selection
    img_data = get_image(image_id, db_path=db_path)
    if not img_data:
        raise ImageEditorError(f"Image with ID '{image_id}' not found.")

    # 3. Retrieve working image path (current active version or original)
    active_ver = img_data.get("active_version")
    if active_ver and active_ver.get("version_path"):
        working_image_path = Path(active_ver["version_path"])
        parent_version_id = active_ver["id"]
        source_version_num = active_ver["version_number"]
    else:
        working_image_path = Path(img_data["original_path"])
        parent_version_id = None
        source_version_num = 1

    if not working_image_path.exists():
        raise ImageEditorError(f"Working image file not found on disk at {working_image_path}")

    # 4. Resolve provider
    active_provider = provider or get_provider()
    if not active_provider.is_configured():
        raise ImageEditorError(
            f"The selected provider '{active_provider.provider_name.capitalize()}' is not configured with an API key. "
            "Please add your API key in Settings or .env file before generating edits."
        )

    # 5. Build specialized prompt
    compiled_prompt = build_editing_prompt(action_type=action_type, user_prompt=clean_prompt)

    # 6. Execute AI edit
    try:
        edited_pil: Image.Image = active_provider.edit_image(
            image_path=working_image_path,
            prompt=compiled_prompt,
        )
    except APIKeyMissingError as e:
        raise ImageEditorError(str(e))
    except AIProviderError as e:
        raise ImageEditorError(f"AI Edit failed ({e.provider}): {e.message}")
    except Exception as e:
        raise ImageEditorError(f"Unexpected error during generative edit: {str(e)}")

    if not isinstance(edited_pil, Image.Image):
        raise ImageEditorError("Provider returned an invalid or corrupted image structure.")

    # 7. Save output as new version
    provider_name = active_provider.provider_name.capitalize()
    model_name = getattr(active_provider, "image_model", "Standard")

    new_version_data = save_edited_version(
        image_id=image_id,
        pil_image=edited_pil,
        prompt=clean_prompt,
        provider=provider_name,
        model=model_name,
        parent_version_id=parent_version_id,
        db_path=db_path,
    )

    return {
        "status": "success",
        "image_id": image_id,
        "new_version": new_version_data,
        "source_version_number": source_version_num,
        "new_version_number": new_version_data["version_number"],
        "source_image_path": str(working_image_path),
        "edited_image_path": new_version_data["version_path"],
        "thumbnail_path": new_version_data["thumbnail_path"],
        "prompt_used": clean_prompt,
        "compiled_prompt": compiled_prompt,
        "provider": provider_name,
        "model": model_name,
    }
