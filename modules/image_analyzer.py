"""
PIXORA AI — Image Analyzer
Orchestrates vision API analysis, scene understanding, and caption generation.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple

from modules.database import get_image, update_image_caption
from modules.providers import BaseAIProvider, get_provider, AIProviderError, APIKeyMissingError


def analyze_image_caption(
    image_id: str,
    provider: Optional[BaseAIProvider] = None,
    force_refresh: bool = False,
    db_path: Optional[Path] = None,
) -> Tuple[bool, str, Optional[str]]:
    """
    Analyze image using configured vision provider and update database.
    
    Returns:
        (success: bool, caption: str, error_message: Optional[str])
    """
    img_data = get_image(image_id, db_path=db_path)
    if not img_data:
        return False, "", "Image record not found in database."

    # Return cached caption if already exists and refresh is not forced
    existing_caption = img_data.get("caption")
    if existing_caption and not force_refresh:
        return True, existing_caption, None

    # Determine image path to analyze (prefer active version, fallback to original)
    active_ver = img_data.get("active_version")
    if active_ver and active_ver.get("version_path"):
        target_path = Path(active_ver["version_path"])
    else:
        target_path = Path(img_data["original_path"])

    if not target_path.exists():
        return False, "", f"Image file not found on disk at {target_path}"

    active_provider = provider or get_provider()

    # Check provider configuration
    if not active_provider.is_configured():
        return False, "", (
            f"Provider '{active_provider.provider_name.capitalize()}' is not configured with an API key. "
            "Please configure your API key in Settings or .env file to enable AI image understanding."
        )

    try:
        caption = active_provider.analyze_image(target_path)
        if caption:
            update_image_caption(image_id, caption, db_path=db_path)
            return True, caption, None
        else:
            return False, "", "Vision provider returned an empty caption response."

    except APIKeyMissingError as e:
        return False, "", str(e)
    except AIProviderError as e:
        return False, "", f"{e.provider} Vision Error: {e.message}"
    except Exception as e:
        return False, "", f"Unexpected error during image analysis: {str(e)}"
