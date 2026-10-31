import os
from pathlib import Path
from dotenv import load_dotenv

# Base directory of the project
BASE_DIR = Path(__file__).resolve().parent

# Load .env file
load_dotenv(BASE_DIR / ".env")


class Config:
    """PIXORA AI Configuration Class."""

    BASE_DIR = BASE_DIR

    # AI Provider: 'gemini', 'openai', or 'mock'
    AI_PROVIDER = os.getenv("AI_PROVIDER", "gemini").lower()

    # Google Gemini
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
    GEMINI_IMAGE_MODEL = os.getenv("GEMINI_IMAGE_MODEL", "gemini-nano-banana-2.1").strip()
    GEMINI_VISION_MODEL = os.getenv("GEMINI_VISION_MODEL", "gemini-3.5-flash-lite").strip()

    # Fallback model codenames
    GEMINI_IMAGE_MODEL_PRIORITY = [
        "gemini-nano-banana-2.1",
        "gemini-3.1-flash-image",
        "imagen-3.0-capability-001",
    ]

    # OpenAI
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
    OPENAI_IMAGE_MODEL = os.getenv("OPENAI_IMAGE_MODEL", "gpt-image-2.5-sunburst").strip()
    OPENAI_VISION_MODEL = os.getenv("OPENAI_VISION_MODEL", "gpt-4o-mini").strip()
    OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small").strip()

    OPENAI_IMAGE_MODEL_PRIORITY = [
        "gpt-image-2.5-sunburst",
        "gpt-image-2",
        "gpt-image-1.5",
        "dall-e-2",
    ]

    # Storage paths
    STORAGE_DIR = BASE_DIR / os.getenv("STORAGE_DIR", "storage")
    ORIGINALS_DIR = STORAGE_DIR / "originals"
    EDITED_DIR = STORAGE_DIR / "edited"
    THUMBNAILS_DIR = STORAGE_DIR / "thumbnails"

    # Database path
    DATABASE_PATH = BASE_DIR / os.getenv("DATABASE_PATH", "database/pixora.db")

    # Upload restrictions
    MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", 15))
    MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024

    SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
    SUPPORTED_MIME_TYPES = {
        "image/jpeg",
        "image/jpg",
        "image/png",
        "image/webp",
    }

    # UI Branding
    APP_NAME = "PIXORA AI"
    APP_TAGLINE = "Your imagination. Edited by AI."

    @classmethod
    def ensure_directories(cls):
        """Ensure all required runtime storage and database directories exist."""
        cls.STORAGE_DIR.mkdir(parents=True, exist_ok=True)
        cls.ORIGINALS_DIR.mkdir(parents=True, exist_ok=True)
        cls.EDITED_DIR.mkdir(parents=True, exist_ok=True)
        cls.THUMBNAILS_DIR.mkdir(parents=True, exist_ok=True)
        cls.DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)


# Initialize directories on import
Config.ensure_directories()
