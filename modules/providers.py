"""
PIXORA AI — AI Provider Abstraction
Unified interface for Google Gemini, OpenAI, and Developer Mock providers.
Audited against official Google GenAI and OpenAI Python SDK specifications.
"""

from abc import ABC, abstractmethod
import base64
import io
import os
from pathlib import Path
from typing import List, Optional, Tuple, Union
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageOps
import requests

from config import Config
from modules.prompt_templates import get_vision_caption_prompt


class AIProviderError(Exception):
    """Custom exception for AI provider failures."""
    def __init__(self, message: str, provider: str = "Unknown", recoverable: bool = True):
        super().__init__(message)
        self.message = message
        self.provider = provider
        self.recoverable = recoverable


class APIKeyMissingError(AIProviderError):
    """Raised when an API key is required but missing."""
    pass


class BaseAIProvider(ABC):
    """Abstract base class for all AI providers."""

    provider_name: str = "base"

    @abstractmethod
    def is_configured(self) -> bool:
        """Check if necessary credentials exist for this provider."""
        pass

    @abstractmethod
    def analyze_image(self, image_path: Path) -> str:
        """Analyze an image using a vision model and return a descriptive caption."""
        pass

    @abstractmethod
    def edit_image(self, image_path: Path, prompt: str, **kwargs) -> Image.Image:
        """Perform generative image-to-image edit and return the resulting PIL Image."""
        pass

    def generate_embedding(self, text: str) -> Optional[List[float]]:
        """Generate semantic vector embedding for search (optional)."""
        return None


# ==========================================
# Google Gemini Provider
# ==========================================

class GeminiProvider(BaseAIProvider):
    """Official Google Gemini provider using google-genai SDK."""

    provider_name: str = "gemini"

    def __init__(self, api_key: Optional[str] = None):
        if api_key is not None:
            self.api_key = api_key.strip()
        else:
            self.api_key = Config.GEMINI_API_KEY or os.getenv("GEMINI_API_KEY", "")
        self.image_model = Config.GEMINI_IMAGE_MODEL or "imagen-3.0-capability-001"
        self.vision_model = Config.GEMINI_VISION_MODEL or "gemini-2.5-flash"
        self._client = None

    def is_configured(self) -> bool:
        return bool(self.api_key and len(self.api_key.strip()) > 5)

    def _get_client(self):
        if not self.is_configured():
            raise APIKeyMissingError(
                "Gemini API key is not configured. Please set GEMINI_API_KEY in your .env file "
                "or configure it in the Settings page.",
                provider="Gemini",
            )
        if self._client is None:
            try:
                from google import genai
                self._client = genai.Client(api_key=self.api_key)
            except Exception as e:
                raise AIProviderError(f"Failed to initialize Gemini client: {e}", provider="Gemini")
        return self._client

    def analyze_image(self, image_path: Path) -> str:
        """Generate descriptive caption using Gemini Vision model."""
        client = self._get_client()
        from google.genai import types

        if not Path(image_path).exists():
            raise AIProviderError(f"Image not found at path: {image_path}", provider="Gemini")

        try:
            with open(image_path, "rb") as f:
                image_bytes = f.read()

            suffix = Path(image_path).suffix.lower()
            mime = "image/jpeg" if suffix in (".jpg", ".jpeg") else "image/png"
            if suffix == ".webp":
                mime = "image/webp"

            prompt = get_vision_caption_prompt()

            # Attempt vision analysis with primary model, fallback if needed
            vision_models = [self.vision_model, "gemini-3.5-flash-lite", "gemini-3.5-flash", "gemini-flash-latest"]
            last_err = None

            for m in vision_models:
                try:
                    response = client.models.generate_content(
                        model=m,
                        contents=[
                            types.Part.from_bytes(data=image_bytes, mime_type=mime),
                            prompt,
                        ],
                    )
                    if response and response.text:
                        return response.text.strip()
                except Exception as ex:
                    last_err = ex
                    continue

            raise last_err or Exception("No response returned from Gemini Vision model.")

        except Exception as e:
            raise AIProviderError(
                f"Gemini image analysis failed: {str(e)}",
                provider="Gemini",
            )

    def edit_image(self, image_path: Path, prompt: str, **kwargs) -> Image.Image:
        """
        Perform image-to-image edit using official Gemini image editing API.
        Supplies the actual source image and decodes the resulting visual output.
        """
        client = self._get_client()
        from google.genai import types

        if not Path(image_path).exists():
            raise AIProviderError(f"Image not found: {image_path}", provider="Gemini")

        with open(image_path, "rb") as f:
            image_bytes = f.read()
        suffix = Path(image_path).suffix.lower()
        mime = "image/png" if suffix == ".png" else ("image/webp" if suffix == ".webp" else "image/jpeg")

        # Prioritize configured model, followed by official Imagen editing capabilities
        model_candidates = [self.image_model]
        for candidate in Config.GEMINI_IMAGE_MODEL_PRIORITY:
            if candidate not in model_candidates:
                model_candidates.append(candidate)

        errors_encountered = []

        for model_name in model_candidates:
            # 1. Official Imagen edit_image API (supports imagen-3.0-capability-001)
            try:
                raw_ref = types.RawReferenceImage(
                    reference_id=1,
                    reference_image=types.Image.from_file(location=str(image_path)),
                )
                response = client.models.edit_image(
                    model=model_name,
                    prompt=prompt,
                    reference_images=[raw_ref],
                    config=types.EditImageConfig(
                        number_of_images=1,
                        output_mime_type="image/png",
                    ),
                )

                if response and response.generated_images:
                    gen_img = response.generated_images[0]
                    if gen_img.image and gen_img.image.image_bytes:
                        return Image.open(io.BytesIO(gen_img.image.image_bytes))

            except Exception as e_edit:
                errors_encountered.append(f"{model_name} (edit_image): {str(e_edit)}")

            # 2. Multimodal generate_content API with image input (e.g. gemini-nano-banana-2.1)
            try:
                resp = client.models.generate_content(
                    model=model_name,
                    contents=[
                        types.Part.from_bytes(data=image_bytes, mime_type=mime),
                        prompt,
                    ],
                )
                if resp and resp.candidates:
                    candidate = resp.candidates[0]
                    if candidate.content and candidate.content.parts:
                        for part in candidate.content.parts:
                            img = part.as_image()
                            if img:
                                return img
                            elif getattr(part, "inline_data", None) and part.inline_data.data:
                                return Image.open(io.BytesIO(part.inline_data.data))
            except Exception as e_content:
                errors_encountered.append(f"{model_name} (generate_content): {str(e_content)}")
                continue

        # Check if quota exhaustion was among the errors
        quota_err = next((err for err in errors_encountered if "429" in err or "RESOURCE_EXHAUSTED" in err), None)
        final_err = quota_err or (errors_encountered[-1] if errors_encountered else "Unknown failure")

        raise AIProviderError(
            f"Gemini image editing failed with candidate models ({', '.join(model_candidates)}): {final_err}",
            provider="Gemini",
        )


# ==========================================
# OpenAI Provider
# ==========================================

class OpenAIProvider(BaseAIProvider):
    """Secondary provider using official OpenAI SDK."""

    provider_name: str = "openai"

    def __init__(self, api_key: Optional[str] = None):
        if api_key is not None:
            self.api_key = api_key.strip()
        else:
            self.api_key = Config.OPENAI_API_KEY or os.getenv("OPENAI_API_KEY", "")
        self.image_model = Config.OPENAI_IMAGE_MODEL or "dall-e-2"
        self.vision_model = Config.OPENAI_VISION_MODEL or "gpt-4o-mini"
        self.embedding_model = Config.OPENAI_EMBEDDING_MODEL or "text-embedding-3-small"
        self._client = None

    def is_configured(self) -> bool:
        return bool(self.api_key and len(self.api_key.strip()) > 5)

    def _get_client(self):
        if not self.is_configured():
            raise APIKeyMissingError(
                "OpenAI API key is not configured. Please set OPENAI_API_KEY in your .env file "
                "or configure it in the Settings page.",
                provider="OpenAI",
            )
        if self._client is None:
            try:
                import openai
                self._client = openai.OpenAI(api_key=self.api_key)
            except Exception as e:
                raise AIProviderError(f"Failed to initialize OpenAI client: {e}", provider="OpenAI")
        return self._client

    def analyze_image(self, image_path: Path) -> str:
        """Analyze image using OpenAI Vision model (GPT-4o / GPT-4o-mini)."""
        client = self._get_client()

        if not Path(image_path).exists():
            raise AIProviderError(f"Image not found at path: {image_path}", provider="OpenAI")

        try:
            with open(image_path, "rb") as f:
                b64_img = base64.b64encode(f.read()).decode("utf-8")

            suffix = Path(image_path).suffix.lower()
            mime = "image/jpeg" if suffix in (".jpg", ".jpeg") else "image/png"
            if suffix == ".webp":
                mime = "image/webp"

            prompt = get_vision_caption_prompt()

            response = client.chat.completions.create(
                model=self.vision_model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:{mime};base64,{b64_img}"},
                            },
                        ],
                    }
                ],
                max_tokens=300,
            )

            if response.choices and response.choices[0].message.content:
                return response.choices[0].message.content.strip()

            raise Exception("No content in OpenAI Vision response.")

        except Exception as e:
            raise AIProviderError(f"OpenAI vision analysis failed: {str(e)}", provider="OpenAI")

    def edit_image(self, image_path: Path, prompt: str, **kwargs) -> Image.Image:
        """
        Edit image using OpenAI images.edit API.
        Prepares a standardized square RGBA PNG file and edit mask to ensure compatibility.
        """
        client = self._get_client()

        if not Path(image_path).exists():
            raise AIProviderError(f"Image not found: {image_path}", provider="OpenAI")

        try:
            # Prepare image: OpenAI requires square PNG format < 4MB
            src_img = Image.open(image_path).convert("RGBA")
            w, h = src_img.size
            max_dim = max(w, h)
            square_img = Image.new("RGBA", (max_dim, max_dim), (0, 0, 0, 0))
            offset = ((max_dim - w) // 2, (max_dim - h) // 2)
            square_img.paste(src_img, offset)

            # Resize to 1024x1024 for DALL-E 2 / OpenAI images API
            square_img = square_img.resize((1024, 1024), Image.Resampling.LANCZOS)

            img_buf = io.BytesIO()
            square_img.save(img_buf, format="PNG")
            img_bytes = img_buf.getvalue()

            # Prepare mask: transparent pixels (alpha = 0) indicate areas to be edited
            # For general prompt edits, an alpha mask with full editability ensures OpenAI does not reject
            mask_img = Image.new("RGBA", (1024, 1024), (0, 0, 0, 0))
            mask_buf = io.BytesIO()
            mask_img.save(mask_buf, format="PNG")
            mask_bytes = mask_buf.getvalue()

            # Models to try: configured model, with fallback priority
            models_to_try = [self.image_model]
            for candidate in Config.OPENAI_IMAGE_MODEL_PRIORITY:
                if candidate not in models_to_try:
                    models_to_try.append(candidate)

            last_err = None
            for m in models_to_try:
                try:
                    response = client.images.edit(
                        model=m,
                        image=("source_image.png", img_bytes, "image/png"),
                        mask=("edit_mask.png", mask_bytes, "image/png"),
                        prompt=prompt[:1000],  # Character limit
                        n=1,
                    )

                    if response.data and getattr(response.data[0], "b64_json", None):
                        b64_data = response.data[0].b64_json
                        return Image.open(io.BytesIO(base64.b64decode(b64_data)))
                    elif response.data and getattr(response.data[0], "url", None):
                        res = requests.get(response.data[0].url, timeout=30)
                        if res.status_code == 200:
                            return Image.open(io.BytesIO(res.content))

                except Exception as ex:
                    last_err = ex
                    continue

            raise last_err or Exception("OpenAI image editing returned no image.")

        except Exception as e:
            raise AIProviderError(f"OpenAI image editing failed: {str(e)}", provider="OpenAI")

    def generate_embedding(self, text: str) -> Optional[List[float]]:
        """Generate OpenAI semantic embedding for text."""
        if not self.is_configured():
            return None
        try:
            client = self._get_client()
            resp = client.embeddings.create(
                model=self.embedding_model,
                input=[text],
            )
            if resp.data:
                return resp.data[0].embedding
        except Exception:
            pass
        return None


# ==========================================
# Developer / Mock Provider (For Tests & Offline Validation)
# ==========================================

class MockProvider(BaseAIProvider):
    """
    Developer Mock Provider for unit testing and offline development.
    Explicitly stamped with [MOCK / TEST MODE] so output is never misrepresented.
    """

    provider_name: str = "mock"

    def is_configured(self) -> bool:
        return True

    def analyze_image(self, image_path: Path) -> str:
        """Return a structured test caption."""
        p = Path(image_path)
        return (
            f"[MOCK / TEST MODE] Visual analysis for '{p.name}'. Main subject clearly framed, "
            f"balanced composition with natural lighting and distinct foreground elements. "
            f"Environment verified suitable for generative editing."
        )

    def edit_image(self, image_path: Path, prompt: str, **kwargs) -> Image.Image:
        """Perform a deterministic PIL transformation with visible test overlay."""
        src = Image.open(image_path).convert("RGBA")
        
        # Apply a visual transformation based on keywords in prompt
        prompt_lower = prompt.lower()
        if "black and white" in prompt_lower or "monochrome" in prompt_lower:
            gray = ImageOps.grayscale(src).convert("RGBA")
            result = gray
        elif "blur" in prompt_lower:
            result = src.filter(ImageFilter.GaussianBlur(radius=6))
        elif "warm" in prompt_lower:
            # Warm tint
            tint = Image.new("RGBA", src.size, (255, 140, 0, 40))
            result = Image.alpha_composite(src, tint)
        elif "cool" in prompt_lower or "twilight" in prompt_lower:
            # Cool tint
            tint = Image.new("RGBA", src.size, (0, 100, 255, 40))
            result = Image.alpha_composite(src, tint)
        else:
            # Vivid contrast enhancement
            enh = ImageOps.autocontrast(src.convert("RGB")).convert("RGBA")
            result = enh

        # Draw clear badge identifying mock execution
        draw = ImageDraw.Draw(result)
        badge_text = "PIXORA MOCK / TEST MODE"
        draw.rectangle([(10, 10), (320, 45)], fill=(20, 20, 30, 220), outline=(99, 102, 241), width=2)
        draw.text((20, 18), badge_text, fill=(255, 255, 255))

        return result


# ==========================================
# Provider Factory
# ==========================================

def get_provider(provider_type: Optional[str] = None) -> BaseAIProvider:
    """
    Factory function to retrieve active provider instance.
    Defaults to Config.AI_PROVIDER ('gemini', 'openai', or 'mock').
    """
    selected = (provider_type or Config.AI_PROVIDER or "gemini").lower().strip()

    if selected == "gemini":
        return GeminiProvider()
    elif selected == "openai":
        return OpenAIProvider()
    elif selected == "mock":
        return MockProvider()
    else:
        # Default to Gemini
        return GeminiProvider()
