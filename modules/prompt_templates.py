"""
PIXORA AI — Prompt Templates
Engineered prompt structures for image-to-image AI editing.
"""

from typing import Dict, Optional


BASE_SYSTEM_INSTRUCTION = """You are an expert professional image editor. Edit the supplied original image according to the user's instruction.

USER REQUEST:
{user_prompt}

INSTRUCTIONS:
1. Use the supplied image as the source image.
2. Make the requested changes accurately.
3. Preserve all unrelated visual elements wherever possible.
4. Maintain realistic lighting, perspective, shadows, proportions, and composition.
5. Maintain the identity and appearance of people unless the user explicitly requests a change.
6. Blend newly added objects naturally into the scene.
7. Avoid unwanted distortions, duplicated objects, artifacts, and inconsistent details.
8. Do not change unrelated parts of the image unnecessarily.
9. Return the actual edited image.

The output must be an edited version of the supplied image, not a separate unrelated image."""


SPECIALIZED_TEMPLATES: Dict[str, str] = {
    # Background Actions
    "background_removal": (
        "Isolate the main subject completely. Remove the existing background and make it clean, neutral, "
        "or transparent without leaving fringe or halo artifacts along the subject edges. Keep subject hair and edge details sharp."
    ),
    "background_replacement": (
        "Replace the background behind the primary subject with: {user_prompt}. "
        "Seamlessly match ambient lighting, reflections, shadows, and perspective to ensure the subject integrates naturally into the new setting."
    ),
    "background_blur": (
        "Apply realistic shallow depth-of-field bokeh blur to the background behind the subject. "
        "Keep the foreground subject crisply in focus while smoothly blurring background elements according to distance."
    ),
    "studio_background": (
        "Replace the background with a minimalist high-end studio backdrop with subtle soft gradations, "
        "seamless floor transitions, and clean diffuse studio lighting."
    ),

    # Object Editing Actions
    "object_removal": (
        "Carefully erase and inpaint {user_prompt} from the photograph. "
        "Reconstruct the occluded background and textures seamlessly with consistent grain, patterns, and lighting so no trace remains."
    ),
    "object_addition": (
        "Naturally add {user_prompt} into the image. "
        "Position it logically in 3D space, matching perspective, contact shadows, surface reflections, scale, and color balance."
    ),
    "object_replacement": (
        "Replace {user_prompt} in the scene while retaining surrounding composition, perspective, scale, and lighting conditions intact."
    ),
    "remove_person": (
        "Carefully remove the person ({user_prompt}) from the image. "
        "Inpaint the background smoothly and realistically without leaving ghost outlines or repetitive texture artifacts."
    ),
    "color_replacement": (
        "Modify the color of {user_prompt}. "
        "Preserve natural highlights, shadows, material texture, specular reflections, and gradients without creating flat or artificial fills."
    ),
    "cleanup": (
        "Perform photographic cleanup. Remove dust spots, digital artifacts, scratches, blemishes, glare, "
        "and distracting visual clutter while preserving authentic surface textures and sharpness."
    ),

    # Style & Enhancement Actions
    "image_enhancement": (
        "Enhance this photograph to maximum commercial clarity. "
        "Optimize dynamic range, restore subtle shadow and highlight details, balance exposure, reduce noise, and refine image sharpness."
    ),
    "cinematic_style": (
        "Transform this image with cinematic visual grading: 35mm anamorphic film color grading, subtle atmospheric depth, "
        "directional lighting, rich shadows, and warm highlight roll-off resembling an award-winning feature film frame."
    ),
    "studio_lighting": (
        "Re-light the scene using professional commercial studio lighting (key, fill, and rim lights). "
        "Accentuate subject contours, soften harsh shadows, and produce clean, balanced highlights."
    ),
    "black_and_white": (
        "Convert this photograph into an expressive fine-art black and white image. "
        "Optimize tonal range from deep velvety blacks to crisp luminous whites, accentuating textural detail, contrast, and depth."
    ),
    "warm_grading": (
        "Apply a golden hour warm color grading with gentle amber highlights, soft glowing sunlight ambiance, "
        "and inviting warm tones across the scene."
    ),
    "cool_grading": (
        "Apply a cool cinematic tone with crisp blue-teal shadows, clean neutral highlights, and a modern, moody atmosphere."
    ),
    "product_photo": (
        "Re-render this as a high-end e-commerce hero product photograph: clean diffused studio lighting, crisp specular reflections, "
        "eliminated blemishes and dust, highlighting premium material finishes."
    ),

    # Generic Custom
    "custom": "{user_prompt}",
}


PRESET_DEFINITIONS = [
    {
        "category": "Background",
        "presets": [
            ("Remove Background", "background_removal", "Remove the background and make it clean and isolated."),
            ("Replace Background", "background_replacement", "A modern luxury penthouse interior with city skyline views."),
            ("Blur Background", "background_blur", "Apply shallow depth-of-field portrait bokeh blur."),
            ("Studio Backdrop", "studio_background", "Clean minimalist dark gray studio cyclorama backdrop."),
        ],
    },
    {
        "category": "Object Editing",
        "presets": [
            ("Remove Unwanted Object", "object_removal", "the power lines and distracting signs in the background"),
            ("Add Object", "object_addition", "a lush green indoor potted plant in the corner"),
            ("Replace Object", "object_replacement", "the coffee mug on the table with an artisan ceramic cup"),
            ("Remove Person", "remove_person", "the person standing on the far left edge of the frame"),
            ("Cleanup Image", "cleanup", "dust, blemishes, scratches, and lens flare artifacts"),
        ],
    },
    {
        "category": "Style & Enhancement",
        "presets": [
            ("Enhance Quality", "image_enhancement", "Enhance image sharpness, dynamic range, and overall clarity."),
            ("Cinematic Mood", "cinematic_style", "Cinematic 35mm film grading with dramatic lighting and color contrast."),
            ("Studio Lighting", "studio_lighting", "Professional 3-point soft studio lighting setup."),
            ("Fine-Art B&W", "black_and_white", "High-contrast dramatic monochrome with rich black levels."),
            ("Warm Golden Hour", "warm_grading", "Warm sunset golden hour glow and amber lighting."),
            ("Cool Twilight", "cool_grading", "Cool twilight moody tone with teal and deep blue accents."),
            ("Product Showcase", "product_photo", "Professional commercial product photography with soft studio reflections."),
        ],
    },
]


def validate_prompt(prompt: Optional[str]) -> str:
    """Validate that prompt string is non-empty and stripped."""
    if not prompt or not prompt.strip():
        raise ValueError("Editing prompt cannot be empty.")
    return prompt.strip()


def build_editing_prompt(action_type: str, user_prompt: str) -> str:
    """
    Combine action template with specific user input to generate the final
    structured prompt sent to the image-to-image editing API.
    """
    clean_user = validate_prompt(user_prompt)
    template = SPECIALIZED_TEMPLATES.get(action_type, SPECIALIZED_TEMPLATES["custom"])

    # If the template requires formatting with user_prompt
    if "{user_prompt}" in template:
        action_instruction = template.format(user_prompt=clean_user)
    else:
        # If template is standalone, append user's details if provided
        action_instruction = f"{template} User specific instruction: {clean_user}"

    return BASE_SYSTEM_INSTRUCTION.format(user_prompt=action_instruction)


def get_vision_caption_prompt() -> str:
    """Standardized prompt for image captioning and scene understanding."""
    return (
        "You are a professional visual analysis assistant. Analyze the provided image and produce a "
        "concise but descriptive caption. Identify the main subject, visible objects, environment, "
        "background, colors, composition, and overall scene. Do not invent details that are not "
        "visually supported by the image. Return the description in a clear, searchable format."
    )
