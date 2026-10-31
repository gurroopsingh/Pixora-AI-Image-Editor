import pytest

from modules.prompt_templates import (
    build_editing_prompt,
    validate_prompt,
    get_vision_caption_prompt,
    SPECIALIZED_TEMPLATES,
)


def test_validate_prompt():
    """Test prompt validation rules."""
    assert validate_prompt("  change color to red  ") == "change color to red"
    
    with pytest.raises(ValueError):
        validate_prompt("")

    with pytest.raises(ValueError):
        validate_prompt("   ")

    with pytest.raises(ValueError):
        validate_prompt(None)


def test_build_editing_prompt_specialized():
    """Test generating specialized prompts for presets."""
    prompt = build_editing_prompt("background_removal", "transparent")
    assert "Isolate the main subject completely" in prompt
    assert "USER REQUEST:" in prompt

    prompt_blur = build_editing_prompt("background_blur", "soft bokeh")
    assert "bokeh blur" in prompt_blur

    prompt_cinematic = build_editing_prompt("cinematic_style", "dramatic sunset")
    assert "cinematic visual grading" in prompt_cinematic
    assert "dramatic sunset" in prompt_cinematic


def test_build_editing_prompt_custom():
    """Test custom prompt formatting."""
    custom_text = "Place a golden retriever sitting next to the sofa"
    prompt = build_editing_prompt("custom", custom_text)
    assert custom_text in prompt
    assert "The output must be an edited version of the supplied image" in prompt


def test_vision_caption_prompt():
    """Test vision caption prompt structure."""
    caption_p = get_vision_caption_prompt()
    assert "professional visual analysis assistant" in caption_p
    assert "main subject, visible objects, environment" in caption_p
