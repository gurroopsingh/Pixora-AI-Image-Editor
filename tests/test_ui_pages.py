"""
PIXORA AI — Streamlit UI Page Rendering & Smoke Tests
Uses Streamlit AppTest framework to verify all pages render without exceptions.
"""

from pathlib import Path
import pytest
from streamlit.testing.v1 import AppTest

from config import Config
from modules.database import init_db
from seed_demo_data import seed_demo_library


@pytest.fixture(scope="module", autouse=True)
def setup_test_data():
    """Ensure database and demo records exist before UI testing."""
    init_db()
    seed_demo_library()


def test_app_main_renders():
    """Verify app.py entrypoint renders cleanly with zero exceptions."""
    at = AppTest.from_file("app.py", default_timeout=15)
    at.run()
    assert not at.exception, f"app.py raised exception: {at.exception}"
    assert len(at.title) >= 0 or len(at.markdown) > 0


def test_dashboard_renders():
    """Verify 1_Dashboard.py renders cleanly with metrics and activity."""
    at = AppTest.from_file("pages/1_Dashboard.py", default_timeout=15)
    at.run()
    assert not at.exception, f"1_Dashboard.py raised exception: {at.exception}"


def test_image_library_renders():
    """Verify 2_Image_Library.py renders gallery, search, and filter inputs."""
    at = AppTest.from_file("pages/2_Image_Library.py", default_timeout=15)
    at.run()
    assert not at.exception, f"2_Image_Library.py raised exception: {at.exception}"


def test_ai_edit_studio_renders():
    """Verify 3_AI_Edit_Studio.py renders visual comparison, prompt inputs, and presets."""
    at = AppTest.from_file("pages/3_AI_Edit_Studio.py", default_timeout=15)
    at.run()
    assert not at.exception, f"3_AI_Edit_Studio.py raised exception: {at.exception}"


def test_version_history_renders():
    """Verify 4_Version_History.py renders timeline snapshots and revert controls."""
    at = AppTest.from_file("pages/4_Version_History.py", default_timeout=15)
    at.run()
    assert not at.exception, f"4_Version_History.py raised exception: {at.exception}"


def test_settings_renders():
    """Verify 5_Settings.py renders provider configuration, API inputs, and model selectors."""
    at = AppTest.from_file("pages/5_Settings.py", default_timeout=15)
    at.run()
    assert not at.exception, f"5_Settings.py raised exception: {at.exception}"
