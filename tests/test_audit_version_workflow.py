"""
PIXORA AI — Rigorous Version History & Revert Audit
Tests the exact sequence:
Original ➔ Edit 1 ➔ Edit 2 ➔ Edit 3 ➔ Revert to Edit 1
"""

import io
import tempfile
from pathlib import Path
import pytest
from PIL import Image

from modules.database import init_db, get_image, get_versions_for_image
from modules.image_editor import execute_image_edit
from modules.image_storage import save_uploaded_image, delete_physical_files
from modules.providers import MockProvider
from modules.version_manager import get_image_version_history, revert_to_version


def test_exact_audit_version_workflow():
    """
    Executes and validates:
    Original (v1) ➔ Edit 1 (v2) ➔ Edit 2 (v3) ➔ Edit 3 (v4) ➔ Revert to Edit 1 (v2).
    """
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = Path(f.name)
    init_db(db_path)

    # 1. Create and Upload Original Image (Version 1)
    buf = io.BytesIO()
    original_pil = Image.new("RGB", (400, 400), color=(100, 150, 200))
    original_pil.save(buf, format="PNG")
    original_bytes = buf.getvalue()

    uploaded = save_uploaded_image("audit_workflow_source.png", original_bytes, db_path=db_path)
    image_id = uploaded["id"]
    original_disk_path = Path(uploaded["original_path"])

    assert original_disk_path.exists()
    original_file_hash_len = len(original_disk_path.read_bytes())

    # Verify initial version is 1
    v1_record = get_image(image_id, db_path=db_path)
    assert v1_record["active_version"]["version_number"] == 1
    assert v1_record["active_version"]["is_original"] == 1

    mock_provider = MockProvider()

    # 2. Edit 1: Background removal ➔ creates Version 2
    prompt_1 = "Remove the background and make it clean"
    edit_1 = execute_image_edit(
        image_id=image_id,
        user_prompt=prompt_1,
        action_type="background_removal",
        provider=mock_provider,
        db_path=db_path,
    )
    assert edit_1["new_version_number"] == 2
    v2_id = edit_1["new_version"]["id"]

    # 3. Edit 2: Style enhancement ➔ creates Version 3
    prompt_2 = "Apply cinematic lighting and 35mm film grading"
    edit_2 = execute_image_edit(
        image_id=image_id,
        user_prompt=prompt_2,
        action_type="cinematic_style",
        provider=mock_provider,
        db_path=db_path,
    )
    assert edit_2["new_version_number"] == 3
    v3_id = edit_2["new_version"]["id"]

    # 4. Edit 3: Color modification ➔ creates Version 4
    prompt_3 = "Change the color tone to warm golden hour"
    edit_3 = execute_image_edit(
        image_id=image_id,
        user_prompt=prompt_3,
        action_type="warm_grading",
        provider=mock_provider,
        db_path=db_path,
    )
    assert edit_3["new_version_number"] == 4
    v4_id = edit_3["new_version"]["id"]

    # Verify database has 4 chronological versions
    history_before_revert = get_image_version_history(image_id, db_path=db_path)
    assert len(history_before_revert) == 4
    assert [v["version_number"] for v in history_before_revert] == [1, 2, 3, 4]
    assert history_before_revert[3]["is_active"] is True

    # 5. Execute Revert to Edit 1 (Version 2)
    revert_result = revert_to_version(image_id=image_id, target_version_id=v2_id, db_path=db_path)
    assert revert_result["status"] == "success"
    assert revert_result["reverted_to_version_number"] == 2

    # --- Verification Checks ---

    # A. Confirm Original remains unchanged on disk and in database
    assert original_disk_path.exists()
    assert len(original_disk_path.read_bytes()) == original_file_hash_len
    img_post_revert = get_image(image_id, db_path=db_path)
    assert img_post_revert["original_path"] == str(original_disk_path)

    # B. Confirm Revert updates active version
    assert img_post_revert["active_version_id"] == v2_id
    assert img_post_revert["active_version"]["version_number"] == 2

    # C. Confirm ALL versions remain available (Versions 3 and 4 NOT deleted!)
    history_post_revert = get_image_version_history(image_id, db_path=db_path)
    assert len(history_post_revert) == 4
    version_numbers = [v["version_number"] for v in history_post_revert]
    assert version_numbers == [1, 2, 3, 4], "Future versions must NOT be deleted upon revert!"

    # D. Confirm active flags
    assert history_post_revert[0]["is_active"] is False  # v1
    assert history_post_revert[1]["is_active"] is True   # v2 (Reverted to this!)
    assert history_post_revert[2]["is_active"] is False  # v3 (still preserved)
    assert history_post_revert[3]["is_active"] is False  # v4 (still preserved)

    # E. Confirm timestamps and prompts are preserved accurately
    assert history_post_revert[1]["prompt"] == prompt_1
    assert history_post_revert[2]["prompt"] == prompt_2
    assert history_post_revert[3]["prompt"] == prompt_3
    for v in history_post_revert:
        assert v["created_at"] is not None
        assert v["formatted_date"] is not None

    # Cleanup
    files_to_clean = [
        uploaded["original_path"],
        uploaded["thumbnail_path"],
        edit_1["edited_image_path"],
        edit_1["thumbnail_path"],
        edit_2["edited_image_path"],
        edit_2["thumbnail_path"],
        edit_3["edited_image_path"],
        edit_3["thumbnail_path"],
    ]
    delete_physical_files(files_to_clean)
    if db_path.exists():
        db_path.unlink()
