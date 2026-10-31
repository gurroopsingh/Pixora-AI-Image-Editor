import io
from pathlib import Path
import streamlit as st
from PIL import Image

from config import Config
from modules.database import get_all_images, get_image
from modules.image_analyzer import analyze_image_caption
from modules.image_editor import execute_image_edit, ImageEditorError
from modules.prompt_templates import PRESET_DEFINITIONS
from modules.providers import get_provider
from modules.ui_components import init_session_state, inject_custom_css, render_header, render_sidebar
from modules.utilities import format_iso_timestamp
from modules.version_manager import get_image_version_history, revert_to_version

init_session_state()
inject_custom_css()
render_sidebar()

render_header(
    title="🎨 AI Edit Studio",
    subtitle="Generative natural-language image editing powered by Gemini & OpenAI.",
)

# Fetch all available images for selector
all_images = get_all_images()

if not all_images:
    st.markdown(
        """
        <div class="empty-state">
            <div style="font-size: 2.5rem;">📷</div>
            <div class="empty-state-title">No Images Available for Editing</div>
            <div class="empty-state-desc">
                Please upload an image first via the Dashboard or Image Library to launch the AI Edit Studio.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.button("📤 Go to Upload & Library", type="primary"):
        st.switch_page("pages/2_Image_Library.py")
    st.stop()

# Image Selection Bar
img_options = {img["id"]: f"{img['original_filename']} (v{img.get('current_version_number') or 1} • {img['version_count']} versions)" for img in all_images}
img_ids = list(img_options.keys())

# Default to session state active_image_id or first image
current_active_id = st.session_state.get("active_image_id")
default_idx = img_ids.index(current_active_id) if current_active_id in img_ids else 0

selected_img_id = st.selectbox(
    "Select Image to Edit",
    options=img_ids,
    index=default_idx,
    format_func=lambda x: img_options[x],
    key="studio_image_selector",
)
st.session_state["active_image_id"] = selected_img_id

# Retrieve full record for active image
active_img = get_image(selected_img_id)
if not active_img:
    st.error("Selected image record could not be loaded.")
    st.stop()

# Determine paths
original_path = Path(active_img["original_path"])
active_ver = active_img.get("active_version")

if active_ver and active_ver.get("version_path") and Path(active_ver["version_path"]).exists():
    working_path = Path(active_ver["version_path"])
    current_version_num = active_ver["version_number"]
else:
    working_path = original_path
    current_version_num = 1

# Studio Layout: 2 Columns (Left: Visual Comparison & Inspector, Right: Prompts & Presets)
col_visual, col_controls = st.columns([1.2, 1.0], gap="large")

with col_visual:
    st.markdown("### 👁️ Visual Workspace")

    # Before and After Comparison
    tab_side_by_side, tab_current, tab_original = st.tabs(["Side-by-Side Comparison", "Current Working Image", "Original Image"])

    with tab_side_by_side:
        comp_col1, comp_col2 = st.columns(2)
        with comp_col1:
            st.markdown(
                f"""
                <div style="font-size: 0.82rem; font-weight: 700; color: #94A3B8; text-transform: uppercase; margin-bottom: 0.35rem;">
                    Original (v1)
                </div>
                """,
                unsafe_allow_html=True,
            )
            if original_path.exists():
                st.image(str(original_path), use_container_width=True)
            else:
                st.warning("Original image file missing.")

        with comp_col2:
            st.markdown(
                f"""
                <div style="font-size: 0.82rem; font-weight: 700; color: #8B5CF6; text-transform: uppercase; margin-bottom: 0.35rem;">
                    Active Working (v{current_version_num})
                </div>
                """,
                unsafe_allow_html=True,
            )
            if working_path.exists():
                st.image(str(working_path), use_container_width=True)
            else:
                st.warning("Working image file missing.")

    with tab_current:
        if working_path.exists():
            st.image(str(working_path), use_container_width=True, caption=f"Active Version {current_version_num}")

    with tab_original:
        if original_path.exists():
            st.image(str(original_path), use_container_width=True, caption="Original Uploaded Source")

    # Download & Export
    if working_path.exists():
        st.markdown("<div style='height: 0.75rem;'></div>", unsafe_allow_html=True)
        down_col1, down_col2 = st.columns(2)
        with down_col1:
            with open(working_path, "rb") as f:
                img_bytes = f.read()
            st.download_button(
                label=f"⬇️ Download PNG (v{current_version_num})",
                data=img_bytes,
                file_name=f"pixora_v{current_version_num}_{active_img['original_filename']}",
                mime="image/png",
                use_container_width=True,
            )
        with down_col2:
            # JPEG converted export
            try:
                rgb_img = Image.open(working_path).convert("RGB")
                buf = io.BytesIO()
                rgb_img.save(buf, format="JPEG", quality=95)
                st.download_button(
                    label=f"⬇️ Download JPEG",
                    data=buf.getvalue(),
                    file_name=f"pixora_v{current_version_num}.jpg",
                    mime="image/jpeg",
                    use_container_width=True,
                )
            except Exception:
                pass

    # AI Vision Caption & Scene Understanding Inspector
    st.markdown("---")
    st.markdown("#### 🧠 AI Vision Analysis & Caption")
    caption = active_img.get("caption")

    if caption:
        st.markdown(
            f"""
            <div style="background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08); border-radius: 10px; padding: 0.85rem; font-size: 0.85rem; color: #CBD5E1; line-height: 1.45;">
                {caption}
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.info("No AI scene description generated yet.")

    prov = get_provider()
    if st.button("🔄 Analyze / Refresh Vision Caption", use_container_width=True):
        if not prov.is_configured():
            st.error(f"Cannot generate caption: {prov.provider_name.capitalize()} API key is not configured.")
        else:
            with st.spinner("Analyzing visual composition with Vision API..."):
                ok, cap_text, err = analyze_image_caption(selected_img_id, provider=prov, force_refresh=True)
                if ok:
                    st.success("Vision caption generated and saved!")
                    st.rerun()
                else:
                    st.error(f"Analysis failed: {err}")

with col_controls:
    st.markdown("### 🪄 Generative Prompt & Presets")

    # Provider Warning if API key is not set
    active_prov = get_provider(st.session_state.get("ai_provider"))
    if not active_prov.is_configured():
        st.warning(
            f"⚠️ **{active_prov.provider_name.capitalize()} API key is not configured!**\n\n"
            "Real AI editing requires a valid API key. Navigate to **5_Settings** to configure your key, "
            "or switch to the **Developer Mock Provider** for testing."
        )

    # Preset Action Buttons
    st.markdown("#### Quick Preset Actions")
    st.caption("Click any preset to load an optimized generative instruction into the prompt editor below.")

    for cat in PRESET_DEFINITIONS:
        st.markdown(f"<div class='preset-section-header'>{cat['category']}</div>", unsafe_allow_html=True)
        # Split presets in category across columns
        presets = cat["presets"]
        cols = st.columns(len(presets))
        for idx, (label, action_key, default_text) in enumerate(presets):
            with cols[idx]:
                if st.button(label, key=f"preset_{action_key}", use_container_width=True):
                    st.session_state["current_prompt"] = default_text
                    st.session_state["active_preset_action"] = action_key
                    st.rerun()

    st.markdown("<div style='height: 0.75rem;'></div>", unsafe_allow_html=True)

    # Natural Language Prompt Input
    st.markdown("#### Natural Language Instruction")
    prompt_input = st.text_area(
        "Describe your edit in plain English",
        value=st.session_state.get("current_prompt", ""),
        height=110,
        placeholder="e.g. 'Remove the background and make it transparent', 'Change the car color to matte black', 'Add a sunset in the sky'",
        key="studio_prompt_box",
        help="Type any instruction. The active preset template will be combined with your custom details.",
    )

    action_type = st.session_state.get("active_preset_action", "custom")
    st.caption(f"Active Prompt Template: **{action_type.replace('_', ' ').title()}**")

    # Generate Edit Button
    generate_btn = st.button("✨ Generate AI Edit", type="primary", use_container_width=True)

    if generate_btn:
        if not prompt_input.strip():
            st.error("Please enter an editing instruction or choose a preset action.")
        elif not active_prov.is_configured():
            st.error(
                f"Cannot generate edit: {active_prov.provider_name.capitalize()} API key is missing. "
                "Configure your API key in Settings or choose Mock Provider for test execution."
            )
        else:
            with st.spinner(f"Sending image to {active_prov.provider_name.capitalize()} ({getattr(active_prov, 'image_model', 'AI')})..."):
                try:
                    result = execute_image_edit(
                        image_id=selected_img_id,
                        user_prompt=prompt_input,
                        action_type=action_type,
                        provider=active_prov,
                    )
                    st.success(
                        f"🎉 Successfully created **Version {result['new_version_number']}** using {result['provider']}!"
                    )
                    # Refresh to show new version
                    st.rerun()
                except ImageEditorError as e:
                    st.error(f"Edit failed: {str(e)}")
                except Exception as e:
                    st.error(f"Unexpected error: {str(e)}")

    st.markdown("---")

    # Fast Version Switcher / Timeline in Studio
    st.markdown("#### 📜 Version Switcher")
    versions = get_image_version_history(selected_img_id)

    for v in reversed(versions):
        is_active = v["is_active"]
        badge = "ACTIVE" if is_active else f"v{v['version_number']}"
        badge_style = "color: #10B981; font-weight: 700;" if is_active else "color: #94A3B8;"

        v_col_thumb, v_col_info, v_col_btn = st.columns([0.25, 0.5, 0.25])
        with v_col_thumb:
            t_path = v.get("thumbnail_path")
            if t_path and Path(t_path).exists():
                st.image(str(t_path), use_container_width=True)

        with v_col_info:
            st.markdown(
                f"""
                <div style="font-size: 0.85rem; font-weight: 600;">
                    v{v['version_number']} <span style="{badge_style}">({badge})</span>
                </div>
                <div style="font-size: 0.75rem; color: #94A3B8; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">
                    "{v['prompt']}"
                </div>
                """,
                unsafe_allow_html=True,
            )

        with v_col_btn:
            if not is_active:
                if st.button("Revert", key=f"quick_revert_{v['id']}", use_container_width=True):
                    revert_to_version(selected_img_id, v["id"])
                    st.success(f"Switched active working image to Version {v['version_number']}!")
                    st.rerun()

    st.caption("Inspect full timeline details and audit trail in **4_Version_History**.")
