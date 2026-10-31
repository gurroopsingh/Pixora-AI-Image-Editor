import streamlit as st
from pathlib import Path
from PIL import Image

from config import Config
from modules.database import get_all_images, get_platform_metrics, init_db
from modules.image_analyzer import analyze_image_caption
from modules.image_storage import save_uploaded_image, StorageValidationError
from modules.providers import get_provider
from modules.ui_components import init_session_state, inject_custom_css, render_header, render_sidebar

# Configure page
st.set_page_config(
    page_title="PIXORA AI — AI Image Editing Platform",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize app state and CSS
init_session_state()
inject_custom_css()
render_sidebar()

# Hero Header
render_header(
    title=f"⚡ {Config.APP_NAME}",
    subtitle=f"{Config.APP_TAGLINE} — Professional generative image editing studio.",
)

# KPI Metrics Bar
metrics = get_platform_metrics()
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(
        f"""
        <div class="metric-card">
            <span class="metric-label">Total Images</span>
            <span class="metric-value">{metrics['total_images']}</span>
            <span class="metric-sub">Stored locally & indexed</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
with c2:
    st.markdown(
        f"""
        <div class="metric-card">
            <span class="metric-label">Generative Edits</span>
            <span class="metric-value">{metrics['total_edits']}</span>
            <span class="metric-sub">AI-powered transformations</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
with c3:
    st.markdown(
        f"""
        <div class="metric-card">
            <span class="metric-label">Saved Versions</span>
            <span class="metric-value">{metrics['total_versions']}</span>
            <span class="metric-sub">Non-destructive history</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
with c4:
    st.markdown(
        f"""
        <div class="metric-card">
            <span class="metric-label">Revert Operations</span>
            <span class="metric-value">{metrics['total_reverts']}</span>
            <span class="metric-sub">Auditable history jumps</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)

# Main Quick Upload & Getting Started Layout
col_upload, col_quickstart = st.columns([1.1, 0.9], gap="large")

with col_upload:
    st.markdown("### 📤 Upload New Image")
    st.caption("Supports JPG, PNG, and WEBP up to 15 MB. Uploaded files automatically create Version 1.")

    uploaded_files = st.file_uploader(
        "Drop your photo here or browse",
        type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=True,
        key="home_uploader",
    )

    if uploaded_files:
        for uf in uploaded_files:
            file_bytes = uf.read()
            try:
                with st.spinner(f"Saving and verifying '{uf.name}'..."):
                    img_data = save_uploaded_image(file_name=uf.name, file_bytes=file_bytes)
                    img_id = img_data["id"]

                    # Automatically run vision captioning if provider is configured
                    provider = get_provider()
                    if provider.is_configured():
                        with st.spinner("Generating AI vision caption & scene understanding..."):
                            analyze_image_caption(image_id=img_id, provider=provider)

                st.success(f"✓ '{uf.name}' successfully added to your library as Version 1!")
                st.session_state["active_image_id"] = img_id

            except StorageValidationError as e:
                st.error(f"Upload failed: {str(e)}")
            except Exception as e:
                st.error(f"Unexpected error: {str(e)}")

        st.info("💡 Head over to **3_AI_Edit_Studio** from the sidebar or click below to start editing.")
        if st.button("🎨 Open Latest in AI Edit Studio", type="primary", use_container_width=True):
            st.switch_page("pages/3_AI_Edit_Studio.py")

with col_quickstart:
    st.markdown("### 🚀 Fast Navigation")
    st.markdown(
        """
        <div class="pixora-card">
            <h4 style="margin: 0 0 0.5rem 0; color: #A5B4FC;">1. Image Library</h4>
            <p style="font-size: 0.85rem; color: #94A3B8; margin-bottom: 0.75rem;">
                Browse all uploaded photos, view AI-generated descriptions, filter by favorites, and search naturally.
            </p>
        </div>
        <div class="pixora-card">
            <h4 style="margin: 0 0 0.5rem 0; color: #8B5CF6;">2. AI Edit Studio</h4>
            <p style="font-size: 0.85rem; color: #94A3B8; margin-bottom: 0.75rem;">
                Type any natural language instruction or use presets (Background removal, Object removal, Cinematic grading).
            </p>
        </div>
        <div class="pixora-card">
            <h4 style="margin: 0 0 0.5rem 0; color: #10B981;">3. Version History & Revert</h4>
            <p style="font-size: 0.85rem; color: #94A3B8; margin-bottom: 0.75rem;">
                Every edit creates an immutable version thumbnail. Preview and revert at any time without losing newer edits.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    btn_c1, btn_c2 = st.columns(2)
    with btn_c1:
        if st.button("📁 View Image Library", use_container_width=True):
            st.switch_page("pages/2_Image_Library.py")
    with btn_c2:
        if st.button("⚙️ Configure API Keys", use_container_width=True):
            st.switch_page("pages/5_Settings.py")

st.divider()

# Recent Library Preview
st.markdown("### 🖼️ Recent Images in Library")
recent_images = get_all_images()[:4]

if not recent_images:
    st.markdown(
        """
        <div class="empty-state">
            <div style="font-size: 2.5rem;">📷</div>
            <div class="empty-state-title">Your Image Library is Empty</div>
            <div class="empty-state-desc">
                Upload an image above to experience generative natural language AI editing with PIXORA AI.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    cols = st.columns(len(recent_images))
    for i, img in enumerate(recent_images):
        with cols[i]:
            thumb_path = img.get("current_thumbnail_path") or img.get("thumbnail_path")
            if thumb_path and Path(thumb_path).exists():
                st.image(str(thumb_path), use_container_width=True)
            else:
                st.image(str(img["original_path"]), use_container_width=True)

            st.markdown(f"**{img['original_filename']}**")
            st.caption(f"v{img.get('current_version_number') or 1} • {img['version_count']} version(s)")
            if st.button(f"Edit Image", key=f"rec_edit_{img['id']}", use_container_width=True):
                st.session_state["active_image_id"] = img["id"]
                st.switch_page("pages/3_AI_Edit_Studio.py")
