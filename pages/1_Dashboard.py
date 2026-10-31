import streamlit as st
from pathlib import Path
from PIL import Image

from config import Config
from modules.database import get_all_images, get_platform_metrics, get_connection
from modules.image_analyzer import analyze_image_caption
from modules.image_storage import save_uploaded_image, StorageValidationError
from modules.providers import get_provider
from modules.ui_components import init_session_state, inject_custom_css, render_header, render_sidebar
from modules.utilities import format_file_size, format_iso_timestamp

init_session_state()
inject_custom_css()
render_sidebar()

render_header(
    title="Dashboard & System Analytics",
    subtitle="Live status, storage consumption, and recent image editing activity.",
)

# Top KPI row
metrics = get_platform_metrics()
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(
        f"""
        <div class="metric-card">
            <span class="metric-label">Original Images</span>
            <span class="metric-value">{metrics['total_images']}</span>
            <span class="metric-sub">Pristine originals preserved</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
with c2:
    st.markdown(
        f"""
        <div class="metric-card">
            <span class="metric-label">Completed Edits</span>
            <span class="metric-value">{metrics['total_edits']}</span>
            <span class="metric-sub">Generative iterations</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
with c3:
    st.markdown(
        f"""
        <div class="metric-card">
            <span class="metric-label">Version Snapshots</span>
            <span class="metric-value">{metrics['total_versions']}</span>
            <span class="metric-sub">Indexed thumbnails & paths</span>
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
            <span class="metric-sub">Non-destructive rollbacks</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)

# Main dashboard split
col_activity, col_system = st.columns([1.3, 0.7], gap="large")

with col_activity:
    st.markdown("### 🕒 Recent Activity Feed")

    # Fetch recent versions and reverts
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT 
                v.*, 
                i.original_filename
            FROM image_versions v
            JOIN images i ON v.image_id = i.id
            ORDER BY v.created_at DESC
            LIMIT 6
            """
        )
        recent_edits = [dict(r) for r in cursor.fetchall()]

    if not recent_edits:
        st.markdown(
            """
            <div class="pixora-card" style="text-align: center; padding: 2.5rem;">
                <p style="color: #94A3B8; margin: 0;">No edits recorded yet. Upload an image to start generating versions!</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        for edit in recent_edits:
            is_orig = edit["is_original"]
            badge = "ORIGINAL" if is_orig else f"VERSION {edit['version_number']}"
            badge_color = "#10B981" if is_orig else "#8B5CF6"

            c_thumb, c_info = st.columns([0.2, 0.8])
            with c_thumb:
                t_path = edit.get("thumbnail_path")
                if t_path and Path(t_path).exists():
                    st.image(str(t_path), use_container_width=True)
                else:
                    st.image(str(edit["version_path"]), use_container_width=True)

            with c_info:
                st.markdown(
                    f"""
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span style="font-weight: 600; color: #F8FAFC; font-size: 0.95rem;">{edit['original_filename']}</span>
                        <span style="font-size: 0.72rem; font-weight: 700; color: {badge_color}; background: rgba(255,255,255,0.05); padding: 0.15rem 0.5rem; border-radius: 4px;">{badge}</span>
                    </div>
                    <div style="font-size: 0.85rem; color: #CBD5E1; margin-top: 0.2rem;">"{edit['prompt']}"</div>
                    <div style="font-size: 0.75rem; color: #64748B; margin-top: 0.25rem;">
                        {format_iso_timestamp(edit['created_at'])} • Engine: {edit['provider']} ({edit['model']})
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                if st.button(f"Open in Studio", key=f"feed_btn_{edit['id']}"):
                    st.session_state["active_image_id"] = edit["image_id"]
                    st.switch_page("pages/3_AI_Edit_Studio.py")

            st.markdown("<hr style='border-color: rgba(255,255,255,0.06); margin: 0.75rem 0;'>", unsafe_allow_html=True)

with col_system:
    st.markdown("### 🖥️ Engine & Storage Status")

    prov = get_provider()
    is_ready = prov.is_configured()

    st.markdown(
        f"""
        <div class="pixora-card">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span style="font-weight: 600; color: #F8FAFC;">Active AI Provider</span>
                <span style="color: {'#10B981' if is_ready else '#EF4444'}; font-weight: 700;">
                    {'● Online' if is_ready else '○ Offline (No API Key)'}
                </span>
            </div>
            <div style="font-size: 0.82rem; color: #94A3B8; margin-top: 0.4rem;">
                Provider: <b>{prov.provider_name.capitalize()}</b><br>
                Editing Model: <b>{getattr(prov, 'image_model', 'N/A')}</b><br>
                Vision Model: <b>{getattr(prov, 'vision_model', 'N/A')}</b>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Calculate storage stats
    def get_dir_size(path: Path) -> int:
        if not path.exists():
            return 0
        return sum(f.stat().st_size for f in path.glob("**/*") if f.is_file())

    orig_size = get_dir_size(Config.ORIGINALS_DIR)
    edited_size = get_dir_size(Config.EDITED_DIR)
    thumb_size = get_dir_size(Config.THUMBNAILS_DIR)
    total_size = orig_size + edited_size + thumb_size

    st.markdown(
        f"""
        <div class="pixora-card">
            <div style="font-weight: 600; color: #F8FAFC; margin-bottom: 0.5rem;">Disk Storage Allocation</div>
            <div style="font-size: 0.82rem; color: #94A3B8; display: flex; justify-content: space-between; margin-bottom: 0.3rem;">
                <span>Originals:</span> <span style="color: #F8FAFC; font-weight: 600;">{format_file_size(orig_size)}</span>
            </div>
            <div style="font-size: 0.82rem; color: #94A3B8; display: flex; justify-content: space-between; margin-bottom: 0.3rem;">
                <span>Generated Edits:</span> <span style="color: #8B5CF6; font-weight: 600;">{format_file_size(edited_size)}</span>
            </div>
            <div style="font-size: 0.82rem; color: #94A3B8; display: flex; justify-content: space-between; margin-bottom: 0.3rem;">
                <span>Thumbnails:</span> <span style="color: #6366F1; font-weight: 600;">{format_file_size(thumb_size)}</span>
            </div>
            <hr style="border-color: rgba(255,255,255,0.06); margin: 0.5rem 0;">
            <div style="font-size: 0.88rem; color: #F8FAFC; display: flex; justify-content: space-between; font-weight: 700;">
                <span>Total Usage:</span> <span>{format_file_size(total_size)}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.button("⚙️ Manage Engine Settings", use_container_width=True):
        st.switch_page("pages/5_Settings.py")
