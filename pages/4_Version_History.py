import streamlit as st
from pathlib import Path
from PIL import Image

from modules.database import get_all_images, get_image, get_connection
from modules.ui_components import init_session_state, inject_custom_css, render_header, render_sidebar
from modules.utilities import format_iso_timestamp
from modules.version_manager import get_image_version_history, revert_to_version, VersionManagerError

init_session_state()
inject_custom_css()
render_sidebar()

render_header(
    title="⏳ Version History & Revert",
    subtitle="Audit-grade timeline of edits, version snapshots, and non-destructive rollbacks.",
)

all_images = get_all_images()
if not all_images:
    st.info("No images uploaded yet. Visit the Dashboard to get started.")
    st.stop()

# Image Selector
img_options = {img["id"]: f"{img['original_filename']} ({img['version_count']} versions)" for img in all_images}
img_ids = list(img_options.keys())

current_active_id = st.session_state.get("active_image_id")
default_idx = img_ids.index(current_active_id) if current_active_id in img_ids else 0

selected_img_id = st.selectbox(
    "Select Image to View History",
    options=img_ids,
    index=default_idx,
    format_func=lambda x: img_options[x],
    key="history_image_selector",
)
st.session_state["active_image_id"] = selected_img_id

# Fetch full version history
try:
    versions = get_image_version_history(selected_img_id)
except VersionManagerError as e:
    st.error(str(e))
    st.stop()

active_version = next((v for v in versions if v["is_active"]), versions[-1] if versions else None)

st.markdown(
    f"""
    <div style="background: rgba(99, 102, 241, 0.08); border: 1px solid rgba(99, 102, 241, 0.25); border-radius: 12px; padding: 1rem 1.25rem; margin-bottom: 1.5rem;">
        <span style="font-weight: 700; color: #F8FAFC;">Active Working Snapshot:</span> 
        <span style="color: #A5B4FC; font-weight: 600;">Version {active_version['version_number'] if active_version else '1'}</span> • 
        <span style="color: #94A3B8;">Prompt: "{active_version['prompt'] if active_version else 'Original'}"</span>
    </div>
    """,
    unsafe_allow_html=True,
)

# 2-Column Layout: Left = Chronological Timeline List, Right = Large Preview & Revert Action
col_timeline, col_preview = st.columns([1.1, 0.9], gap="large")

with col_timeline:
    st.markdown("### 📋 Version Timeline")
    st.caption("All saved versions are preserved sequentially. Click 'Inspect' on any card to preview.")

    # Show in reverse order (newest on top)
    for v in reversed(versions):
        is_active = v["is_active"]
        is_original = v["is_original"]

        card_class = "version-card active-version" if is_active else "version-card"
        badge_text = "CURRENT ACTIVE" if is_active else ("ORIGINAL (v1)" if is_original else f"VERSION {v['version_number']}")
        badge_color = "#10B981" if is_active else ("#6366F1" if is_original else "#8B5CF6")

        st.markdown(
            f"""
            <div class="{card_class}">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="font-weight: 700; font-size: 1rem; color: #F8FAFC;">
                        Version {v['version_number']}
                    </span>
                    <span style="font-size: 0.72rem; font-weight: 700; color: {badge_color}; background: rgba(255,255,255,0.06); padding: 0.2rem 0.55rem; border-radius: 6px;">
                        ● {badge_text}
                    </span>
                </div>
                <div class="version-prompt">"{v['prompt']}"</div>
                <div class="version-meta">
                    Timestamp: {v['formatted_date']}<br>
                    Engine: {v['provider']} • Model: {v['model']}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        c_vthumb, c_vaction = st.columns([0.4, 0.6])
        with c_vthumb:
            t_path = v.get("thumbnail_path")
            if t_path and Path(t_path).exists():
                st.image(str(t_path), use_container_width=True)
            else:
                st.image(str(v["version_path"]), use_container_width=True)

        with c_vaction:
            if st.button(f"🔍 Inspect Snapshot", key=f"inspect_{v['id']}", use_container_width=True):
                st.session_state["inspected_version_id"] = v["id"]
                st.rerun()

            if not is_active:
                if st.button(f"↩️ Revert to v{v['version_number']}", key=f"revert_{v['id']}", type="primary", use_container_width=True):
                    try:
                        res = revert_to_version(selected_img_id, v["id"])
                        st.success(f"✓ Successfully reverted active working image to Version {v['version_number']}!")
                        st.rerun()
                    except Exception as ex:
                        st.error(f"Revert failed: {str(ex)}")

        st.markdown("<hr style='border-color: rgba(255,255,255,0.06); margin: 0.75rem 0;'>", unsafe_allow_html=True)

with col_preview:
    st.markdown("### 🖼️ Selected Version Preview")

    inspected_id = st.session_state.get("inspected_version_id", active_version["id"] if active_version else None)
    target_v = next((v for v in versions if v["id"] == inspected_id), active_version)

    if target_v and Path(target_v["version_path"]).exists():
        st.image(str(target_v["version_path"]), use_container_width=True, caption=f"Version {target_v['version_number']} Preview")

        st.markdown(
            f"""
            <div class="pixora-card">
                <div style="font-weight: 700; color: #F8FAFC; margin-bottom: 0.4rem;">Snapshot Information</div>
                <div style="font-size: 0.85rem; color: #94A3B8;">
                    <b>Version Number:</b> {target_v['version_number']}<br>
                    <b>Prompt:</b> "{target_v['prompt']}"<br>
                    <b>Provider:</b> {target_v['provider']}<br>
                    <b>Model:</b> {target_v['model']}<br>
                    <b>Created:</b> {target_v['formatted_date']}<br>
                    <b>Active Status:</b> {'Yes (Current working version)' if target_v['is_active'] else 'Historical Snapshot'}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if not target_v["is_active"]:
            st.info(
                "💡 **Revert Safety:** Rolling back to this version will restore it as your active editing image. "
                "Subsequent versions will NOT be erased and will remain permanently accessible in this timeline."
            )
            if st.button(f"↩️ Confirm Revert to Version {target_v['version_number']}", type="primary", use_container_width=True):
                revert_to_version(selected_img_id, target_v["id"])
                st.success(f"Reverted to Version {target_v['version_number']}!")
                st.rerun()
        else:
            if st.button("🎨 Continue Editing in AI Edit Studio", type="primary", use_container_width=True):
                st.switch_page("pages/3_AI_Edit_Studio.py")

# Revert Audit Trail Log Table
st.divider()
st.markdown("### 📜 Audit Log of Revert Events")

with get_connection() as conn:
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT 
            r.*,
            v1.version_number as from_version_num,
            v2.version_number as to_version_num
        FROM image_reverts r
        LEFT JOIN image_versions v1 ON r.from_version_id = v1.id
        LEFT JOIN image_versions v2 ON r.to_version_id = v2.id
        WHERE r.image_id = ?
        ORDER BY r.created_at DESC
        """,
        (selected_img_id,),
    )
    revert_logs = [dict(r) for r in cursor.fetchall()]

if not revert_logs:
    st.caption("No revert operations performed on this image yet.")
else:
    for rlog in revert_logs:
        st.markdown(
            f"""
            <div style="font-size: 0.82rem; color: #94A3B8; padding: 0.4rem 0; border-bottom: 1px solid rgba(255,255,255,0.05);">
                • <b>{format_iso_timestamp(rlog['created_at'])}</b>: Reverted active snapshot from 
                <b>Version {rlog.get('from_version_num') or '?'}</b> ➔ <b>Version {rlog.get('to_version_num') or '?'}</b>
            </div>
            """,
            unsafe_allow_html=True,
        )
