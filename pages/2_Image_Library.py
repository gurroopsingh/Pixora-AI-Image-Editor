import streamlit as st
from pathlib import Path
from PIL import Image

from config import Config
from modules.database import (
    get_all_images,
    toggle_image_favorite,
    delete_image,
)
from modules.image_analyzer import analyze_image_caption
from modules.image_search import search_images
from modules.image_storage import delete_physical_files, save_uploaded_image, StorageValidationError
from modules.providers import get_provider
from modules.ui_components import init_session_state, inject_custom_css, render_header, render_sidebar
from modules.utilities import format_file_size, format_iso_timestamp

init_session_state()
inject_custom_css()
render_sidebar()

render_header(
    title="Image Library & Gallery",
    subtitle="Search, filter, and inspect your full collection of uploaded and edited photographs.",
)

# Upload Expander on Library Page
with st.expander("➕ Upload Additional Photos to Library", expanded=False):
    up_files = st.file_uploader(
        "Upload images (JPG, PNG, WEBP)",
        type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=True,
        key="lib_uploader",
    )
    if up_files:
        for uf in up_files:
            try:
                img_data = save_uploaded_image(file_name=uf.name, file_bytes=uf.read())
                prov = get_provider()
                if prov.is_configured():
                    analyze_image_caption(image_id=img_data["id"], provider=prov)
                st.success(f"Added '{uf.name}' to library.")
            except Exception as e:
                st.error(f"Error saving '{uf.name}': {str(e)}")
        st.rerun()

# Search and Filter Toolbar
col_search, col_filter, col_format = st.columns([1.8, 1, 0.8])

with col_search:
    search_query = st.text_input(
        "🔍 Natural Language Search",
        placeholder="e.g. 'Show sunset photos', 'car', 'product shots', 'beach'",
        help="Searches AI scene captions, filenames, and edit prompt history.",
    )

with col_filter:
    filter_option = st.selectbox(
        "Filter Collection",
        options=["All Images", "Recently Uploaded", "Recently Edited", "Favorites Only"],
        index=0,
    )

with col_format:
    format_option = st.selectbox(
        "Format",
        options=["All", "JPEG", "PNG", "WEBP"],
        index=0,
    )

# Map UI filter options to DB filter keys
filter_map = {
    "All Images": "all",
    "Recently Uploaded": "all",
    "Recently Edited": "recent_edits",
    "Favorites Only": "favorites",
}
active_filter_key = filter_map.get(filter_option, "all")
active_format_key = format_option.lower() if format_option != "All" else None

# Execute search/filter query
if search_query:
    images, search_mode = search_images(
        query=search_query,
        filter_type=active_filter_key,
        format_filter=active_format_key,
    )
    st.markdown(
        f"""
        <div style="margin-bottom: 1.25rem;">
            <span class="pill-badge">Search Mode: {search_mode}</span>
            <span style="font-size: 0.85rem; color: #94A3B8; margin-left: 0.75rem;">
                Found <b>{len(images)}</b> matching image(s)
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    images = get_all_images(
        filter_type=active_filter_key,
        format_filter=active_format_key,
    )

# Render Image Gallery Grid
if not images:
    st.markdown(
        """
        <div class="empty-state">
            <div style="font-size: 2.5rem;">🔍</div>
            <div class="empty-state-title">No Images Found</div>
            <div class="empty-state-desc">
                No photographs matched your query or filter criteria. Try adjusting your search term or upload a new image.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    # Render in 3-column responsive card layout
    num_cols = 3
    for row_idx in range(0, len(images), num_cols):
        row_images = images[row_idx : row_idx + num_cols]
        cols = st.columns(num_cols)

        for col_idx, img in enumerate(row_images):
            with cols[col_idx]:
                with st.container():
                    # Thumbnail preview
                    thumb_p = img.get("current_thumbnail_path") or img.get("thumbnail_path")
                    if thumb_p and Path(thumb_p).exists():
                        st.image(str(thumb_p), use_container_width=True)
                    else:
                        st.image(str(img["original_path"]), use_container_width=True)

                    # Card info
                    st.markdown(f"**{img['original_filename']}**")

                    # AI Caption snippet
                    caption = img.get("caption") or "No AI caption generated yet."
                    st.caption(f"🤖 {caption[:100]}..." if len(caption) > 100 else f"🤖 {caption}")

                    # Meta details
                    upload_date = format_iso_timestamp(img.get("upload_timestamp"))
                    st.markdown(
                        f"""
                        <div style="font-size: 0.75rem; color: #64748B; margin-bottom: 0.5rem;">
                            Uploaded: {upload_date}<br>
                            Resolution: {img['width']}×{img['height']} • Versions: <b>{img['version_count']}</b>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    # Card Action Buttons
                    btn_c1, btn_c2 = st.columns([1.3, 0.7])
                    with btn_c1:
                        if st.button("🎨 Edit Studio", key=f"open_studio_{img['id']}", use_container_width=True):
                            st.session_state["active_image_id"] = img["id"]
                            st.switch_page("pages/3_AI_Edit_Studio.py")

                    with btn_c2:
                        fav_label = "★ Unfav" if img.get("is_favorite") else "☆ Fav"
                        if st.button(fav_label, key=f"fav_{img['id']}", use_container_width=True):
                            toggle_image_favorite(img["id"])
                            st.rerun()

                    # Safe Delete with confirmation expander
                    with st.expander("🗑️ Delete Image", expanded=False):
                        st.caption("Permanently removes original, all edited versions, and thumbnails.")
                        if st.button("Confirm Delete", key=f"del_conf_{img['id']}", type="primary"):
                            files_to_delete = delete_image(img["id"])
                            delete_physical_files(files_to_delete)
                            if st.session_state.get("active_image_id") == img["id"]:
                                st.session_state["active_image_id"] = None
                            st.success("Deleted image and version files.")
                            st.rerun()

                st.markdown("<div style='height: 1.25rem;'></div>", unsafe_allow_html=True)
