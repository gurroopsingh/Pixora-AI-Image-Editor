"""
PIXORA AI — Shared UI Components & Layout Utilities
"""

from pathlib import Path
import streamlit as st

from config import Config
from modules.database import get_platform_metrics, init_db
from modules.providers import get_provider


def inject_custom_css():
    """Load and inject custom design system CSS."""
    css_path = Config.BASE_DIR / "assets" / "custom.css"
    if css_path.exists():
        with open(css_path, "r", encoding="utf-8") as f:
            css_content = f.read()
        st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)


def init_session_state():
    """Ensure session state variables are properly initialized."""
    init_db()

    if "active_image_id" not in st.session_state:
        st.session_state["active_image_id"] = None

    if "current_prompt" not in st.session_state:
        st.session_state["current_prompt"] = ""

    if "active_preset_action" not in st.session_state:
        st.session_state["active_preset_action"] = "custom"

    if "ai_provider" not in st.session_state:
        st.session_state["ai_provider"] = Config.AI_PROVIDER

    if "edit_in_progress" not in st.session_state:
        st.session_state["edit_in_progress"] = False


def render_sidebar():
    """Render PIXORA AI sidebar with branding, metrics, and provider status."""
    with st.sidebar:
        # Branding
        st.markdown(
            f"""
            <div style="padding: 0.5rem 0 1.25rem 0;">
                <h2 style="font-weight: 800; font-size: 1.6rem; margin: 0; background: linear-gradient(135deg, #FFF 0%, #A5B4FC 60%, #8B5CF6 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                    ⚡ {Config.APP_NAME}
                </h2>
                <div style="font-size: 0.8rem; color: #94A3B8; margin-top: 0.2rem;">
                    {Config.APP_TAGLINE}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.divider()

        # Provider Status Badge
        active_prov = get_provider(st.session_state.get("ai_provider"))
        prov_name = active_prov.provider_name.capitalize()
        is_ready = active_prov.is_configured()

        status_color = "#10B981" if is_ready else "#F59E0B"
        status_text = "Connected" if is_ready else "Needs API Key"
        if active_prov.provider_name == "mock":
            status_color = "#6366F1"
            status_text = "Mock / Dev Mode"

        st.markdown(
            f"""
            <div style="background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08); border-radius: 10px; padding: 0.75rem 1rem; margin-bottom: 1.25rem;">
                <div style="font-size: 0.75rem; color: #64748B; font-weight: 600; text-transform: uppercase;">Active AI Engine</div>
                <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 0.35rem;">
                    <span style="font-weight: 700; font-size: 0.95rem; color: #F8FAFC;">{prov_name}</span>
                    <span style="font-size: 0.75rem; font-weight: 600; color: {status_color}; background: rgba(255,255,255,0.06); padding: 0.15rem 0.55rem; border-radius: 6px;">
                        ● {status_text}
                    </span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Platform Metrics
        metrics = get_platform_metrics()
        st.markdown(
            f"""
            <div style="font-size: 0.75rem; color: #64748B; font-weight: 600; text-transform: uppercase; margin-bottom: 0.5rem;">Library Analytics</div>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0.5rem; margin-bottom: 1.25rem;">
                <div style="background: #121622; border: 1px solid rgba(255,255,255,0.06); border-radius: 8px; padding: 0.6rem;">
                    <div style="font-size: 0.7rem; color: #94A3B8;">Images</div>
                    <div style="font-size: 1.2rem; font-weight: 700; color: #F8FAFC;">{metrics['total_images']}</div>
                </div>
                <div style="background: #121622; border: 1px solid rgba(255,255,255,0.06); border-radius: 8px; padding: 0.6rem;">
                    <div style="font-size: 0.7rem; color: #94A3B8;">AI Edits</div>
                    <div style="font-size: 1.2rem; font-weight: 700; color: #8B5CF6;">{metrics['total_edits']}</div>
                </div>
                <div style="background: #121622; border: 1px solid rgba(255,255,255,0.06); border-radius: 8px; padding: 0.6rem;">
                    <div style="font-size: 0.7rem; color: #94A3B8;">Versions</div>
                    <div style="font-size: 1.2rem; font-weight: 700; color: #6366F1;">{metrics['total_versions']}</div>
                </div>
                <div style="background: #121622; border: 1px solid rgba(255,255,255,0.06); border-radius: 8px; padding: 0.6rem;">
                    <div style="font-size: 0.7rem; color: #94A3B8;">Reverts</div>
                    <div style="font-size: 1.2rem; font-weight: 700; color: #10B981;">{metrics['total_reverts']}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.divider()

        # Quick navigation tips
        st.markdown(
            """
            <div style="font-size: 0.75rem; color: #64748B; line-height: 1.4;">
                <b>Workflow:</b><br>
                1. Upload in <b>Dashboard</b> or <b>Library</b><br>
                2. Edit in <b>AI Edit Studio</b><br>
                3. Inspect & revert in <b>Version History</b>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_header(title: str, subtitle: str):
    """Render consistent header banner."""
    st.markdown(
        f"""
        <div class="pixora-header-card">
            <h1 class="pixora-title">{title}</h1>
            <div class="pixora-tagline">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
