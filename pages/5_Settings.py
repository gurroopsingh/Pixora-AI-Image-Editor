import os
from pathlib import Path
import streamlit as st

from config import Config
from modules.providers import GeminiProvider, OpenAIProvider, MockProvider, get_provider
from modules.ui_components import init_session_state, inject_custom_css, render_header, render_sidebar

init_session_state()
inject_custom_css()
render_sidebar()

render_header(
    title="⚙️ Settings & Engine Configuration",
    subtitle="Manage AI providers, API credentials, model parameters, and local storage limits.",
)

tab_provider, tab_models, tab_storage = st.tabs(["AI Providers & API Keys", "Model Selection", "Storage & Database"])

with tab_provider:
    st.markdown("### 🔌 Primary AI Provider Selection")

    current_provider = st.session_state.get("ai_provider", Config.AI_PROVIDER).lower()
    provider_options = ["gemini", "openai", "mock"]
    provider_labels = {
        "gemini": "Google Gemini (Official google-genai — Recommended)",
        "openai": "OpenAI (GPT-4o Vision & Image Editing)",
        "mock": "Developer Mock Engine (Offline Unit Testing & Demo Mode)",
    }

    selected_prov = st.radio(
        "Active Engine",
        options=provider_options,
        index=provider_options.index(current_provider) if current_provider in provider_options else 0,
        format_func=lambda x: provider_labels[x],
    )

    if selected_prov != current_provider:
        st.session_state["ai_provider"] = selected_prov
        st.success(f"Active provider switched to **{selected_prov.capitalize()}**.")
        st.rerun()

    st.markdown("---")
    st.markdown("### 🔑 API Credentials Configuration")
    st.caption("Credentials can be set here and saved to `.env`. Secrets are never exposed in logs.")

    # Gemini Key
    gemini_key = st.text_input(
        "Google Gemini API Key",
        value=Config.GEMINI_API_KEY,
        type="password",
        help="Obtain a free API key from Google AI Studio (https://aistudio.google.com).",
    )

    # OpenAI Key
    openai_key = st.text_input(
        "OpenAI API Key",
        value=Config.OPENAI_API_KEY,
        type="password",
        help="Obtain an API key from OpenAI Platform (https://platform.openai.com/api-keys).",
    )

    # Save button
    if st.button("💾 Save Credentials to .env", type="primary"):
        env_path = Config.BASE_DIR / ".env"
        # Read or create
        lines = []
        if env_path.exists():
            with open(env_path, "r", encoding="utf-8") as f:
                lines = f.readlines()

        keys_to_set = {
            "AI_PROVIDER": selected_prov,
            "GEMINI_API_KEY": gemini_key.strip(),
            "OPENAI_API_KEY": openai_key.strip(),
        }

        new_lines = []
        set_keys = set()
        for line in lines:
            line_stripped = line.strip()
            matched = False
            for k, v in keys_to_set.items():
                if line_stripped.startswith(f"{k}="):
                    new_lines.append(f"{k}={v}\n")
                    set_keys.add(k)
                    matched = True
                    break
            if not matched:
                new_lines.append(line)

        for k, v in keys_to_set.items():
            if k not in set_keys:
                new_lines.append(f"{k}={v}\n")

        with open(env_path, "w", encoding="utf-8") as f:
            f.writelines(new_lines)

        Config.AI_PROVIDER = selected_prov
        Config.GEMINI_API_KEY = gemini_key.strip()
        Config.OPENAI_API_KEY = openai_key.strip()

        st.success("✓ Settings and credentials saved successfully to `.env`!")
        st.rerun()

    st.markdown("---")
    st.markdown("### 🧪 Connection Verification")

    c_test_gemini, c_test_openai = st.columns(2)
    with c_test_gemini:
        if st.button("Test Google Gemini Connection"):
            if not gemini_key.strip():
                st.error("Please enter a Gemini API Key first.")
            else:
                with st.spinner("Pinging Gemini API..."):
                    try:
                        from google import genai
                        client = genai.Client(api_key=gemini_key.strip())
                        resp = client.models.generate_content(
                            model=Config.GEMINI_VISION_MODEL or "gemini-3.5-flash-lite",
                            contents="Respond with 'PIXORA_OK' to verify connectivity.",
                        )
                        if resp and resp.text:
                            st.success(f"✓ Gemini API connected successfully! Model responded: '{resp.text.strip()}'")
                        else:
                            st.warning("Connected but received empty response.")
                    except Exception as e:
                        st.error(f"Gemini connection test failed: {str(e)}")

    with c_test_openai:
        if st.button("Test OpenAI Connection"):
            if not openai_key.strip():
                st.error("Please enter an OpenAI API Key first.")
            else:
                with st.spinner("Pinging OpenAI API..."):
                    try:
                        import openai
                        client = openai.OpenAI(api_key=openai_key.strip())
                        resp = client.chat.completions.create(
                            model=Config.OPENAI_VISION_MODEL or "gpt-4o-mini",
                            messages=[{"role": "user", "content": "Respond with 'PIXORA_OK'"}],
                            max_tokens=10,
                        )
                        st.success(f"✓ OpenAI API connected successfully! Response: '{resp.choices[0].message.content.strip()}'")
                    except Exception as e:
                        st.error(f"OpenAI connection test failed: {str(e)}")

with tab_models:
    st.markdown("### 🤖 Model Customization")

    c_m1, c_m2 = st.columns(2)
    with c_m1:
        st.markdown("#### Google Gemini Models")
        st.text_input("Gemini Image Editing Model", value=Config.GEMINI_IMAGE_MODEL, disabled=True)
        st.caption("Defaults to `imagen-3.0-capability-001` or prioritized fallback `gemini-nano-banana-2.1`.")

        st.text_input("Gemini Vision Captioning Model", value=Config.GEMINI_VISION_MODEL, disabled=True)
        st.caption("Defaults to `gemini-2.5-flash` / `gemini-1.5-flash` for visual scene analysis.")

    with c_m2:
        st.markdown("#### OpenAI Models")
        st.text_input("OpenAI Image Editing Model", value=Config.OPENAI_IMAGE_MODEL, disabled=True)
        st.caption("Defaults to `dall-e-2` or prioritized fallback `gpt-image-2.5-sunburst`.")

        st.text_input("OpenAI Vision Model", value=Config.OPENAI_VISION_MODEL, disabled=True)
        st.caption("Defaults to `gpt-4o-mini` / `gpt-4o`.")

with tab_storage:
    st.markdown("### 💾 Storage & Database Architecture")
    st.markdown(
        f"""
        <div class="pixora-card">
            <b>Project Root:</b> <code>{Config.BASE_DIR}</code><br>
            <b>Database Path:</b> <code>{Config.DATABASE_PATH}</code><br>
            <b>Originals Directory:</b> <code>{Config.ORIGINALS_DIR}</code><br>
            <b>Edited Directory:</b> <code>{Config.EDITED_DIR}</code><br>
            <b>Thumbnails Directory:</b> <code>{Config.THUMBNAILS_DIR}</code><br>
            <b>Max Upload Limit:</b> <code>{Config.MAX_UPLOAD_MB} MB</code>
        </div>
        """,
        unsafe_allow_html=True,
    )
