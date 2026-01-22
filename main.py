"""
Leishmania Activity Prediction Web Application.

A Streamlit-based UI for predicting anti-leishmanial activity of chemical compounds
using SMILES notation.
"""
import base64
import os
from pathlib import Path
from typing import Optional

import streamlit as st

from constants import UIConfig, PredictorType
from predictors import create_predictor
from model_utils import get_available_rf_models, get_available_vae_models, get_available_xgb_models


# =============================================================================
# Font Loading
# =============================================================================

def load_font_base64(font_path: Path) -> str:
    """Load a font file and return its base64 encoding."""
    with open(font_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def get_font_path() -> Path:
    """Get the path to the custom font file, with validation."""
    font_path = Path(os.getcwd()) / UIConfig.FONT_FILENAME
    if not font_path.exists():
        st.error(
            f"ERROR: Font file `{UIConfig.FONT_FILENAME}` not found at: `{font_path}`.\n"
            "Please ensure the font file is in the same directory as your Streamlit script."
        )
        st.stop()
    return font_path


# =============================================================================
# Styling
# =============================================================================

def get_custom_css(font_base64: str) -> str:
    """Generate custom CSS with the USAL branding and embedded font."""
    return f"""
    <style>
        @font-face {{
            font-family: 'USAL';
            src: url(data:font/otf;base64,{font_base64}) format('opentype');
            font-weight: normal;
            font-style: normal;
        }}

        html, body, [class*="css"], .stApp {{
            font-family: 'USAL', serif !important;
            background-color: {UIConfig.BACKGROUND_COLOR};
            color: {UIConfig.TEXT_COLOR};
        }}

        h1, [class^="stMarkdown"] h1 {{
            font-family: 'USAL', serif !important;
            color: {UIConfig.PRIMARY_COLOR};
        }}

        .stTextArea > div > div > textarea {{
            font-family: 'monospace';
        }}

        .stTextArea label,
        .stTextArea label span,
        .stTextArea label div {{
            color: {UIConfig.TEXT_COLOR} !important;
        }}

        .block-container {{
            padding-top: 2rem;
        }}

        .dataframe {{
            width: 100%;
        }}

        .dataframe td:nth-child(1) {{
            width: 90%;
        }}

        .dataframe td:nth-child(2) {{
            width: 10%;
        }}

        .stButton > button {{
            background-color: {UIConfig.PRIMARY_COLOR} !important;
            color: white !important;
            border: none !important;
            font-family: 'USAL', serif !important;
            font-size: 1rem !important;
            padding: 0.75rem 1.25rem !important;
            line-height: 1.2 !important;
            border-radius: 0.3rem !important;
            transition: background-color 0.2s ease-in-out;
        }}

        .stButton > button:hover {{
            background-color: {UIConfig.PRIMARY_HOVER} !important;
            color: white !important;
        }}

        .stButton > button:focus {{
            outline: 2px solid {UIConfig.PRIMARY_FOCUS} !important;
            outline-offset: 2px !important;
        }}
    </style>
    """


def apply_page_config():
    """Configure Streamlit page settings."""
    st.set_page_config(
        page_title=UIConfig.PAGE_TITLE,
        page_icon=UIConfig.PAGE_ICON,
        layout="wide",
        initial_sidebar_state="collapsed",
    )


def apply_custom_styling():
    """Load font and apply custom CSS styling."""
    font_path = get_font_path()
    font_base64 = load_font_base64(font_path)
    st.markdown(get_custom_css(font_base64), unsafe_allow_html=True)


# =============================================================================
# UI Components
# =============================================================================

def render_header():
    """Render the page header."""
    st.markdown(
        "<h1>Departamento de Ciencias Farmacéuticas - USAL</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<h6>Herramienta para predecir actividad contra Leishmania* con IC50 &lt; 10 µM.</h6>",
        unsafe_allow_html=True,
    )


def render_model_selector() -> tuple[PredictorType, Optional[str], Optional[str], Optional[str]]:
    """
    Render model selection dropdown.

    Returns:
        Tuple of (predictor_type, rf_model_path, vae_ae_path, vae_clf_path)
    """
    rf_models = get_available_rf_models()
    vae_model_pairs = get_available_vae_models()
    xgb_models = get_available_xgb_models()

    # Build options list
    options = []
    option_data = []

    for model in rf_models:
        options.append(f"🌲 {model.display_name}")
        option_data.append((PredictorType.RANDOM_FOREST, str(model.path), None, None))

    for model in xgb_models:
        options.append(f"🚀 {model.display_name}")
        option_data.append((PredictorType.XGBOOST, str(model.path), None, None))

    for ae_model, clf_model in vae_model_pairs:
        date_str = ae_model.timestamp.strftime("%Y-%m-%d %H:%M")
        display = f"🧠 VAE - {date_str} - Accuracy: {ae_model.accuracy:.2%}"
        options.append(display)
        option_data.append((PredictorType.VAE, None, str(ae_model.path), str(clf_model.path)))

    if not options:
        st.warning("No hay modelos entrenados disponibles. Por favor entrene un modelo primero.")
        return PredictorType.RANDOM_FOREST, None, None, None

    selected_idx = st.selectbox(
        "Seleccione el modelo:",
        range(len(options)),
        format_func=lambda i: options[i],
    )

    return option_data[selected_idx]


def render_smiles_input() -> str:
    """Render the SMILES input text area and return the input value."""
    return st.text_area(
        label="Ingrese uno o más SMILES (uno por línea):",
        height=200,
        placeholder="Ejemplo:\nCC(=O)Oc1ccccc1C(=O)O\nCCN(CC)CC\nC1=CC=CN=C1",
    )


def render_prediction_results(
    smiles_input: str,
    predictor_type: PredictorType,
    rf_path: Optional[str],
    vae_ae_path: Optional[str],
    vae_clf_path: Optional[str],
):
    """Process SMILES input and display prediction results."""
    if st.button("Predecir actividad"):
        if not smiles_input.strip():
            st.warning("Por favor, ingrese al menos un SMILES.")
            return

        if rf_path is None and vae_ae_path is None:
            st.error("No hay modelo seleccionado.")
            return

        smiles_list = [s.strip() for s in smiles_input.strip().splitlines() if s.strip()]

        predictor = create_predictor(
            predictor_type=predictor_type,
            model_path=rf_path,
            vae_ae_path=vae_ae_path,
            vae_clf_path=vae_clf_path,
        )
        results_df = predictor.predict(smiles_list)
        st.dataframe(results_df, use_container_width=True)


def render_footer():
    """Render interpretation notes and footer."""
    st.markdown("<h4>Recuerde que:</h4>", unsafe_allow_html=True)
    st.markdown(
        "<h6>Una probabilidad mayor a 0,5 indica actividad potencial.</h6>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<h6>Mayor probabilidad no indica mayor actividad.</h6>",
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <span style="font-size: 0.9rem; color: {UIConfig.MUTED_TEXT_COLOR}; margin-top: 3rem; font-style: italic;">
            * L. major, L. donovani, L. infantum, L. mexicana, L. braziliensis
        </span>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <p style="font-size: 0.9rem; color: {UIConfig.MUTED_TEXT_COLOR}; margin-top: 3rem;">
            © 2025 - Aplicación desarrollada por estudiantes de Maestría del Software con énfasis
            en Inteligencia Artificial de Universidad CENFOTEC para el Departamento de Ciencias
            Farmacéuticas de la Universidad de Salamanca.<br>
            Para consultas técnicas, puede escribir a
            <a href="mailto:avillalobosh@ucenfotec.ac.cr">avillalobosh@ucenfotec.ac.cr</a>.
        </p>
        """,
        unsafe_allow_html=True,
    )


# =============================================================================
# Main Application
# =============================================================================

def main():
    """Main application entry point."""
    apply_page_config()
    apply_custom_styling()

    render_header()

    # Model selection
    predictor_type, rf_path, vae_ae_path, vae_clf_path = render_model_selector()

    # SMILES input and prediction
    smiles_input = render_smiles_input()
    render_prediction_results(smiles_input, predictor_type, rf_path, vae_ae_path, vae_clf_path)

    render_footer()


if __name__ == "__main__":
    main()
