import streamlit as st
import pandas as pd
import lesihmania_activity_predictions
import base64
import os

# --- VERY IMPORTANT DEBUGGING LINES ---
font_filename = "VITOR.otf" # Let's stick to the .otf as per USAL instructions

current_dir = os.getcwd()

font_path = os.path.join(current_dir, font_filename)

if not os.path.exists(font_path):
    st.error(f"ERROR: The font file `{font_filename}` was NOT found at the expected path: `{font_path}`.")
    st.error("Please ensure the font file is in the same directory as your Streamlit script.")
    st.stop() # Stop execution if file not found, no point in proceeding

def load_font_base64(path):
    try:
        with open(path, "rb") as f:
            encoded_font = base64.b64encode(f.read()).decode('utf-8')
            return encoded_font
    except Exception as e:
        st.error(f"Error reading or encoding font file '{path}': {e}")
        st.stop() # Stop if there's an error during file reading/encoding

font_base64 = load_font_base64(font_path)

if not font_base64:
    st.error("ERROR: Base64 encoding resulted in an empty string. This should not happen if the file was read.")
    st.stop() # Stop if base64 is empty

st.set_page_config(
    page_title="Departamento de Ciencias Farmacéuticas - USAL",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown(f"""
    <style>
        /* Font-face declaration */
        @font-face {{
            font-family: 'USAL';
            src: url(data:font/otf;base64,{font_base64}) format('opentype');
            /* Consider adding these if the font supports them and you want to be explicit */
            font-weight: normal;
            font-style: normal;
            /* display: swap; /* Helps with font loading strategy */
        }}

        /* Apply font to relevant elements */
        html, body, [class*="css"], .stApp {{ /* .stApp targets the main Streamlit container */
            font-family: 'USAL', serif !important; /* !important to override other rules */
            background-color: #ffffff;
            color: #2a2a2a;
        }}

        h1, h2, h3, h4, h5, h6 {{
            font-family: 'USAL', serif !important; /* Ensure headers also use it */
            color: #990000; /* USAL red */
        }}

        .stTextArea > div > div > textarea {{
            font-family: 'monospace'; /* Keep text area as monospace for SMILES */
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

        /* Target Streamlit buttons correctly */
        .stButton > button {{
            background-color: #990000 !important;
            color: white !important;
            border: none !important;
            font-family: 'USAL', serif !important;
            font-size: 1rem !important;
            padding: 0.75rem 1.25rem !important;
            line-height: 1.2 !important;
            border-radius: 0.3rem !important;
            transition: background-color 0.2s ease-in-out;
        }}

        /* Hover state */
        .stButton > button:hover {{
            background-color: #800000 !important;
            color: white !important;
        }}

        /* Focus state */
        .stButton > button:focus {{
            outline: 2px solid #990000 !important;
            outline-offset: 2px !important;
        }}
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1>Departamento de Ciencias Farmacéuticas - USAL</h1>", unsafe_allow_html=True)

st.markdown("<h6>Predice actividad contra leishmania con IC50 < 10 µM.</h6>", unsafe_allow_html=True)

smiles_input = st.text_area(
    label="Ingrese una lista de SMILES (uno por línea):",
    height=200,
    placeholder="Ejemplo:\nCC(=O)Oc1ccccc1C(=O)O\nCCN(CC)CC\nC1=CC=CN=C1"
)

if st.button("Predecir actividad"):
    if smiles_input.strip():
        smiles_list = [s.strip() for s in smiles_input.strip().splitlines() if s.strip()]
        probabilities_df = lesihmania_activity_predictions.calculate_leishmania_activity(smiles_list)
        st.dataframe(probabilities_df, use_container_width=True)
    else:
        st.warning("Por favor, ingrese al menos un SMILES.")

st.markdown("<h4>Recuerde que:.</h4>", unsafe_allow_html=True)
st.markdown("<h6>Una probabilidad mayor a 0,5 indica actividad potencial.</h6>", unsafe_allow_html=True)
st.markdown("<h6>Mayor probabilidad no indica mayor actividad.</h6>", unsafe_allow_html=True)

st.markdown("""
<p style="font-size: 0.9rem; color: #555; margin-top: 3rem;">
    © 2025 - Aplicación desarrollada por CENFOTEC para el Departamento de Ciencias Farmacéuticas de la Universidad de Salamanca.<br>
    Para consultas técnicas, puede escribir a <a href="mailto:avillalobosh@ucenfotec.ac.cr">avillalobosh@ucenfotec.ac.cr</a>.
</p>
""", unsafe_allow_html=True)
