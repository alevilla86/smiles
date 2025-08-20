
import os
import json
import tempfile
import pandas as pd
import numpy as np
import streamlit as st

try:
    from predict_nn_original import predict_from_dataframe
except Exception as e:
    predict_from_dataframe = None
    _import_error = e
else:
    _import_error = None

st.set_page_config(page_title="Inferencia NN (Keras + Scaler)", layout="wide")
st.title("🧠 Inferencia con red neuronal (Keras + Scaler)")
st.caption("Usa tus artefactos: **model.keras**, **x_scaler.pkl** y **meta.json** para predecir sobre un CSV.")

with st.expander("ℹ️ Ayuda rápida", expanded=False):
    st.markdown(
        """
**Pasos básicos**
1) Carga los **artefactos** (tres archivos) desde la barra lateral o escribe una ruta local con esos archivos.
2) Carga el **CSV** de entrada (debe tener las columnas de features definidas en meta.json).
3) Pulsa **Predecir**. Podrás descargar un CSV con la columna *Coeficiente CENFOTEC*, ordenado de mayor a menor.

Si te falta TensorFlow, el módulo de predicción te indicará cómo instalarlo.
        """
    )

st.sidebar.header("⚙️ Artefactos del modelo")

artifacts_mode = st.sidebar.radio(
    "¿Cómo quieres proporcionar los artefactos?",
    options=["Subir archivos", "Ruta local"],
    index=0
)

artifacts_dir = None
tmp_artifacts_dir = None  

def _write_uploaded_file(uploaded_file, dst_path):
    with open(dst_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

def _artifacts_are_valid(folder):
    required = ["model.keras", "x_scaler.pkl", "meta.json"]
    for name in required:
        if not os.path.isfile(os.path.join(folder, name)):
            return False
    return True

def _load_meta_features(folder):
    meta_path = os.path.join(folder, "meta.json")
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        return meta, meta.get("features", [])
    except Exception:
        return None, []

if artifacts_mode == "Subir archivos":
    st.sidebar.write("Sube los **tres** archivos:")
    up_model = st.sidebar.file_uploader(
        "model.keras", type=["keras", "h5"], accept_multiple_files=False, key="up_model"
    )
    up_scaler = st.sidebar.file_uploader(
        "x_scaler.pkl", type=["pkl"], accept_multiple_files=False, key="up_scaler"
    )
    up_meta = st.sidebar.file_uploader(
        "meta.json", type=["json"], accept_multiple_files=False, key="up_meta"
    )
    if up_model and up_scaler and up_meta:
        tmp_artifacts_dir = tempfile.mkdtemp(prefix="artifacts_")
        _write_uploaded_file(up_model, os.path.join(tmp_artifacts_dir, "model.keras"))
        _write_uploaded_file(up_scaler, os.path.join(tmp_artifacts_dir, "x_scaler.pkl"))
        _write_uploaded_file(up_meta, os.path.join(tmp_artifacts_dir, "meta.json"))
        artifacts_dir = tmp_artifacts_dir
        st.sidebar.success("Artefactos cargados en memoria temporal.")
    else:
        st.sidebar.info("Faltan archivos. Sube los tres para continuar.")
else:
    artifacts_dir = st.sidebar.text_input(
        "Ruta local a la carpeta con model.keras, x_scaler.pkl y meta.json",
        value="artifacts_nn_original"
    )
    if artifacts_dir:
        if _artifacts_are_valid(artifacts_dir):
            st.sidebar.success("Carpeta válida ✅")
        else:
            st.sidebar.warning("No se encontraron **los tres** archivos requeridos en esa carpeta.")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📁 CSV de entrada")
    uploaded_csv = st.file_uploader("Sube tu CSV", type=["csv"], accept_multiple_files=False)
    df = None
    if uploaded_csv is not None:
        try:
            df = pd.read_csv(uploaded_csv)
            st.write(f"**Dimensiones:** {df.shape[0]} filas × {df.shape[1]} columnas")
            st.dataframe(df.head(20), use_container_width=True)
        except Exception as e:
            st.error(f"No se pudo leer el CSV: {e}")

with col2:
    st.subheader("🧾 Metadatos del modelo")
    if artifacts_dir and _artifacts_are_valid(artifacts_dir):
        meta, features = _load_meta_features(artifacts_dir)
        if meta:
            st.json(meta)
            if features:
                st.info(f"**Se esperan {len(features)} columnas de features**: {features}")
        else:
            st.warning("No se pudo leer meta.json.")
    else:
        st.info("Carga artefactos para ver meta.json.")

st.markdown("---")

can_predict = (
    artifacts_dir is not None
    and _artifacts_are_valid(artifacts_dir)
    and (df is not None)
    and (predict_from_dataframe is not None)
)

predict_btn = st.button("🚀 Predecir", disabled=not can_predict, use_container_width=True)

if predict_from_dataframe is None and _import_error is not None:
    with st.expander("⚠️ Problema importando el módulo de predicción"):
        st.error(f"No se pudo importar predict_nn_original.py:\n\n{_import_error}")

if predict_btn:
    try:
        with st.spinner("Ejecutando inferencia..."):
            preds = predict_from_dataframe(df, artifacts_dir)

            out = df.copy()
            out["Coeficiente CENFOTEC"] = np.asarray(preds, dtype=float)
            out_sorted = out.sort_values(
                by="Coeficiente CENFOTEC", ascending=False, kind="mergesort"
            ).reset_index(drop=True)

            st.success(f"¡Listo! Se generaron {len(out_sorted)} predicciones.")
            st.dataframe(out_sorted.head(50), use_container_width=True)

            csv_bytes = out_sorted.to_csv(index=False).encode("utf-8")
            st.download_button(
                "💾 Descargar CSV con predicciones",
                data=csv_bytes,
                file_name="predicciones.csv",
                mime="text/csv",
                use_container_width=True,
            )
    except Exception as e:
        st.error("Ocurrió un error durante la predicción.")
        with st.expander("Detalle del error"):
            st.exception(e)

st.markdown("---")
st.caption("Hecho con ❤️ usando Streamlit y tu módulo predict_nn_original.py.")
