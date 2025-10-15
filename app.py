import streamlit as st
st.set_page_config(page_title="Predicción de Diabetes", layout="centered")

import pandas as pd
import numpy as np
import joblib

# =========================
# CARGA DE MODELOS
# =========================
@st.cache_resource
def load_modelos():
    try:
        modelo = joblib.load("modelo_diabetes.joblib")
        preprocessor = joblib.load("Artefactos/preprocessor.joblib")
        return modelo, preprocessor
    except Exception as e:
        st.error("❌ No se pudo cargar el modelo o el preprocesador.")
        st.code(str(e))
        st.stop()

modelo, preprocessor = load_modelos()

st.title("🩺 Predicción de Diabetes")
st.markdown(
    """
    Carga un archivo **CSV** con los datos de pacientes para obtener las predicciones.
    El archivo debe incluir exactamente las columnas con las que se entrenó el preprocesador.
    """
)

# Detectar columnas esperadas
expected_cols = None
if hasattr(preprocessor, "feature_names_in_"):
    expected_cols = list(preprocessor.feature_names_in_)
else:
    st.warning("No se pudo leer `feature_names_in_` del preprocesador. Se intentará transformar sin validación de columnas.")

# Botón para descargar plantilla CSV
if expected_cols:
    plantilla_df = pd.DataFrame(columns=expected_cols)
    st.download_button(
        "📥 Descargar plantilla CSV (columnas esperadas)",
        data=plantilla_df.to_csv(index=False).encode("utf-8"),
        file_name="plantilla_prediccion_diabetes.csv",
        mime="text/csv",
        help="Usa esta plantilla para estructurar tus datos con las columnas exactas."
    )

uploaded_file = st.file_uploader("📂 Selecciona un archivo CSV", type=["csv"])

def align_columns(df_in: pd.DataFrame, expected: list[str]) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Alinea df_in a las columnas esperadas: agrega faltantes con 0 y elimina extras. Reordena."""
    cols_in = df_in.columns.tolist()
    missing = [c for c in expected if c not in cols_in]
    extra   = [c for c in cols_in if c not in expected]

    # Agregar columnas faltantes con 0 (útil si tu dataset ya tiene dummies tipo 'race:...')
    for c in missing:
        df_in[c] = 0

    # Eliminar extras
    if extra:
        df_in = df_in.drop(columns=extra)

    # Reordenar
    df_in = df_in[expected]
    return df_in, missing, extra

if uploaded_file is not None:
    try:
        df_nuevo = pd.read_csv(uploaded_file)
        st.success(f"Archivo cargado: {df_nuevo.shape[0]} filas, {df_nuevo.shape[1]} columnas.")
        st.dataframe(df_nuevo.head())

        # Alinear columnas si sabemos las esperadas
        if expected_cols:
            df_aligned, missing, extra = align_columns(df_nuevo.copy(), expected_cols)

            if missing:
                st.warning(f"Se agregaron {len(missing)} columnas faltantes con valor 0: {missing}")
            if extra:
                st.info(f"Se ignoraron {len(extra)} columnas no utilizadas por el modelo: {extra}")

        else:
            df_aligned = df_nuevo

        # Botón para predecir
        if st.button("🔮 Realizar predicciones"):
            X_new = preprocessor.transform(df_aligned)
            preds = modelo.predict(X_new)

            if hasattr(modelo, "predict_proba"):
                probs = modelo.predict_proba(X_new)[:, 1]
            else:
                # Fallback si el modelo no tiene predict_proba
                probs = np.full(shape=len(preds), fill_value=np.nan)

            df_result = df_nuevo.copy()
            df_result["Predicción"] = np.where(preds == 1, "Diabético", "No diabético")
            if not np.isnan(probs).all():
                df_result["Probabilidad (%)"] = np.round(probs * 100, 2)

            st.subheader("📊 Resultados")
            st.dataframe(df_result)

            # Descargar resultados
            csv_out = df_result.to_csv(index=False).encode("utf-8")
            st.download_button(
                "💾 Descargar resultados (CSV)",
                csv_out,
                "predicciones_diabetes.csv",
                "text/csv"
            )

    except Exception as e:
        st.error("Error al procesar el archivo para predicción.")
        st.code(str(e))
else:
    st.info("Sube un CSV con los datos a evaluar.")
