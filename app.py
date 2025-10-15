# app.py (modo depuración seguro)
import os, sys, platform, traceback
import streamlit as st
st.set_page_config(page_title="Predicción de Diabetes", layout="centered")

import pandas as pd
import numpy as np
import joblib

# ---------- Panel de diagnóstico ----------
with st.expander("🔎 Diagnóstico del entorno (clic para abrir)", expanded=True):
    st.write({
        "Python": sys.version.split()[0],
        "SO": platform.platform(),
    })
    try:
        import sklearn, streamlit as _st
        st.write({
            "streamlit": _st.__version__,
            "scikit_learn": sklearn.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
        })
    except Exception as e:
        st.write("No se pudieron leer versiones:", e)

    st.write("Archivos presentes en el repo raíz:")
    st.write(sorted(os.listdir(".")))
    st.write("Contenido de ./Artefactos (si existe):")
    st.write(sorted(os.listdir("./Artefactos")) if os.path.isdir("./Artefactos") else "No existe")

MODEL_PATH = "modelo_diabetes.joblib"
PREPROC_PATH = "Artefactos/preprocessor.joblib"

# ---------- Carga robusta de artefactos ----------
@st.cache_resource
def load_modelos():
    # Validación de rutas
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"No se encontró {MODEL_PATH}")
    if not os.path.exists(PREPROC_PATH):
        raise FileNotFoundError(f"No se encontró {PREPROC_PATH}")

    # Carga con manejo de errores
    try:
        modelo = joblib.load(MODEL_PATH)
    except Exception as e:
        raise RuntimeError(f"Error cargando modelo ({MODEL_PATH}): {e}")

    try:
        preprocessor = joblib.load(PREPROC_PATH)
    except Exception as e:
        raise RuntimeError(f"Error cargando preprocesador ({PREPROC_PATH}): {e}")

    return modelo, preprocessor

try:
    modelo, preprocessor = load_modelos()
except Exception as e:
    st.error("❌ No se pudo cargar el modelo o el preprocesador.")
    st.exception(e)  # muestra traceback completo en la UI
    st.stop()

# ---------- UI ----------
st.title("🩺 Predicción de Diabetes (CSV)")
st.markdown("Sube un **CSV** con las columnas que espera el preprocesador. "
            "Usa la plantilla si tienes dudas del esquema.")

# Intentar leer columnas esperadas del preprocesador
expected_cols = None
try:
    if hasattr(preprocessor, "feature_names_in_"):
        expected_cols = list(preprocessor.feature_names_in_)
except Exception:
    expected_cols = None

# Botón para descargar plantilla
if expected_cols:
    plantilla_df = pd.DataFrame(columns=expected_cols)
    st.download_button(
        "📥 Descargar plantilla CSV (columnas esperadas)",
        data=plantilla_df.to_csv(index=False).encode("utf-8"),
        file_name="plantilla_prediccion_diabetes.csv",
        mime="text/csv",
    )
else:
    st.warning("No fue posible leer `feature_names_in_` del preprocesador. "
               "Si luego falla la transformación, revisa la compatibilidad de scikit-learn y re-genera el preprocesador.")

# Alineación de columnas al esquema esperado
def align_columns(df_in: pd.DataFrame, expected: list[str]):
    cols_in = df_in.columns.tolist()
    missing = [c for c in expected if c not in cols_in]
    extra   = [c for c in cols_in if c not in expected]

    for c in missing:
        # Valor neutro: 0 suele funcionar para dummies; si una numérica falta y debe tener rango,
        # idealmente preparar plantilla o completar antes de subir.
        df_in[c] = 0

    if extra:
        df_in = df_in.drop(columns=extra)

    return df_in[expected], missing, extra

uploaded_file = st.file_uploader("📂 Selecciona un archivo CSV", type=["csv"])

if uploaded_file is not None:
    try:
        # Intenta lectura con fallback de encoding
        try:
            df_nuevo = pd.read_csv(uploaded_file)
        except UnicodeDecodeError:
            df_nuevo = pd.read_csv(uploaded_file, encoding="latin-1")

        st.success(f"Archivo: {df_nuevo.shape[0]} filas, {df_nuevo.shape[1]} columnas.")
        st.dataframe(df_nuevo.head())

        if expected_cols:
            df_aligned, missing, extra = align_columns(df_nuevo.copy(), expected_cols)
            if missing:
                st.warning(f"Se agregaron {len(missing)} columnas faltantes con 0: {missing}")
            if extra:
                st.info(f"Se ignoraron {len(extra)} columnas no usadas: {extra}")
        else:
            df_aligned = df_nuevo

        if st.button("🔮 Realizar predicciones"):
            try:
                X_new = preprocessor.transform(df_aligned)
            except Exception as e:
                st.error("Fallo al transformar con el preprocesador. Suele ser por incompatibilidad de versiones "
                         "o por columnas con nombres/formato no esperado.")
                st.exception(e)
                st.stop()

            # Predicción + probas (si el modelo las soporta)
            preds = modelo.predict(X_new)
            if hasattr(modelo, "predict_proba"):
                try:
                    probs = modelo.predict_proba(X_new)[:, 1]
                except Exception:
                    probs = np.full(len(preds), np.nan)
            else:
                probs = np.full(len(preds), np.nan)

            df_result = df_nuevo.copy()
            df_result["Predicción"] = np.where(preds == 1, "Diabético", "No diabético")
            if not np.isnan(probs).all():
                df_result["Probabilidad (%)"] = np.round(probs * 100, 2)

            st.subheader("📊 Resultados")
            st.dataframe(df_result)

            csv_out = df_result.to_csv(index=False).encode("utf-8")
            st.download_button("💾 Descargar resultados (CSV)", csv_out,
                               "predicciones_diabetes.csv", "text/csv")

    except Exception as e:
        st.error("Error general procesando el CSV.")
        st.exception(e)
else:
    st.info("Sube un CSV con los datos a evaluar.")
