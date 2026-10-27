
"""Ruta centralizada para nuestro proyecto."""

from pathlib import Path

# Directorio raíz del proyecto
BASE_DIR = Path(__file__).resolve().parents[1]

# ==============================
# DATOS
# ==============================

DATASET_DIR = BASE_DIR / "Data"

RAW_DATA_PATH = DATASET_DIR / "riesgo_crediticio.csv"

DATA_PREP_DIR = DATASET_DIR / "processed"

# Reportes y gráficos del análisis exploratorio
EDA_REPORT_DIR = DATASET_DIR / "eda_report"

# Etapas del procesamiento de datos
INGESTION_PATH = DATA_PREP_DIR / "raw_ingested.csv"
VALIDATION_PATH = DATA_PREP_DIR / "validated.csv"
EDA_PATH = DATA_PREP_DIR / "eda.csv"
CLEANING_PATH = DATA_PREP_DIR / "cleaned.csv"
FEATURED_PATH = DATA_PREP_DIR / "featured.csv"

# Datos preparados para entrenamiento y prueba
PREPROCESSED_TRAIN_PATH = DATA_PREP_DIR / "train.csv"
PREPROCESSED_TEST_PATH = DATA_PREP_DIR / "test.csv"

# ==============================
# MODELOS Y ARTEFACTOS
# ==============================

# Carpeta para el modelo base y los resultados de evaluación
ARTIFACT_DIR = BASE_DIR / "models"

# Modelo base entrenado
MODEL_PATH = ARTIFACT_DIR / "modelo_xgboost.pkl"

# Archivo de métricas
METRICS_PATH = ARTIFACT_DIR / "metrics.json"

# Modelo ajustado: misma ubicación que utiliza la API
TUNED_MODEL_PATH = (
    BASE_DIR
    / "inference"
    / "artifacts"
    / "tuned_model.pkl"
)

# ==============================
# CREACIÓN DE DIRECTORIOS
# ==============================

DATA_PREP_DIR.mkdir(parents=True, exist_ok=True)
EDA_REPORT_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

# Asegurar que exista la carpeta donde se guarda el modelo ajustado
TUNED_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)