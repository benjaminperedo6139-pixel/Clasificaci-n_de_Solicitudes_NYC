# Configuracion de mis variables utilizadas
from pathlib import Path


# General
SEED = 42
ROOT = Path(__file__).resolve().parent.parent

# API
API = "https://data.cityofnewyork.us/resource/erm2-nwe9.json"
COLUMNS = ["unique_key", "created_date", "closed_date", "complaint_type",
           "borough", "open_data_channel_type", "location_type"]
START_DATE = "2025-01-01T00:00:00"
END_DATE = "2025-04-01T00:00:00"   # exclusivo -> enero a marzo 2025
QUERY = {
    "$select": ",".join(COLUMNS),
    "$where": f"created_date >= '{START_DATE}' AND "
              f"created_date < '{END_DATE}' AND unique_key like '%0'",
    "$order": "created_date ASC, unique_key ASC",
}
PAGE_SIZE = 5000
MAX_ROWS = 500_000                 # límite de seguridad de la descarga
CACHE = ROOT / "data" / "nyc311_ene_mar2025.json"

# VARIABLES (solo información disponible al registrar la solicitud)
DATE_COL = "created_date"
CAT_FEATURES = ["complaint_type", "borough", "open_data_channel_type", "location_type", "weekday"]
NUM_FEATURES = ["hour"]
TARGET = "P_24h"

# VALIDACIÓN TEMPORAL (ventana expansiva)
TEST_DAYS = 14        # último bloque de días reservado para la prueba final
N_FOLDS = 3           # número de cortes entrenamiento/validación
VAL_DAYS = 14         # días de cada bloque de validación
GAP_HOURS = 24        # la etiqueta P_24h solo se conoce 24h después de crear la solicitud

# MÉTRICA PRINCIPAL (definida antes de comparar modelos)
PRIMARY_METRIC = "f1"

# SALIDAS (contrato con la interfaz Streamlit)
RESULTS_DIR = ROOT / "results"
METRICS_BY_FOLD_FILE = RESULTS_DIR / "metrics_by_fold.csv"
FOLDS_FILE = RESULTS_DIR / "folds.csv"
TEST_METRICS_FILE = RESULTS_DIR / "test_metrics.json"
CONFUSION_MATRIX_FILE = RESULTS_DIR / "confusion_matrix.csv"
BEST_MODEL_FILE = RESULTS_DIR / "best_model.joblib"

## MLFLOW (Model Versioning)
MLFLOW_TRACKING_URL = "mlruns/"
#MLFLOW_TRACKING_URL = "http://mlflow.vanotole-lab.com"
MLFLOW_EXPERIMENT_NAME = "nyc311_p24h"
