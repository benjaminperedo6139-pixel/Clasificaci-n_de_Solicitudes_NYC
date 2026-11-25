# Configuracion de mis variables utilizadas
from pathlib import Path


# General
SEED = 42

# API
API = "https://data.cityofnewyork.us/resource/erm2-nwe9.json"
COLUMNS = ["unique_key", "created_date", "closed_date", "complaint_type",
           "borough", "open_data_channel_type", "location_type"]
QUERY = {
    "$select": ",".join(COLUMNS),
    "$where": "created_date >= '2025-01-01T00:00:00' AND "
              "created_date < '2025-02-01T00:00:00' AND unique_key like '%0'",
    "$order": "created_date ASC, unique_key ASC",
}
CACHE = Path(__file__).resolve().parent / "data" / "nyc311_enero2025.json"

# VARIABLES
CAT_FEATURES = ["complaint_type", "borough", "open_data_channel_type", "location_type", "weekday"]
NUM_FEATURES = ["hour"]
TARGET = "P_24h"

# SPLIT
TEST_SIZE = 0.2

## MLFLOW (Model Versioning)
MLFLOW_TRACKING_URL = "mlruns/"
#MLFLOW_TRACKING_URL = "http://mlflow.vanotole-lab.com"
MLFLOW_EXPERIMENT_NAME = "housing_ml"