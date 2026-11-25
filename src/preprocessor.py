import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from src.config import CAT_FEATURES, NUM_FEATURES
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

def unificar_nulos(X: pd.DataFrame) -> pd.DataFrame:
    """
    Convierte cualquier valor faltante (None, pd.NA) en np.nan,
    que es el único que SimpleImputer reconoce por defecto.
    """
    return X.astype(object).where(X.notna(), np.nan)

def build_preprocessor() -> ColumnTransformer:
    """
    Función que escala columnas numéricas y codifica columnas categóricas,

    Returns: ColumnTransformer. Objeto que condensa el flujo (preprocesamiento)
    de los datos de entrada.

    """

    # Codificador
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    # Escalador
    scaler = StandardScaler()

    # Imputador
    # hour es la única numérica: no hay otras columnas para buscar vecinos, basta la mediana
    imputer_num = SimpleImputer(strategy="median")
    imputer_cat = SimpleImputer(strategy="constant", fill_value="sin_dato")

    # Unificador de nulos (feature_names_out conserva los nombres de columna)
    nulos = FunctionTransformer(unificar_nulos, feature_names_out="one-to-one")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", imputer_num),("scaler", scaler)]), NUM_FEATURES),
            ("cat", Pipeline([("nulos", nulos),('imputer', imputer_cat),("encoder", encoder)]), CAT_FEATURES),
        ]
    )

    return preprocessor