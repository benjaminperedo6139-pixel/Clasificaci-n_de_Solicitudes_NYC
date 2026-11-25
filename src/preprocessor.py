from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from src.config import CAT_FEATURES, NUM_FEATURES
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

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

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", imputer_num),("scaler", scaler)]), NUM_FEATURES),
            ("cat", Pipeline([('imputer', imputer_cat),("encoder", encoder)]), CAT_FEATURES),
        ]
    )

    return preprocessor