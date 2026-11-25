"""
Modelos y rejillas de hiperparámetros.              RESPONSABLE: Integrante B

Contrato (no cambiar la firma sin avisar al equipo):

    models = get_models()

- Devuelve dict[str, tuple[Pipeline, dict]]:
      {"nombre_modelo": (pipeline, param_grid), ...}
- pipeline: sklearn Pipeline con pasos ("preprocessor", build_preprocessor()) y ("model", ...)
- param_grid: dict para GridSearchCV, con claves prefijadas "model__<param>".
  Ej: {"model__C": [0.1, 1, 10]}
- Debe incluir obligatoriamente un modelo de referencia "baseline" (DummyClassifier).

Modelos sugeridos: baseline (DummyClassifier), logistic_regression, random_forest,
knn, svm (LinearSVC o SVC sobre submuestra). Todos son CLASIFICADORES.

Para el README, documentar por modelo: ventaja esperada, limitación y rejilla probada.
"""

from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC

from src.config import SEED
from src.preprocessor import build_preprocessor


def make_pipeline(estimator) -> Pipeline:
    """Envuelve un estimador con el preprocesador común (se ajusta solo con train en cada corte)."""
    return Pipeline([
        ("preprocessor", build_preprocessor()),
        ("model", estimator),
    ])


def get_models() -> dict[str, tuple[Pipeline, dict]]:
    """Devuelve los modelos a comparar y sus rejillas de hiperparámetros."""
    # Cada entrada: nombre -> (pipeline, rejilla). Una rejilla vacía {} usa los valores por defecto.
    models = {
        "baseline": (
            make_pipeline(DummyClassifier(random_state=SEED)),
            {"model__strategy": ["most_frequent", "stratified"]},
        ),
        # Modelo lineal; max_iter=1000 (por defecto 100) para que el solver alcance a converger
        "logistic_regression": (
            make_pipeline(LogisticRegression(max_iter=1000, random_state=SEED)),
            {},
        ),
        # Muchos árboles de decisión que votan; la semilla hace repetible el azar
        "random_forest": (
            make_pipeline(RandomForestClassifier(random_state=SEED)),
            {},
        ),
        # Clasifica según los vecinos más cercanos; no usa azar, por eso no lleva semilla
        "knn": (
            make_pipeline(KNeighborsClassifier()),
            {},
        ),
        # SVM lineal: SVC con kernel es inviable con ~73 000 filas
        # max_iter=5000 (por defecto 1000) da margen para converger
        "svm": (
            make_pipeline(LinearSVC(max_iter=5000, random_state=SEED)),
            {},
        ),
    }
    return models
