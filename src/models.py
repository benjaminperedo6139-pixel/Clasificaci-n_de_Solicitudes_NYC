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
    # Cada entrada: nombre -> (pipeline, rejilla). GridSearchCV prueba todas las combinaciones
    # de la rejilla; lo que no aparece en ella queda fijo en el pipeline.
    models = {
        # Referencia sin aprendizaje. strategy: "most_frequent" predice siempre la clase más común;
        # "stratified" predice al azar respetando la proporción de clases
        "baseline": (
            make_pipeline(DummyClassifier(random_state=SEED)),
            {"model__strategy": ["most_frequent", "stratified"]},
        ),
        # Modelo lineal. C: inverso de la regularización (C pequeño = modelo más simple), en escala
        # logarítmica alrededor del valor por defecto (1). max_iter=1000 (por defecto 100) es solo
        # un tope de margen: el solver se detiene antes si ya convergió
        # No se prueba class_weight="balanced": las clases están casi equilibradas (~57/43) y resta
        # peso a la clase positiva (la mayoritaria), lo que no suele favorecer a su F1
        "logistic_regression": (
            make_pipeline(LogisticRegression(max_iter=1000, random_state=SEED)),
            {"model__C": [0.01, 0.1, 1, 10]},
        ),
        # Muchos árboles de decisión que votan; la semilla hace repetible el azar.
        # n_jobs=-1 entrena los árboles en todos los núcleos; se usan los 100 árboles por defecto
        # min_samples_leaf: mínimo de filas por hoja (más alto = árboles más pequeños, menos sobreajuste)
        # max_features: columnas sorteadas en cada división; "sqrt" son ~16 de ~270 y 0.2 son ~54
        "random_forest": (
            make_pipeline(RandomForestClassifier(n_jobs=-1, random_state=SEED)),
            {
                "model__min_samples_leaf": [1, 5, 20],
                "model__max_features": ["sqrt", 0.2],
            },
        ),
        # Clasifica según los vecinos más cercanos; no usa azar, por eso no lleva semilla
        # n_neighbors: cuántos vecinos votan (pocos = sensible al ruido, muchos = más suave);
        # impares para evitar empates con voto uniforme
        # weights: "uniform" todos valen igual; "distance" los más cercanos pesan más
        "knn": (
            make_pipeline(KNeighborsClassifier()),
            {
                "model__n_neighbors": [5, 15, 51, 101],
                "model__weights": ["uniform", "distance"],
            },
        ),
        # SVM lineal: SVC con kernel es inviable con ~73 000 filas
        # C: penalización por errores (C pequeño = más regularización). Va una década por debajo
        # de la rejilla logística porque los C grandes cuestan más y regularizan muy poco
        # El max_iter por defecto (1000) es suficiente: con más filas que columnas converge rápido
        "svm": (
            make_pipeline(LinearSVC(random_state=SEED)),
            {"model__C": [0.001, 0.01, 0.1, 1]},
        ),
    }
    return models
