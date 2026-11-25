"""
Evaluación, selección de modelo y prueba final.     RESPONSABLE: Integrante C

Contrato (no cambiar la firma sin avisar al equipo):

    results = run_experiment(X, y, folds, models)
    best_name, best_params = select_best(results)
    test_metrics = evaluate_on_test(model, X_dev, y_dev, X_test, y_test)

- X, y: features y target del periodo de desarrollo (todo excepto el bloque de prueba),
        ordenados por fecha; `folds` son índices posicionales sobre ellos.
- folds: salida de make_temporal_folds (src/temporal_split.py).
- models: salida de get_models() (src/models.py).
- results: pd.DataFrame, una fila por (modelo, configuración, corte) con columnas:
        model | params | fold | accuracy | precision | recall | f1 | roc_auc
- La selección se hace con PRIMARY_METRIC (F1) promediado sobre los cortes.
- evaluate_on_test: reentrena con X_final/y_final (desarrollo cuya etiqueta ya se conocía al
  inicio de la prueba, ver get_final_train_idx), evalúa UNA vez en el bloque reservado y
  devuelve métricas + matriz de confusión.

Archivos de salida (contrato con Streamlit, rutas en config.py):
    METRICS_BY_FOLD_FILE, FOLDS_FILE, TEST_METRICS_FILE, CONFUSION_MATRIX_FILE, BEST_MODEL_FILE
"""

import pandas as pd

from src.config import PRIMARY_METRIC


def run_experiment(X: pd.DataFrame, y: pd.Series, folds: list, models: dict) -> pd.DataFrame:
    """Evalúa cada modelo y cada configuración en los MISMOS cortes temporales."""
    raise NotImplementedError("Integrante C: implementar en la rama feature/evaluacion-reporte")


def select_best(results: pd.DataFrame, metric: str = PRIMARY_METRIC) -> tuple[str, dict]:
    """Elige el modelo y configuración con mayor `metric` promedio entre cortes."""
    raise NotImplementedError("Integrante C: implementar en la rama feature/evaluacion-reporte")


def evaluate_on_test(model, X_dev, y_dev, X_test, y_test) -> dict:
    """Reentrena con todo el desarrollo y evalúa en el bloque final reservado."""
    raise NotImplementedError("Integrante C: implementar en la rama feature/evaluacion-reporte")
