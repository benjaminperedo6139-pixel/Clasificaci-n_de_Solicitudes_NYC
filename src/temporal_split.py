"""
Validación temporal (ventana expansiva).            RESPONSABLE: Integrante A

Contrato (no cambiar la firma sin avisar al equipo):

    folds, test_idx, folds_info = make_temporal_folds(df)

- df: DataFrame que devuelve Dataset().load_df_target(); debe tener DATE_COL.
- folds: list[tuple[np.ndarray, np.ndarray]] -> (train_idx, valid_idx) posicionales
         sobre df ordenado por DATE_COL. Mismo formato que TimeSeriesSplit.split(),
         así puede pasarse directo como `cv=` a GridSearchCV si se desea.
- test_idx: np.ndarray con los índices del bloque final reservado (últimos TEST_DAYS).
- folds_info: pd.DataFrame con una fila por corte (+ una fila "test"):
         fold | train_start | train_end | valid_start | valid_end | n_train | n_valid

Reglas que debe cumplir la implementación:
1. Ordenar por DATE_COL antes de cortar.
2. Reservar los últimos TEST_DAYS días como prueba final (nunca se usan para elegir modelo).
3. Con los días anteriores, N_FOLDS cortes de VAL_DAYS días cada uno, ventana EXPANSIVA
   (cada corte entrena con todo el pasado disponible).
4. GAP_HOURS: en cada corte, solo se entrena con solicitudes cuya etiqueta ya se conocía
   al inicio de la validación:  created_date + GAP_HOURS <= valid_start.
   Verificarlo con assert.
"""

import numpy as np
import pandas as pd

from src.config import DATE_COL, TEST_DAYS, N_FOLDS, VAL_DAYS, GAP_HOURS


def make_temporal_folds(
        df: pd.DataFrame,
        n_folds: int = N_FOLDS,
        val_days: int = VAL_DAYS,
        test_days: int = TEST_DAYS,
        gap_hours: int = GAP_HOURS,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], np.ndarray, pd.DataFrame]:
    """Construye los cortes temporales de entrenamiento/validación y el bloque de prueba."""
    raise NotImplementedError("Integrante A: implementar en la rama feature/datos-validacion")
