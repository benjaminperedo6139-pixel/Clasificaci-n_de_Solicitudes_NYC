"""
Validación temporal (ventana expansiva).            RESPONSABLE: Integrante A

Contrato:

    folds, test_idx, folds_info = make_temporal_folds(df)
    final_idx = get_final_train_idx(df)

- df: DataFrame que devuelve Dataset().load_df_target(), ORDENADO por DATE_COL.
- folds: list[tuple[np.ndarray, np.ndarray]] -> (train_idx, valid_idx) posicionales.
         Mismo formato que TimeSeriesSplit.split(), se puede pasar como `cv=` a GridSearchCV.
- test_idx: índices del bloque final reservado (últimos TEST_DAYS días).
- folds_info: DataFrame con una fila por corte + una fila "test".
- get_final_train_idx: índices para reentrenar el modelo elegido antes de la prueba final
  (todo lo anterior al bloque de prueba cuya etiqueta ya se conocía al inicio del bloque).

Por qué cortes por DÍAS y no TimeSeriesSplit directamente:
    TimeSeriesSplit corta por número de filas y su `gap` también se mide en filas. Como el
    volumen diario varía (≈720 a 1500 solicitudes), cortar por fecha garantiza bloques de
    calendario completos y un gap de exactamente 24 h, que es lo que tarda en conocerse P_24h.

Regla anti-fuga (se verifica con assert en cada corte):
    una solicitud entra al entrenamiento solo si  created_date + GAP_HOURS <= inicio_validación,
    es decir, si su etiqueta ya se conocía en la fecha del corte.
"""

import numpy as np
import pandas as pd

from src.config import DATE_COL, TARGET, TEST_DAYS, N_FOLDS, VAL_DAYS, GAP_HOURS


def _check_sorted(dates: pd.Series) -> None:
    if not dates.is_monotonic_increasing:
        raise ValueError(f"df debe estar ordenado por {DATE_COL} (usar sort_values + reset_index).")


def _test_start(dates: pd.Series, test_days: int) -> pd.Timestamp:
    """Inicio del bloque de prueba: medianoche, `test_days` antes del fin del último día."""
    last_day_end = dates.max().normalize() + pd.Timedelta(days=1)
    return last_day_end - pd.Timedelta(days=test_days)


def _check_labels_known(df: pd.DataFrame, train_idx: np.ndarray, cutoff: pd.Timestamp, name: str) -> None:
    """Verifica que todas las etiquetas de entrenamiento ya se conocían en `cutoff`."""
    known_at = df["label_known_at"] if "label_known_at" in df else None
    created = pd.to_datetime(df[DATE_COL]).iloc[train_idx]
    assert (created < cutoff).all(), f"{name}: hay solicitudes de entrenamiento posteriores al corte"
    if known_at is not None:
        assert (known_at.iloc[train_idx] <= cutoff).all(), \
            f"{name}: hay etiquetas de entrenamiento que aún no se conocían en {cutoff}"


def _rate(df: pd.DataFrame, idx: np.ndarray) -> float:
    return round(float(df[TARGET].iloc[idx].mean()), 3) if TARGET in df and len(idx) else np.nan


def make_temporal_folds(
        df: pd.DataFrame,
        n_folds: int = N_FOLDS,
        val_days: int = VAL_DAYS,
        test_days: int = TEST_DAYS,
        gap_hours: int = GAP_HOURS,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], np.ndarray, pd.DataFrame]:
    """Construye los cortes temporales de entrenamiento/validación y el bloque de prueba."""
    dates = pd.to_datetime(df[DATE_COL]).reset_index(drop=True)
    _check_sorted(dates)
    gap = pd.Timedelta(hours=gap_hours)
    val_len = pd.Timedelta(days=val_days)

    test_start = _test_start(dates, test_days)
    first_valid_start = test_start - n_folds * val_len
    if dates.min() + gap >= first_valid_start:
        raise ValueError("No hay suficientes días para entrenar el primer corte; "
                         "reduzca n_folds, val_days o test_days.")

    folds, rows = [], []
    for k in range(n_folds):
        valid_start = first_valid_start + k * val_len
        valid_end = valid_start + val_len

        # Ventana expansiva: todo el pasado cuya etiqueta ya se conocía en valid_start
        train_idx = np.flatnonzero((dates + gap <= valid_start).to_numpy())
        valid_idx = np.flatnonzero(((dates >= valid_start) & (dates < valid_end)).to_numpy())

        _check_labels_known(df, train_idx, valid_start, f"corte {k + 1}")
        assert len(np.intersect1d(train_idx, valid_idx)) == 0
        assert (dates.iloc[valid_idx] < test_start).all(), "la validación invade el bloque de prueba"

        folds.append((train_idx, valid_idx))
        rows.append({
            "fold": str(k + 1),
            "train_start": dates.iloc[train_idx[0]], "train_end": dates.iloc[train_idx[-1]],
            "valid_start": valid_start, "valid_end": valid_end - pd.Timedelta(seconds=1),
            "n_train": len(train_idx), "n_valid": len(valid_idx),
            "p24_train": _rate(df, train_idx), "p24_valid": _rate(df, valid_idx),
        })

    test_idx = np.flatnonzero((dates >= test_start).to_numpy())
    final_idx = get_final_train_idx(df, test_days=test_days, gap_hours=gap_hours)
    rows.append({
        "fold": "test",
        "train_start": dates.iloc[final_idx[0]], "train_end": dates.iloc[final_idx[-1]],
        "valid_start": test_start, "valid_end": dates.iloc[test_idx[-1]],
        "n_train": len(final_idx), "n_valid": len(test_idx),
        "p24_train": _rate(df, final_idx), "p24_valid": _rate(df, test_idx),
    })

    return folds, test_idx, pd.DataFrame(rows)


def get_final_train_idx(
        df: pd.DataFrame,
        test_days: int = TEST_DAYS,
        gap_hours: int = GAP_HOURS,
) -> np.ndarray:
    """Índices para reentrenar el modelo elegido antes de evaluar en el bloque de prueba."""
    dates = pd.to_datetime(df[DATE_COL]).reset_index(drop=True)
    _check_sorted(dates)
    test_start = _test_start(dates, test_days)
    final_idx = np.flatnonzero((dates + pd.Timedelta(hours=gap_hours) <= test_start).to_numpy())
    _check_labels_known(df, final_idx, test_start, "entrenamiento final")
    return final_idx
