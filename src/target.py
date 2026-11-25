"""
Construcción del target P_24h.                      RESPONSABLE: Integrante A

P_24h = 1  si la solicitud se cerró en menos de HORIZON_HOURS desde su creación.
P_24h = 0  si tardó HORIZON_HOURS o más, o si sigue abierta y ya pasaron HORIZON_HOURS.

Cuándo se conoce la etiqueta:
    Siempre se conoce, como muy tarde, en  created_date + HORIZON_HOURS
    (para entonces o ya se cerró -> 1, o no se cerró a tiempo -> 0).
    Esa fecha se guarda en la columna `label_known_at` y la usa temporal_split.py
    para comprobar que ningún corte entrena con etiquetas que aún no se conocían.
"""

import numpy as np
import pandas as pd

from src.config import DATE_COL, TARGET, HORIZON_HOURS, MIN_DURATION_SECONDS


def make_target(
        df: pd.DataFrame,
        as_of: pd.Timestamp | str | None = None,
        horizon_hours: int = HORIZON_HOURS,
        min_seconds: int = MIN_DURATION_SECONDS,
) -> pd.DataFrame:
    """
    Argumentos:
        df - DataFrame crudo con created_date y closed_date
        as_of - fecha en que se descargaron los datos. Las solicitudes abiertas creadas
                menos de `horizon_hours` antes de esta fecha tienen etiqueta desconocida
                y se descartan. Si es None se usa el instante actual.
    Return: DataFrame filtrado con TARGET, label_known_at, hour y weekday.
    """
    df = df.copy().drop_duplicates(subset="unique_key")
    horizon = pd.Timedelta(hours=horizon_hours)
    as_of = pd.Timestamp.now() if as_of is None else pd.Timestamp(as_of).tz_localize(None)

    # Pasar a tipo datetime
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df["closed_date"] = pd.to_datetime(df["closed_date"])

    # Duración (horas); NaN si sigue abierta
    seconds = (df["closed_date"] - df[DATE_COL]).dt.total_seconds()
    df["duration_hours"] = seconds / 3600
    is_open = df["closed_date"].isna()

    # Filtros
    # - cerradas con duración <= min_seconds (incluye negativas): cierres automáticos o errores
    # - abiertas cuya ventana de 24 h aún no termina en `as_of`: etiqueta desconocida
    valid_closed = ~is_open & (seconds > min_seconds)
    known_open = is_open & (df[DATE_COL] + horizon <= as_of)
    df = df[valid_closed | known_open].copy()

    # Target: abiertas (ya pasadas las 24 h) -> 0
    df[TARGET] = np.where(df["duration_hours"] < horizon_hours, 1, 0)

    # Momento en que la etiqueta ya es conocida
    df["label_known_at"] = df[DATE_COL] + horizon

    # Variables disponibles al registrar la solicitud
    df["hour"] = df[DATE_COL].dt.hour
    df["weekday"] = df[DATE_COL].dt.day_name()

    return df.sort_values(DATE_COL).reset_index(drop=True)
