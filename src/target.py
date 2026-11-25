import numpy as np
import pandas as pd

def make_target(
        df: pd.DataFrame,
):

    # Pasar a tipo datetime
    df["created_date"] = pd.to_datetime(df["created_date"])
    df["closed_date"] = pd.to_datetime(df["closed_date"])

    # Diff
    df["Diff"] = (df["closed_date"] - df["created_date"])

    # Filter
    df_filtered = df.copy()
    df_filtered = df_filtered[df_filtered["Diff"].dt.total_seconds() > 150]

    # Target
    df_filtered["P_24h"] = np.where(df_filtered["Diff"].dt.days>0, 0, 1)

    # Hora y dia de la semana
    df_filtered["hour"] = df_filtered["created_date"].dt.hour
    df_filtered["weekday"] = df_filtered["created_date"].dt.day_name()

    return df_filtered

