
# Programacion orientada a objetos

import pandas as pd
import json
import requests
import os
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from datetime import datetime, timezone


from src.config import SEED, CACHE, QUERY, API, COLUMNS
from src.config import NUM_FEATURES, CAT_FEATURES, TARGET
from src.target import make_target

class Dataset:

    # Constructor
    # Atributos
    def __init__(self, num_samples: int = None, seed: int = 42):
        self.num_samples = num_samples
        self.seed = seed

    # Importacion de datos (método)
    def load_data(self):
        """
        Método de importación de datos
        
        Return: DataFrame
        """

        if CACHE.exists():
            payload = json.loads(CACHE.read_text())
            if payload["query"] != QUERY or payload["api"] != API:
                raise ValueError("La caché no corresponde a esta consulta. Renómbrala para descargar de nuevo.")
            return payload
        
        session = requests.Session()
        session.mount("https://", HTTPAdapter(max_retries=Retry(
            total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])))
        token = os.environ.get("SOCRATA_APP_TOKEN")
        if token:
            session.headers["X-App-Token"] = token
        rows = []
        for offset in range(0, 100000, 5000):
            response = session.get(API, params={**QUERY, "$limit": 5000,
                                                "$offset": offset}, timeout=90)
            response.raise_for_status()
            page = response.json()
            if not isinstance(page, list):
                raise ValueError("Respuesta inesperada de Socrata.")
            rows.extend(page)
            if len(page) < 5000:
                break
        else:
            raise ValueError("Se alcanzó el límite de seguridad; no se guarda una muestra truncada.")
        if not rows:
            raise ValueError("La API no devolvió filas. Revisa la consulta.")
        payload = {"api": API, "query": QUERY,
                "downloaded_at": datetime.now(timezone.utc).isoformat(), "rows": rows}
        CACHE.parent.mkdir(exist_ok=True)
        temporary = CACHE.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False))
        temporary.replace(CACHE)
        
        return payload
    
    def load_df(self):

        payload = self.load_data()

        # save to DataFrame
        df = pd.DataFrame(payload["rows"]).reindex(columns=COLUMNS)

        return df
    
    def load_df_target(self):

        df = self.load_df()

        df_target = make_target(df)

        return df_target
    
    def load_xy(self):

        df = self.load_df_target()

        X = df[NUM_FEATURES + CAT_FEATURES]
        y = df[TARGET]

        return X, y


