"""
Interfaz gráfica del experimento NYC 311 · P_24h.   RESPONSABLE: Benjamin Peredo Alonso

Ejecutar (desde la raíz del repo, después de `python train.py`):

    streamlit run app.py

La app NO reentrena: solo lee los archivos que deja train.py en results/
(rutas definidas en src/config.py, sección "SALIDAS"):

    FOLDS_FILE             folds.csv            cortes temporales + fila "test"
    METRICS_BY_FOLD_FILE   metrics_by_fold.csv  model | params | fold | accuracy | precision | recall | f1 | roc_auc
    TEST_METRICS_FILE      test_metrics.json    métricas en el bloque de prueba
    CONFUSION_MATRIX_FILE  confusion_matrix.csv matriz 2x2 (filas = real, columnas = predicho; orden 0, 1)
    BEST_MODEL_FILE        best_model.joblib    Pipeline (preprocesador + modelo) reentrenado
"""

import json
from datetime import datetime

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st

from src.config import (
    BEST_MODEL_FILE, CACHE, CAT_FEATURES, CONFUSION_MATRIX_FILE, FOLDS_FILE,
    METRICS_BY_FOLD_FILE, NUM_FEATURES, PRIMARY_METRIC, TARGET, TEST_METRICS_FILE,
)

WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
WEEKDAYS_ES = dict(zip(WEEKDAYS, ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"]))
METRIC_COLS = ["accuracy", "precision", "recall", "f1", "roc_auc"]

st.set_page_config(page_title="NYC 311 · P_24h", page_icon="🗽", layout="wide")


# ---------------------------------------------------------------- carga de artefactos
@st.cache_resource
def load_model(path):
    return joblib.load(path)


@st.cache_data
def load_folds(path):
    df = pd.read_csv(path, dtype={"fold": str})
    for c in ["train_start", "train_end", "valid_start", "valid_end"]:
        df[c] = pd.to_datetime(df[c])
    return df


@st.cache_data
def load_metrics(path):
    df = pd.read_csv(path)
    df["fold"] = df["fold"].astype(str)
    df["params"] = df["params"].astype(str)
    return df


@st.cache_data
def load_test_metrics(path):
    with open(path) as f:
        data = json.load(f)
    # acepta {"f1": .., ...} o {"metrics": {...}, ...}
    data = data.get("metrics", data)
    return {k: float(v) for k, v in data.items() if isinstance(v, (int, float))}


@st.cache_data
def load_confusion(path):
    df = pd.read_csv(path)
    values = df.select_dtypes("number").to_numpy()[-2:, -2:]   # ignora columnas de etiquetas
    return values.astype(int)


@st.cache_data
def load_raw_data():
    """Datos de la caché local (solo para la pestaña de exploración). No descarga nada."""
    if not CACHE.exists():
        return None
    from src.io import Dataset
    return Dataset().load_df_target()


def encoder_categories(model) -> dict[str, list]:
    """Categorías vistas en el entrenamiento, leídas del OneHotEncoder del pipeline."""
    try:
        cat = model.named_steps["preprocessor"].named_transformers_["cat"]
        enc = cat.named_steps["encoder"]
        return {col: [c for c in cats if c != "sin_dato"] for col, cats in zip(CAT_FEATURES, enc.categories_)}
    except (AttributeError, KeyError):
        return {}


def score_positive(model, X) -> tuple[float, bool]:
    """Devuelve (score, es_probabilidad). LinearSVC no tiene predict_proba."""
    if hasattr(model, "predict_proba"):
        return float(model.predict_proba(X)[0, 1]), True
    return float(model.decision_function(X)[0]), False


files = {
    "folds.csv": FOLDS_FILE, "metrics_by_fold.csv": METRICS_BY_FOLD_FILE,
    "test_metrics.json": TEST_METRICS_FILE, "confusion_matrix.csv": CONFUSION_MATRIX_FILE,
    "best_model.joblib": BEST_MODEL_FILE,
}
missing = [name for name, path in files.items() if not path.exists()]

# ---------------------------------------------------------------- encabezado
st.title("🗽 NYC 311 · ¿La solicitud se cierra en menos de 24 h?")
st.caption(
    "Predicción al momento de registrar la solicitud. Validación temporal con ventana expansiva, "
    f"gap de 24 h y bloque de prueba reservado. Métrica principal: **{PRIMARY_METRIC.upper()}** "
    f"de la clase `{TARGET} = 1`."
)

if missing:
    st.warning(
        "Faltan resultados en `results/`: " + ", ".join(f"`{m}`" for m in missing)
        + ". Corre `python train.py` para generarlos. Las pestañas que dependen de ellos aparecen vacías."
    )

model = load_model(BEST_MODEL_FILE) if BEST_MODEL_FILE.exists() else None

with st.sidebar:
    st.header("Modelo elegido")
    if model is not None:
        est = model.named_steps.get("model", model) if hasattr(model, "named_steps") else model
        st.markdown(f"**{type(est).__name__}**")
        defaults = type(est)().get_params()
        changed = {k: v for k, v in est.get_params().items() if defaults.get(k) != v}
        st.json(changed or {"(valores por defecto)": True})
    else:
        st.info("Aún no hay `best_model.joblib`.")
    st.divider()
    st.markdown(
        "**Variables de entrada** (conocidas al registrar):  \n"
        + ", ".join(f"`{c}`" for c in NUM_FEATURES + CAT_FEATURES)
    )

tab_val, tab_cmp, tab_test, tab_pred, tab_eda = st.tabs(
    ["📅 Validación temporal", "📊 Comparación de modelos", "🎯 Prueba final",
     "🔮 Predecir solicitud", "🔎 Datos"]
)

# ---------------------------------------------------------------- 1. cortes
with tab_val:
    if not FOLDS_FILE.exists():
        st.info("Falta `folds.csv`.")
    else:
        folds = load_folds(FOLDS_FILE)
        folds["corte"] = np.where(folds["fold"] == "test", "Prueba final", "Corte " + folds["fold"])

        bars = pd.concat([
            folds.assign(fase="Entrenamiento", inicio=folds["train_start"], fin=folds["train_end"], n=folds["n_train"]),
            folds.assign(fase=np.where(folds["fold"] == "test", "Prueba", "Validación"),
                         inicio=folds["valid_start"], fin=folds["valid_end"], n=folds["n_valid"]),
        ])[["corte", "fase", "inicio", "fin", "n"]]

        chart = alt.Chart(bars).mark_bar(cornerRadius=3, height=22).encode(
            x=alt.X("inicio:T", title="Fecha"),
            x2="fin:T",
            y=alt.Y("corte:N", sort=list(folds["corte"]), title=None),
            color=alt.Color("fase:N", scale=alt.Scale(
                domain=["Entrenamiento", "Validación", "Prueba"],
                range=["#4C78A8", "#F58518", "#E45756"]), title=None),
            tooltip=["corte", "fase", alt.Tooltip("inicio:T", format="%Y-%m-%d %H:%M"),
                     alt.Tooltip("fin:T", format="%Y-%m-%d %H:%M"), alt.Tooltip("n:Q", format=",")],
        ).properties(height=220)
        st.altair_chart(chart, width="stretch")

        show = folds[["corte", "train_start", "train_end", "valid_start", "valid_end",
                      "n_train", "n_valid", "p24_train", "p24_valid"]].rename(columns={
            "train_start": "Train inicio", "train_end": "Train fin",
            "valid_start": "Val/Test inicio", "valid_end": "Val/Test fin",
            "n_train": "n train", "n_valid": "n val/test",
            "p24_train": "% P_24h train", "p24_valid": "% P_24h val/test",
        })
        st.dataframe(
            show, hide_index=True, width="stretch",
            column_config={
                "% P_24h train": st.column_config.NumberColumn(format="percent"),
                "% P_24h val/test": st.column_config.NumberColumn(format="percent"),
            },
        )
        st.caption("Una solicitud entra al entrenamiento solo si `created_date + 24 h ≤ inicio de validación`: "
                   "su etiqueta ya se conocía en la fecha del corte.")

# ---------------------------------------------------------------- 2. comparación
with tab_cmp:
    if not METRICS_BY_FOLD_FILE.exists():
        st.info("Falta `metrics_by_fold.csv`.")
    else:
        res = load_metrics(METRICS_BY_FOLD_FILE)
        available = [c for c in METRIC_COLS if c in res.columns]
        metric = st.selectbox("Métrica", available,
                              index=available.index(PRIMARY_METRIC) if PRIMARY_METRIC in available else 0)

        summary = (res.groupby(["model", "params"])[metric]
                   .agg(promedio="mean", desv="std", minimo="min").reset_index()
                   .sort_values("promedio", ascending=False))
        best = summary.loc[summary.groupby("model")["promedio"].idxmax()].sort_values("promedio", ascending=False)
        winner = best.iloc[0]

        c1, c2, c3 = st.columns(3)
        c1.metric("Mejor modelo", winner["model"])
        c2.metric(f"{metric} promedio", f"{winner['promedio']:.3f}")
        if "baseline" in best["model"].values:
            base = best.loc[best["model"] == "baseline", "promedio"].iloc[0]
            c3.metric("vs. baseline", f"{base:.3f}", delta=f"{winner['promedio'] - base:+.3f}")
        st.caption(f"Configuración: `{winner['params']}`")

        st.subheader("Mejor configuración de cada modelo, por corte")
        best_runs = res.merge(best[["model", "params"]], on=["model", "params"])
        line = alt.Chart(best_runs).mark_line(point=True).encode(
            x=alt.X("fold:N", title="Corte"),
            y=alt.Y(f"{metric}:Q", scale=alt.Scale(zero=False), title=metric),
            color=alt.Color("model:N", title="Modelo"),
            tooltip=["model", "params", "fold", alt.Tooltip(f"{metric}:Q", format=".3f")],
        ).properties(height=320)
        st.altair_chart(line, width="stretch")

        pivot = best_runs.pivot_table(index="model", columns="fold", values=available).round(3)
        st.dataframe(pivot, width="stretch")

        with st.expander(f"Todas las configuraciones ({len(summary)}) ordenadas por {metric} promedio"):
            st.dataframe(summary.round(4), hide_index=True, width="stretch")

# ---------------------------------------------------------------- 3. prueba final
with tab_test:
    if TEST_METRICS_FILE.exists():
        tm = load_test_metrics(TEST_METRICS_FILE)
        cols = st.columns(max(len(tm), 1))
        for col, (k, v) in zip(cols, tm.items()):
            col.metric(k, f"{v:.3f}")
    else:
        st.info("Falta `test_metrics.json`.")

    if CONFUSION_MATRIX_FILE.exists():
        cm = load_confusion(CONFUSION_MATRIX_FILE)
        tn, fp, fn, tp = cm.ravel()
        total = cm.sum()
        labels = ["≥ 24 h (0)", "< 24 h (1)"]
        cm_df = pd.DataFrame([
            {"Real": labels[i], "Predicho": labels[j], "n": int(cm[i, j]),
             "tipo": [["VN", "FP"], ["FN", "VP"]][i][j]}
            for i in range(2) for j in range(2)
        ])
        cm_df["texto"] = cm_df["tipo"] + ": " + cm_df["n"].map("{:,}".format) \
            + " (" + (cm_df["n"] / total).map("{:.1%}".format) + ")"

        base = alt.Chart(cm_df).encode(
            x=alt.X("Predicho:N", sort=labels), y=alt.Y("Real:N", sort=labels))
        heat = base.mark_rect().encode(color=alt.Color("n:Q", scale=alt.Scale(scheme="blues"), legend=None))
        txt = base.mark_text(fontSize=15).encode(
            text="texto:N",
            color=alt.condition(alt.datum.n > cm.max() / 2, alt.value("white"), alt.value("black")))

        c1, c2 = st.columns([1, 1])
        with c1:
            st.subheader("Matriz de confusión")
            st.altair_chart((heat + txt).properties(height=320), width="stretch")
        with c2:
            st.subheader("¿Qué significa cada error?")
            st.markdown(f"""
Clase positiva: **se cierra en menos de 24 h**.

- **Falso positivo ({fp:,})**: el modelo dijo "< 24 h" y tardó más.
  El ciudadano recibe una expectativa falsa y la solicitud **no se prioriza** aunque iba a tardar.
  Es el error más costoso para la operación.
- **Falso negativo ({fn:,})**: el modelo dijo "≥ 24 h" y se cerró rápido.
  Se asignan seguimiento o recursos **de más**, pero nadie sale perjudicado.
- **Aciertos**: {tp:,} cierres rápidos y {tn:,} lentos bien identificados
  (exactitud {(tp + tn) / total:.1%}).
""")
    else:
        st.info("Falta `confusion_matrix.csv`.")

# ---------------------------------------------------------------- 4. predicción
with tab_pred:
    if model is None:
        st.info("Falta `best_model.joblib`.")
    else:
        st.subheader("Simular una solicitud recién registrada")
        cats = encoder_categories(model)
        raw = load_raw_data()

        c1, c2, c3 = st.columns(3)
        fecha = c1.date_input("Fecha de registro", value=datetime(2025, 3, 20).date())
        hora = c2.time_input("Hora de registro", value=datetime(2025, 3, 20, 10, 0).time())
        created = datetime.combine(fecha, hora)
        c3.text_input("Día de la semana", WEEKDAYS_ES[WEEKDAYS[created.weekday()]], disabled=True)

        agency = c1.selectbox("agency", cats.get("agency", []))
        # Si hay datos locales, filtra los tipos de queja de esa agencia
        ct_options = cats.get("complaint_type", [])
        if raw is not None:
            seen = raw.loc[raw["agency"] == agency, "complaint_type"].value_counts().index
            ct_options = [c for c in seen if c in ct_options] or ct_options
        complaint = c2.selectbox("complaint_type", ct_options)
        borough = c3.selectbox("borough", cats.get("borough", []))
        channel = c1.selectbox("open_data_channel_type", cats.get("open_data_channel_type", []))
        loc_opts = ["(sin dato)"] + cats.get("location_type", [])
        location = c2.selectbox("location_type", loc_opts)

        row = {
            "hour": created.hour,
            "weekday": WEEKDAYS[created.weekday()],
            "agency": agency, "complaint_type": complaint, "borough": borough,
            "open_data_channel_type": channel,
            "location_type": np.nan if location == "(sin dato)" else location,
        }
        X = pd.DataFrame([row])[NUM_FEATURES + CAT_FEATURES]

        if st.button("Predecir", type="primary"):
            score, is_proba = score_positive(model, X)
            pred = int(model.predict(X)[0])
            r1, r2 = st.columns([1, 2])
            if is_proba:
                r1.metric("P(cierre < 24 h)", f"{score:.1%}")
                r1.progress(score)
            else:
                r1.metric("Puntaje de decisión", f"{score:+.2f}")
                r1.caption("LinearSVC no da probabilidades: > 0 significa clase 1.")
            if pred == 1:
                r2.success("### ✅ Se cerrará en **menos de 24 h**")
            else:
                r2.warning("### ⏳ Tardará **24 h o más**")

            if raw is not None:
                same = raw[(raw["agency"] == agency) & (raw["complaint_type"] == complaint)]
                if len(same):
                    r2.caption(f"Referencia histórica: {len(same):,} solicitudes de {agency} · {complaint}, "
                               f"{same[TARGET].mean():.1%} cerradas en < 24 h.")

# ---------------------------------------------------------------- 5. exploración
with tab_eda:
    raw = load_raw_data()
    if raw is None:
        st.info("No hay datos en caché (`data/`). Corre `python train.py` para descargarlos.")
    else:
        st.caption(f"{len(raw):,} solicitudes después de construir el target · "
                   f"{raw[TARGET].mean():.1%} con P_24h = 1")
        var = st.selectbox("Variable", CAT_FEATURES + NUM_FEATURES, index=0)
        g = (raw.groupby(var, dropna=False)[TARGET].agg(n="size", p24="mean").reset_index()
             .sort_values("n", ascending=False).head(25))
        g[var] = g[var].fillna("sin_dato").astype(str)
        sort = sorted(g[var], key=int) if var in NUM_FEATURES else list(g[var])
        bar = alt.Chart(g).mark_bar().encode(
            x=alt.X("p24:Q", title="% cerradas en < 24 h", axis=alt.Axis(format="%")),
            y=alt.Y(f"{var}:N", sort=sort, title=None),
            tooltip=[var, alt.Tooltip("n:Q", format=","), alt.Tooltip("p24:Q", format=".1%")],
        ).properties(height=max(200, 22 * len(g)))
        st.altair_chart(bar, width="stretch")

        daily = raw.set_index("created_date")[TARGET].resample("D").agg(["size", "mean"]).reset_index()
        st.subheader("Deriva temporal: % P_24h por día")
        st.altair_chart(
            alt.Chart(daily).mark_line().encode(
                x=alt.X("created_date:T", title=None),
                y=alt.Y("mean:Q", title="% P_24h", axis=alt.Axis(format="%"), scale=alt.Scale(zero=False)),
                tooltip=[alt.Tooltip("created_date:T"), alt.Tooltip("size:Q", title="n"),
                         alt.Tooltip("mean:Q", format=".1%")],
            ).properties(height=250),
            width="stretch",
        )
