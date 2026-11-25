# PARA EL USUARIO
# Ejecutar:  python train.py

from src.io import Dataset
from src.config import DATE_COL, NUM_FEATURES, CAT_FEATURES, TARGET, RESULTS_DIR, FOLDS_FILE
from src.temporal_split import make_temporal_folds, get_final_train_idx
from src.models import get_models
from src.evaluation import run_experiment, select_best, evaluate_on_test


def train():

    # 1. Datos + target P_24h (Integrante A)
    data = Dataset()
    df = data.load_df_target().sort_values(DATE_COL).reset_index(drop=True)

    # 2. Cortes temporales + bloque de prueba (Integrante A)
    folds, test_idx, folds_info = make_temporal_folds(df)
    print(folds_info.to_string(index=False))
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    folds_info.to_csv(FOLDS_FILE, index=False)

    features = NUM_FEATURES + CAT_FEATURES
    # Desarrollo: todo lo anterior al bloque de prueba (los folds indexan sobre este prefijo)
    df_dev = df.drop(index=test_idx).reset_index(drop=True)
    X_dev, y_dev = df_dev[features], df_dev[TARGET]
    # Reentrenamiento final: solo solicitudes con etiqueta conocida al inicio de la prueba (gap 24 h)
    final_idx = get_final_train_idx(df)
    X_final, y_final = df.loc[final_idx, features], df.loc[final_idx, TARGET]
    # Prueba final
    df_test = df.loc[test_idx].reset_index(drop=True)
    X_test, y_test = df_test[features], df_test[TARGET]

    # 3. Modelos e hiperparámetros (Integrante B)
    models = get_models()

    # 4. Comparación por corte y selección (Integrante C)
    results = run_experiment(X_dev, y_dev, folds, models)
    best_name, best_params = select_best(results)
    print(f"Mejor modelo: {best_name} {best_params}")

    # 5. Evaluación final en el bloque reservado (Integrante C)
    best_model = models[best_name][0].set_params(**best_params)
    test_metrics = evaluate_on_test(best_model, X_final, y_final, X_test, y_test)
    print(test_metrics)


if __name__ == "__main__":
    train()
