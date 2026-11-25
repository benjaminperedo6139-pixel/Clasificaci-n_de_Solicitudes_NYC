import pandas as pd
import numpy as np

# Scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, roc_auc_score

# modules
from src.config import SEED, TEST_SIZE

class ModelEvaluation:
    """
    Argumentos:
        X - Matriz de características (variables, columnas)
        y - Vector target
        test_size - Tamaño de datos de validacion
        shuffle - mezclar datos antes de split
        seed - semilla aleatoria
    """

    def __init__(self, X: pd.DataFrame, y: pd.Series, test_size: float = TEST_SIZE, shuffle: bool = True, seed:int = SEED):

        self.X_train, self.X_valid, self.y_train, self.y_valid = train_test_split(
            X,
            y,
            test_size=test_size,
            shuffle=shuffle,
            random_state=seed
        )

    def evaluate_model(self, model):
        """
        Argumentos:
            model - Pipeline (sklearn)
        """

        model_type = type(model.named_steps["model"]).__name__
        print(f"Model Type: {model_type}")

        # Entrenamiento (con datos de entrenamiento, X_train, y_train)
        model.fit(self.X_train, self.y_train)

        # Predicciones (con datos de validacion, X_valid)
        y_pred = model.predict(self.X_valid)

        # Evaluacion (métricas)
        accuracy = accuracy_score(y_pred=y_pred, y_true=self.y_valid)
        precision = precision_score(y_pred=y_pred, y_true=self.y_valid)
        recall = recall_score(y_pred=y_pred, y_true=self.y_valid)
        f1 = f1_score(y_pred=y_pred, y_true=self.y_valid)

        print(f"Accuracy: {accuracy:.2f}")
        print(f"Precision: {precision:.2f}")
        print(f"Recall: {recall:.2f}")
        print(f"F1: {f1:.2f}")

        