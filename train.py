# PARA EL USUARIO

from src.io import Dataset
from src.evaluation import ModelEvaluation

# Modelos
from src.preprocessor import build_preprocessor
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor


def train():

    # Importacion y preprocesamiento datos
    data = Dataset(seed=43, num_samples=None)
    X, y = data.load_xy()

    # Model pipeline (preprocessing + model, Linear Regression)
    pipeline_lr = Pipeline(
        [
            ("preprocessor", build_preprocessor()),
            ("model", LogisticRegression(max_iter=1000,
                                         tol=0.0001,
                                         solver='lbfgs'))
        ]
    )

    # Entrenamiento y métricas
    ev = ModelEvaluation(X=X, y=y)
    ev.evaluate_model(model=pipeline_lr)

#    return

if __name__ == "__main__":
    train()