from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier


# Prefer the existing artifact in the repo
MODEL_PATHS = [Path("knn_model.sav"), Path("modelKNN1.pkl")]


def _train_and_save_model(model_path: Path) -> KNeighborsClassifier:
    """Train a simple KNN on the Iris dataset and persist it."""
    iris = load_iris(as_frame=True)
    X = iris.data[["sepal length (cm)", "sepal width (cm)", "petal length (cm)", "petal width (cm)"]]
    y = iris.target_names[iris.target]

    clf = KNeighborsClassifier()
    clf.fit(X, y)

    joblib.dump(clf, model_path)
    return clf


@lru_cache(maxsize=1)
def _load_model():
    for path in MODEL_PATHS:
        if path.exists():
            return joblib.load(path)
    # If nothing is found, retrain and save to the first path
    return _train_and_save_model(MODEL_PATHS[0])


def predict(data: np.ndarray):
    """Predict iris species; loads existing model file or trains a fallback."""
    clf = _load_model()
    return clf.predict(data)
