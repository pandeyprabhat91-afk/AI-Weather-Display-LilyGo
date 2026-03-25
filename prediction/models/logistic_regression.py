import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from features.config import RANDOM_SEED


class LogisticRegressionModel:
    def __init__(self, C: float = 1.0, random_seed: int = RANDOM_SEED):
        self.model = LogisticRegression(
            solver="lbfgs",
            max_iter=1000,
            C=C,
            class_weight="balanced",
            random_state=random_seed,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRegressionModel":
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)

    def get_feature_importances(self) -> np.ndarray:
        """Returns coefficient matrix shape (n_classes, n_features)."""
        return self.model.coef_

    def save(self, path: str) -> None:
        joblib.dump(self.model, path)

    @classmethod
    def load(cls, path: str) -> "LogisticRegressionModel":
        instance = cls()
        instance.model = joblib.load(path)
        return instance
