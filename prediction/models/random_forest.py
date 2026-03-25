import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from features.config import RANDOM_SEED


class RandomForestModel:
    def __init__(self, n_estimators: int = 400, max_depth: int = 12, random_seed: int = RANDOM_SEED):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=2,
            class_weight=None,
            random_state=random_seed,
            n_jobs=-1,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomForestModel":
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)

    def get_feature_importances(self) -> np.ndarray:
        return self.model.feature_importances_

    def save(self, path: str) -> None:
        joblib.dump(self.model, path)

    @classmethod
    def load(cls, path: str) -> "RandomForestModel":
        instance = cls()
        instance.model = joblib.load(path)
        return instance
