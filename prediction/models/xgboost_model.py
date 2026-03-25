import numpy as np
import joblib
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from features.config import RANDOM_SEED, N_CLASSES


class XGBoostModel:
    """XGBoost classifier with internal label encoding.

    Handles non-consecutive class labels gracefully (e.g. when Stormy class
    is absent from data, labels are [0,1,2,4] rather than [0,1,2,3,4]).
    predict_proba always returns N_CLASSES columns, zero-padded for absent classes.
    """

    def __init__(self, random_seed: int = RANDOM_SEED):
        self.random_seed = random_seed
        self._le = LabelEncoder()
        self._classes = None  # original class labels seen during fit
        self.model = XGBClassifier(
            n_estimators=600,
            max_depth=8,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            gamma=0.1,
            reg_alpha=0.1,
            reg_lambda=1.0,
            eval_metric="mlogloss",
            random_state=random_seed,
            n_jobs=-1,
            tree_method="hist",
        )

    def fit(self, X, y, sample_weight=None):
        y_enc = self._le.fit_transform(y)
        self._classes = self._le.classes_  # e.g. [0, 1, 2, 4]
        self.model.fit(X, y_enc, sample_weight=sample_weight, verbose=False)
        return self

    def predict(self, X) -> np.ndarray:
        y_enc = self.model.predict(X)
        return self._le.inverse_transform(y_enc).astype(np.int32)

    def predict_proba(self, X) -> np.ndarray:
        raw = self.model.predict_proba(X).astype(np.float32)
        # Expand to N_CLASSES columns (missing classes stay at 0 probability)
        out = np.zeros((raw.shape[0], N_CLASSES), dtype=np.float32)
        for enc_idx, orig_class in enumerate(self._classes):
            out[:, int(orig_class)] = raw[:, enc_idx]
        return out

    def get_feature_importances(self) -> np.ndarray:
        return self.model.feature_importances_

    def save(self, path: str) -> None:
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str) -> "XGBoostModel":
        return joblib.load(path)
