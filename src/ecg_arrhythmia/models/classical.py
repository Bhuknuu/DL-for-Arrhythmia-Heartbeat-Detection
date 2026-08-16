"""Classical machine learning models for ECG arrhythmia classification."""

from typing import Any, Dict, Optional
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


class ClassicalModelWrapper:
    """Base wrapper ensuring standard API across scikit-learn models."""

    def __init__(self, estimator: Any, name: str):
        self.estimator = estimator
        self.name = name

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ClassicalModelWrapper":
        # Flatten if 3D tensor (N, 1, L) -> (N, L)
        if X.ndim == 3:
            X = X.reshape(X.shape[0], -1)
        self.estimator.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if X.ndim == 3:
            X = X.reshape(X.shape[0], -1)
        return self.estimator.predict(X)

    def predict_proba(self, X: np.ndarray, num_classes: int = 5) -> np.ndarray:
        if X.ndim == 3:
            X = X.reshape(X.shape[0], -1)
        
        full_probs = np.zeros((len(X), num_classes), dtype=np.float32)
        
        if hasattr(self.estimator, "predict_proba"):
            raw_probs = self.estimator.predict_proba(X)
            classes = getattr(self.estimator, "classes_", np.arange(raw_probs.shape[1]))
            for idx, cls_idx in enumerate(classes):
                if cls_idx < num_classes:
                    full_probs[:, cls_idx] = raw_probs[:, idx]
            return full_probs
        elif hasattr(self.estimator, "decision_function"):
            df = self.estimator.decision_function(X)
            exp_df = np.exp(df - np.max(df, axis=1, keepdims=True))
            raw_probs = exp_df / np.sum(exp_df, axis=1, keepdims=True)
            classes = getattr(self.estimator, "classes_", np.arange(raw_probs.shape[1]))
            for idx, cls_idx in enumerate(classes):
                if cls_idx < num_classes:
                    full_probs[:, cls_idx] = raw_probs[:, idx]
            return full_probs
        else:
            preds = self.predict(X)
            for i, p in enumerate(preds):
                if p < num_classes:
                    full_probs[i, p] = 1.0
            return full_probs


def build_logistic_regression(
    C: float = 1.0, max_iter: int = 1000, random_state: int = 42, **kwargs
) -> ClassicalModelWrapper:
    clf = LogisticRegression(
        C=C, max_iter=max_iter, random_state=random_state, solver="lbfgs"
    )
    return ClassicalModelWrapper(clf, "logistic_regression")


def build_random_forest(
    n_estimators: int = 100, max_depth: Optional[int] = 15, random_state: int = 42, **kwargs
) -> ClassicalModelWrapper:
    clf = RandomForestClassifier(
        n_estimators=n_estimators, max_depth=max_depth, random_state=random_state, n_jobs=-1
    )
    return ClassicalModelWrapper(clf, "random_forest")


def build_gradient_boosting(
    n_estimators: int = 100, learning_rate: float = 0.1, max_depth: int = 5, random_state: int = 42, **kwargs
) -> ClassicalModelWrapper:
    clf = GradientBoostingClassifier(
        n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth, random_state=random_state
    )
    return ClassicalModelWrapper(clf, "gradient_boosting")


def build_svm(
    C: float = 1.0, kernel: str = "rbf", probability: bool = True, random_state: int = 42, **kwargs
) -> ClassicalModelWrapper:
    clf = SVC(C=C, kernel=kernel, probability=probability, random_state=random_state)
    return ClassicalModelWrapper(clf, "svm")
