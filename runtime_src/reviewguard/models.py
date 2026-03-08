from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


def build_models(random_state: int = 42):
    return {
        "naive_bayes": GaussianNB(),
        "random_forest": RandomForestClassifier(
            n_estimators=150,
            max_depth=10,
            min_samples_split=4,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=random_state,
        ),
        "svm": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", LinearSVC()),
            ]
        ),
        "mlp": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    MLPClassifier(
                        hidden_layer_sizes=(64, 32),
                        activation="relu",
                        alpha=0.0001,
                        max_iter=300,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
    }


@dataclass
class EvaluationResult:
    name: str
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    std_accuracy: float


def evaluate_models(X: pd.DataFrame, y: pd.Series, models=None, cv_splits: int = 5):
    if models is None:
        models = build_models()

    scoring = {
        "accuracy": "accuracy",
        "precision": make_scorer(precision_score, zero_division=0),
        "recall": make_scorer(recall_score, zero_division=0),
        "f1": make_scorer(f1_score, zero_division=0),
        "roc_auc": "roc_auc",
    }

    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)
    results = []
    for name, model in models.items():
        try:
            scores = cross_validate(model, X, y, cv=cv, scoring=scoring, n_jobs=None)
            results.append(
                EvaluationResult(
                    name=name,
                    accuracy=float(np.mean(scores["test_accuracy"])),
                    precision=float(np.mean(scores["test_precision"])),
                    recall=float(np.mean(scores["test_recall"])),
                    f1=float(np.mean(scores["test_f1"])),
                    roc_auc=float(np.mean(scores["test_roc_auc"])),
                    std_accuracy=float(np.std(scores["test_accuracy"])),
                )
            )
        except Exception:
            results.append(
                EvaluationResult(
                    name=name,
                    accuracy=float("nan"),
                    precision=float("nan"),
                    recall=float("nan"),
                    f1=float("nan"),
                    roc_auc=float("nan"),
                    std_accuracy=float("nan"),
                )
            )
    return results


def feature_importance_map(model, feature_names: Iterable[str]) -> dict[str, float]:
    feature_names = list(feature_names)

    if hasattr(model, "feature_importances_"):
        values = np.asarray(model.feature_importances_, dtype=float)
    else:
        return {}

    if values.shape[0] != len(feature_names):
        return {}

    pairs = sorted(zip(feature_names, values.tolist()), key=lambda item: item[1], reverse=True)
    return {name: float(score) for name, score in pairs}
