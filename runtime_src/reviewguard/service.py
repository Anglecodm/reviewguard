from __future__ import annotations

import json
import math
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from .features import FEATURE_NAMES, vectorize_texts

try:
    import joblib
except ImportError:  # pragma: no cover - fallback for minimal runtime
    joblib = None

try:
    import tensorflow as tf
except ImportError:
    tf = None


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


@dataclass
class Prediction:
    label: str
    score: float
    decision: str
    reason: str | None = None


def _resolve_path(path: str | Path) -> Path:
    path = Path(path)
    if path.exists():
        return path
    base = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent.parent))
    candidate = base / path
    if candidate.exists():
        return candidate
    # Try only the filename under base/models for packaged apps
    candidate = base / "models" / path.name
    if candidate.exists():
        return candidate
    return path


class ReviewGuardService:
    def __init__(self, model_path: str | Path, metadata_path: str | Path):
        self.model_path = _resolve_path(model_path)
        self.metadata_path = _resolve_path(metadata_path)
        meta = json.loads(self.metadata_path.read_text(encoding="utf-8"))

        self.feature_names = meta.get("feature_names", FEATURE_NAMES)
        self.keywords = meta.get("keywords", [])
        self.model_type = meta.get("model_type", "sklearn")
        policy = meta.get("decision_policy", {})
        self.default_threshold = _clamp01(policy.get("fraud_threshold", 0.5))
        self.review_band = min(max(float(policy.get("review_band", 0.1)), 0.0), 0.49)
        self.short_review_words = max(int(policy.get("short_review_words", 10)), 1)

        if not self.model_path.exists() and meta.get("model_path"):
            self.model_path = _resolve_path(meta["model_path"])

        if self.model_type == "keras":
            if tf is None:
                raise ImportError("TensorFlow is required to load keras model.")
            self.model = tf.keras.models.load_model(self.model_path)
        else:
            if joblib is not None:
                self.model = joblib.load(self.model_path)
            else:
                with self.model_path.open("rb") as handle:
                    self.model = pickle.load(handle)

    def _score(self, X):
        if self.model_type == "keras":
            probs = self.model.predict(X, verbose=0).reshape(-1)
            return probs
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)[:, 1]
        if hasattr(self.model, "decision_function"):
            raw = self.model.decision_function(X)
            return np.array([_sigmoid(float(val)) for val in raw])
        return self.model.predict(X)

    def predict(self, texts: Iterable[str], threshold: float | None = None) -> list[Prediction]:
        texts = [text if text is not None else "" for text in texts]
        fraud_threshold = _clamp01(threshold if threshold is not None else self.default_threshold)
        X = vectorize_texts(texts, keywords=self.keywords)
        X = X[self.feature_names]
        scores = self._score(X)
        preds = []
        for idx, score in enumerate(scores):
            score = _clamp01(float(score))
            label = "fraud" if score >= fraud_threshold else "authentic"
            word_count = int(X.iloc[idx].get("word_count", 0))

            # Always return a hard verdict; keep auxiliary confidence signals as metadata.
            decision = label
            reason = None
            if word_count < self.short_review_words:
                reason = "short_review_signal"
            elif abs(score - fraud_threshold) <= self.review_band:
                reason = "low_confidence_signal"

            preds.append(Prediction(label=label, score=score, decision=decision, reason=reason))
        return preds
