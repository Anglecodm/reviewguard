from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

try:
    import tensorflow as tf
except ImportError as exc:  # pragma: no cover - optional dependency
    tf = None
    _TF_IMPORT_ERROR = exc
else:
    _TF_IMPORT_ERROR = None


@dataclass
class KerasEvaluation:
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    std_accuracy: float


def _require_tf():
    if tf is None:
        raise ImportError("TensorFlow is required for keras_mlp model.") from _TF_IMPORT_ERROR


def build_keras_model(input_dim: int):
    _require_tf()
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(input_dim,)),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def evaluate_keras(
    X: np.ndarray,
    y: np.ndarray,
    epochs: int = 100,
    batch_size: int = 32,
    random_state: int = 42,
):
    _require_tf()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    accs = []
    precisions = []
    recalls = []
    f1s = []
    aucs = []

    for train_idx, test_idx in cv.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model = build_keras_model(input_dim=X.shape[1])
        callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
        ]
        model.fit(
            X_train,
            y_train,
            validation_data=(X_test, y_test),
            epochs=epochs,
            batch_size=batch_size,
            verbose=0,
            callbacks=callbacks,
        )
        probs = model.predict(X_test, verbose=0).reshape(-1)
        preds = (probs >= 0.5).astype(int)

        accs.append(accuracy_score(y_test, preds))
        precisions.append(precision_score(y_test, preds, zero_division=0))
        recalls.append(recall_score(y_test, preds, zero_division=0))
        f1s.append(f1_score(y_test, preds, zero_division=0))
        aucs.append(roc_auc_score(y_test, probs))

    return KerasEvaluation(
        accuracy=float(np.mean(accs)),
        precision=float(np.mean(precisions)),
        recall=float(np.mean(recalls)),
        f1=float(np.mean(f1s)),
        roc_auc=float(np.mean(aucs)),
        std_accuracy=float(np.std(accs)),
    )


def train_keras(
    X: np.ndarray,
    y: np.ndarray,
    epochs: int = 100,
    batch_size: int = 32,
):
    _require_tf()
    model = build_keras_model(input_dim=X.shape[1])
    callbacks = [tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)]
    model.fit(
        X,
        y,
        validation_split=0.2,
        epochs=epochs,
        batch_size=batch_size,
        verbose=0,
        callbacks=callbacks,
    )
    return model
