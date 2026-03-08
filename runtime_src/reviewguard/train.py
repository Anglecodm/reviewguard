from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

from .data import load_reviews
from .features import FEATURE_NAMES, vectorize_texts
from .keras_model import evaluate_keras, train_keras
from .keywords import load_keywords
from .models import build_models, evaluate_models, feature_importance_map

try:
    import joblib
except ImportError:  # pragma: no cover - fallback for minimal runtime
    joblib = None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _label_counts(series) -> dict[str, int]:
    counts = series.value_counts(dropna=False).sort_index()
    return {str(key): int(value) for key, value in counts.items()}


def _score_model(model, X) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(X)[:, 1], dtype=float)
    if hasattr(model, "decision_function"):
        raw = np.asarray(model.decision_function(X), dtype=float)
        return 1.0 / (1.0 + np.exp(-raw))
    return np.asarray(model.predict(X), dtype=float)


def _build_holdout_metrics(model, X, y, threshold: float) -> dict[str, float | None]:
    scores = _score_model(model, X)
    preds = (scores >= threshold).astype(int)
    y_arr = np.asarray(y, dtype=int)
    payload = {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_arr, preds)),
        "precision": float(precision_score(y_arr, preds, zero_division=0)),
        "recall": float(recall_score(y_arr, preds, zero_division=0)),
        "f1": float(f1_score(y_arr, preds, zero_division=0)),
    }
    if len(np.unique(y_arr)) > 1:
        payload["roc_auc"] = float(roc_auc_score(y_arr, scores))
    else:
        payload["roc_auc"] = None
    return payload


def _save_sklearn_model(model, output_path: Path) -> None:
    if joblib is not None:
        joblib.dump(model, output_path)
        return
    with output_path.open("wb") as handle:
        pickle.dump(model, handle)


def main():
    parser = argparse.ArgumentParser(description="Train Review Guard models.")
    parser.add_argument("--data", default="data/sample_reviews.csv", help="CSV with text and label columns.")
    parser.add_argument("--text-col", default="text", help="Column name for review text.")
    parser.add_argument("--label-col", default="label", help="Column name for labels (0=authentic, 1=fraud).")
    parser.add_argument("--keywords", default=None, help="Path to promo keywords file.")
    parser.add_argument(
        "--model",
        default="random_forest",
        choices=["naive_bayes", "random_forest", "svm", "mlp", "keras_mlp"],
        help="Model to train for deployment.",
    )
    parser.add_argument("--evaluate", action="store_true", help="Run cross-validation evaluation.")
    parser.add_argument("--outdir", default="models", help="Output directory for trained model.")
    parser.add_argument("--epochs", type=int, default=100, help="Epochs for keras training.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for keras training.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split fraction.")
    parser.add_argument("--val-size", type=float, default=0.1, help="Validation split fraction.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--min-rows-warning",
        type=int,
        default=300,
        help="Warn when dataset has fewer rows than this recommendation.",
    )
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="Calibrate sklearn model probabilities using validation split.",
    )
    parser.add_argument(
        "--calibration-method",
        default="sigmoid",
        choices=["sigmoid", "isotonic"],
        help="Calibration method for sklearn models.",
    )
    parser.add_argument(
        "--fraud-threshold-default",
        type=float,
        default=0.65,
        help="Default fraud threshold stored in metadata for serving.",
    )
    parser.add_argument(
        "--review-band",
        type=float,
        default=0.1,
        help="Low-confidence margin around threshold sent to manual review.",
    )
    parser.add_argument(
        "--short-review-words",
        type=int,
        default=10,
        help="Reviews shorter than this are marked for manual review.",
    )
    args = parser.parse_args()

    data_path = Path(args.data).resolve()
    df = load_reviews(data_path, text_col=args.text_col, label_col=args.label_col)
    keywords = load_keywords(args.keywords)
    warnings: list[str] = []

    if len(df) < max(4, args.min_rows_warning):
        warnings.append(
            f"Dataset has {len(df)} rows. Recommended minimum for production is {args.min_rows_warning} rows."
        )

    if not 0 < args.test_size < 1:
        raise ValueError("--test-size must be between 0 and 1.")
    if not 0 <= args.val_size < 1:
        raise ValueError("--val-size must be between 0 and 1.")
    if args.test_size + args.val_size >= 1:
        raise ValueError("--test-size + --val-size must be less than 1.")

    try:
        train_df, test_df = train_test_split(
            df,
            test_size=args.test_size,
            random_state=args.random_state,
            stratify=df["label"],
        )
    except ValueError:
        train_df, test_df = train_test_split(
            df,
            test_size=args.test_size,
            random_state=args.random_state,
            stratify=None,
        )
        warnings.append("Fell back to non-stratified test split due to dataset constraints.")

    val_df = train_df.iloc[0:0].copy()
    if args.val_size > 0:
        relative_val_size = args.val_size / (1.0 - args.test_size)
        if 0 < relative_val_size < 1:
            try:
                train_df, val_df = train_test_split(
                    train_df,
                    test_size=relative_val_size,
                    random_state=args.random_state,
                    stratify=train_df["label"],
                )
            except ValueError:
                warnings.append("Validation split unavailable with stratification; skipping calibration set.")

    X_train = vectorize_texts(train_df["text"].tolist(), keywords=keywords)
    y_train = train_df["label"]
    X_test = vectorize_texts(test_df["text"].tolist(), keywords=keywords)
    y_test = test_df["label"]
    X_val = vectorize_texts(val_df["text"].tolist(), keywords=keywords) if not val_df.empty else None
    y_val = val_df["label"] if not val_df.empty else None

    models = build_models(random_state=args.random_state)
    metrics = []
    if args.evaluate and args.model != "keras_mlp":
        class_floor = int(y_train.value_counts().min()) if not y_train.empty else 0
        cv_splits = min(5, class_floor)
        if cv_splits >= 2:
            metrics = evaluate_models(X_train, y_train, models=models, cv_splits=cv_splits)
        else:
            warnings.append("Cross-validation skipped: not enough class samples in training split.")
    elif args.evaluate and args.model == "keras_mlp":
        metrics = [
            evaluate_keras(
                X_train.to_numpy(dtype=np.float32),
                y_train.to_numpy(dtype=np.int32),
                epochs=args.epochs,
                batch_size=args.batch_size,
                random_state=args.random_state,
            )
        ]

    calibrated = False
    importance_payload: dict[str, float] = {}

    if args.model == "keras_mlp":
        model = train_keras(
            X_train.to_numpy(dtype=np.float32),
            y_train.to_numpy(dtype=np.int32),
            epochs=args.epochs,
            batch_size=args.batch_size,
        )
        model_type = "keras"
    else:
        model = models[args.model]
        model.fit(X_train, y_train)
        if args.model == "random_forest":
            importance_payload = feature_importance_map(model, FEATURE_NAMES)

        can_calibrate = hasattr(model, "predict_proba") or hasattr(model, "decision_function")
        if args.calibrate:
            if not can_calibrate:
                warnings.append("Calibration skipped: selected model does not expose score probabilities.")
            elif X_val is None or y_val is None or len(y_val) < 2:
                warnings.append("Calibration skipped: validation split unavailable.")
            else:
                calibrator = CalibratedClassifierCV(model, method=args.calibration_method, cv="prefit")
                calibrator.fit(X_val, y_val)
                model = calibrator
                calibrated = True

        model_type = "sklearn"

    holdout_metrics = (
        _build_holdout_metrics(model, X_test, y_test, threshold=args.fraud_threshold_default)
        if model_type == "sklearn"
        else {}
    )

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if model_type == "keras":
        model_path = outdir / "reviewguard_model.keras"
        model.save(model_path)
    else:
        model_path = outdir / "reviewguard_model.joblib"
        _save_sklearn_model(model, model_path)

    metadata = {
        "model_name": args.model,
        "model_type": model_type,
        "model_path": str(model_path),
        "feature_names": FEATURE_NAMES,
        "keywords": keywords,
        "calibrated": calibrated,
        "decision_policy": {
            "fraud_threshold": float(args.fraud_threshold_default),
            "review_band": float(args.review_band),
            "short_review_words": int(args.short_review_words),
        },
        "dataset": {
            "path": str(data_path),
            "sha256": _sha256_file(data_path),
            "rows": int(len(df)),
            "label_counts": _label_counts(df["label"]),
            "text_col": args.text_col,
            "label_col": args.label_col,
        },
        "split": {
            "train_rows": int(len(train_df)),
            "val_rows": int(len(val_df)),
            "test_rows": int(len(test_df)),
            "train_label_counts": _label_counts(train_df["label"]),
            "val_label_counts": _label_counts(val_df["label"]) if not val_df.empty else {},
            "test_label_counts": _label_counts(test_df["label"]),
        },
        "training_warnings": warnings,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "holdout_metrics": holdout_metrics,
    }
    metadata_path = outdir / "reviewguard_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    dataset_manifest = {
        "dataset_path": str(data_path),
        "sha256": metadata["dataset"]["sha256"],
        "rows": metadata["dataset"]["rows"],
        "label_counts": metadata["dataset"]["label_counts"],
        "text_col": args.text_col,
        "label_col": args.label_col,
        "generated_at_utc": metadata["trained_at_utc"],
    }
    (outdir / "dataset_manifest.json").write_text(json.dumps(dataset_manifest, indent=2), encoding="utf-8")
    (outdir / "data_split_summary.json").write_text(json.dumps(metadata["split"], indent=2), encoding="utf-8")
    if holdout_metrics:
        (outdir / "holdout_metrics.json").write_text(json.dumps(holdout_metrics, indent=2), encoding="utf-8")
    if importance_payload:
        (outdir / "feature_importance.json").write_text(json.dumps(importance_payload, indent=2), encoding="utf-8")

    if metrics:
        metrics_path = outdir / "evaluation_metrics.json"
        if args.model == "keras_mlp":
            r = metrics[0]
            metrics_payload = [
                {
                    "model": "keras_mlp",
                    "accuracy": r.accuracy,
                    "precision": r.precision,
                    "recall": r.recall,
                    "f1": r.f1,
                    "roc_auc": r.roc_auc,
                    "std_accuracy": r.std_accuracy,
                }
            ]
        else:
            metrics_payload = [
                {
                    "model": r.name,
                    "accuracy": r.accuracy,
                    "precision": r.precision,
                    "recall": r.recall,
                    "f1": r.f1,
                    "roc_auc": r.roc_auc,
                    "std_accuracy": r.std_accuracy,
                }
                for r in metrics
            ]
        metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
