from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from reviewguard.features import vectorize_texts


def parse_label(value) -> int | None:
    if value is None:
        return None
    token = str(value).strip().lower()
    if token in {"0", "authentic", "real", "genuine", "legit", "clean"}:
        return 0
    if token in {"1", "fraud", "fake", "spam", "manipulated", "suspicious"}:
        return 1
    return None


def load_model(model_path: Path):
    try:
        return joblib.load(model_path)
    except Exception:
        with model_path.open("rb") as handle:
            return pickle.load(handle)


def score_model(model, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(X)[:, 1], dtype=float)
    if hasattr(model, "decision_function"):
        raw = np.asarray(model.decision_function(X), dtype=float)
        return 1.0 / (1.0 + np.exp(-raw))
    return np.asarray(model.predict(X), dtype=float)


def safe_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Threshold tuning report with Recall@Precision target.")
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--model-path", default="models/reviewguard_model.joblib")
    parser.add_argument("--metadata-path", default="models/reviewguard_metadata.json")
    parser.add_argument("--text-col", default="text")
    parser.add_argument("--label-col", default="label")
    parser.add_argument("--target-precision", type=float, default=0.9)
    parser.add_argument("--threshold-min", type=float, default=0.05)
    parser.add_argument("--threshold-max", type=float, default=0.95)
    parser.add_argument("--threshold-step", type=float, default=0.01)
    parser.add_argument(
        "--review-band",
        type=float,
        default=-1.0,
        help="Set >=0 to evaluate triage band; -1 uses metadata value if present.",
    )
    parser.add_argument(
        "--short-review-words",
        type=int,
        default=-1,
        help="Set >=0 to evaluate triage short-review rule; -1 uses metadata value if present.",
    )
    parser.add_argument("--outdir", default="reports")
    parser.add_argument("--prefix", default="threshold_eval")
    args = parser.parse_args()

    csv_path = Path(args.input_csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {csv_path}")

    meta_path = Path(args.metadata_path)
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata JSON not found: {meta_path}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    model_path = Path(args.model_path)
    if not model_path.exists() and meta.get("model_path"):
        model_path = Path(meta["model_path"])
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    df = pd.read_csv(csv_path)
    if args.text_col not in df.columns or args.label_col not in df.columns:
        raise ValueError(f"CSV must include '{args.text_col}' and '{args.label_col}'.")

    df["text"] = df[args.text_col].fillna("").astype(str)
    df["label_parsed"] = df[args.label_col].map(parse_label)
    df = df[df["label_parsed"].isin([0, 1])].copy()
    if df.empty:
        raise ValueError("No valid binary labels found after parsing.")

    feature_names = meta.get("feature_names")
    keywords = meta.get("keywords", [])
    X = vectorize_texts(df["text"].tolist(), keywords=keywords)
    if feature_names:
        X = X[feature_names]
    y = df["label_parsed"].astype(int).to_numpy()
    scores = score_model(load_model(model_path), X)

    thresholds = np.arange(args.threshold_min, args.threshold_max + 1e-12, args.threshold_step)
    policy = meta.get("decision_policy", {})
    review_band = args.review_band if args.review_band >= 0 else float(policy.get("review_band", 0.1))
    short_words = args.short_review_words if args.short_review_words >= 0 else int(policy.get("short_review_words", 10))
    words = X["word_count"].to_numpy() if "word_count" in X.columns else np.zeros(len(X))

    rows = []
    for thr in thresholds:
        pred = (scores >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        base = safe_metrics(y, pred)

        uncertain = (np.abs(scores - thr) <= review_band) | (words < short_words)
        decided_mask = ~uncertain
        if decided_mask.any():
            y_dec = y[decided_mask]
            pred_dec = pred[decided_mask]
            triage_metrics = safe_metrics(y_dec, pred_dec)
            coverage = float(decided_mask.mean())
        else:
            triage_metrics = {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
            coverage = 0.0

        rows.append(
            {
                "threshold": float(round(float(thr), 5)),
                "accuracy": base["accuracy"],
                "precision": base["precision"],
                "recall": base["recall"],
                "f1": base["f1"],
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
                "coverage_auto_decision": coverage,
                "triage_precision": triage_metrics["precision"],
                "triage_recall": triage_metrics["recall"],
                "triage_f1": triage_metrics["f1"],
            }
        )

    report_df = pd.DataFrame(rows)
    feasible = report_df[report_df["precision"] >= args.target_precision].copy()

    if not feasible.empty:
        feasible = feasible.sort_values(["recall", "precision", "f1"], ascending=[False, False, False])
        best = feasible.iloc[0]
        meets_target = True
    else:
        report_sorted = report_df.sort_values(["precision", "recall", "f1"], ascending=[False, False, False])
        best = report_sorted.iloc[0]
        meets_target = False

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    csv_out = outdir / f"{args.prefix}_table.csv"
    json_out = outdir / f"{args.prefix}_summary.json"
    report_df.to_csv(csv_out, index=False)

    summary = {
        "input_csv": str(csv_path),
        "rows_evaluated": int(len(df)),
        "positive_rate": float((y == 1).mean()),
        "target_precision": float(args.target_precision),
        "target_met": bool(meets_target),
        "recommended_threshold": float(best["threshold"]),
        "recommended_metrics": {
            "accuracy": float(best["accuracy"]),
            "precision": float(best["precision"]),
            "recall": float(best["recall"]),
            "f1": float(best["f1"]),
            "coverage_auto_decision": float(best["coverage_auto_decision"]),
            "triage_precision": float(best["triage_precision"]),
            "triage_recall": float(best["triage_recall"]),
            "triage_f1": float(best["triage_f1"]),
        },
        "triage_policy_used": {
            "review_band": float(review_band),
            "short_review_words": int(short_words),
        },
        "outputs": {
            "table_csv": str(csv_out),
            "summary_json": str(json_out),
        },
    }
    json_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Wrote: {csv_out}")
    print(f"Wrote: {json_out}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
