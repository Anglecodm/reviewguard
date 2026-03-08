from __future__ import annotations

import argparse
import json
import pickle
import re
from pathlib import Path
import sys

import numpy as np
import pandas as pd

try:
    import joblib
except ImportError:  # pragma: no cover - optional fallback
    joblib = None

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


def normalize_text(text: str) -> str:
    text = (text or "").lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_model(model_path: Path):
    if joblib is not None:
        try:
            return joblib.load(model_path)
        except Exception:
            pass
    with model_path.open("rb") as handle:
        return pickle.load(handle)


def score_model(model, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(X)[:, 1], dtype=float)
    if hasattr(model, "decision_function"):
        raw = np.asarray(model.decision_function(X), dtype=float)
        return 1.0 / (1.0 + np.exp(-raw))
    return np.asarray(model.predict(X), dtype=float)


def score_to_decision(score: float, threshold: float, review_band: float, short_words: int, word_count: int) -> str:
    if word_count < short_words:
        return "manual_review"
    if abs(score - threshold) <= review_band:
        return "manual_review"
    if score >= threshold:
        return "fraud"
    return "authentic"


def row_signals(row: pd.Series, threshold: float) -> tuple[list[str], float, float, float]:
    signals: list[str] = []
    if row["duplicate_text_count"] >= 2:
        signals.append("duplicate_text")
    if row.get("promo_keyword_count", 0.0) >= 2:
        signals.append("promo_keywords")
    if row.get("caps_ratio", 0.0) >= 0.2:
        signals.append("caps_heavy")
    if row.get("exclamation_count", 0.0) >= 2:
        signals.append("multi_exclaim")
    if row.get("word_count", 0.0) <= 4:
        signals.append("very_short")
    if row["model_score"] >= max(0.7, threshold):
        signals.append("model_high_fraud")
    if abs(row["model_score"] - threshold) <= 0.08:
        signals.append("model_uncertain")

    signal_score = min(len(signals) / 7.0, 1.0)
    duplicate_score = min(max(float(row["duplicate_text_count"]) - 1.0, 0.0) / 5.0, 1.0)
    uncertainty_score = 1.0 - min(abs(float(row["model_score"]) - threshold) / 0.5, 1.0)
    return signals, signal_score, duplicate_score, uncertainty_score


def select_with_platform_diversity(df: pd.DataFrame, batch_size: int, seed: int, platform_col: str) -> pd.DataFrame:
    if len(df) <= batch_size:
        return df.sort_values("priority_score", ascending=False).copy()
    if platform_col not in df.columns:
        return df.sort_values("priority_score", ascending=False).head(batch_size).copy()

    pools = {}
    for platform, part in df.groupby(platform_col):
        pools[str(platform)] = part.sort_values("priority_score", ascending=False).copy()
    total = len(df)

    quotas: dict[str, int] = {}
    for platform, part in pools.items():
        quota = int(round(batch_size * (len(part) / total)))
        quotas[platform] = min(len(part), max(1, quota))

    selected_parts = []
    for platform, quota in quotas.items():
        selected_parts.append(pools[platform].head(quota))

    selected = pd.concat(selected_parts, axis=0).drop_duplicates(subset=["record_id"], keep="first")
    if len(selected) < batch_size:
        remaining = df[~df["record_id"].isin(selected["record_id"])].copy()
        need = batch_size - len(selected)
        selected = pd.concat([selected, remaining.sort_values("priority_score", ascending=False).head(need)], axis=0)
    elif len(selected) > batch_size:
        selected = selected.sort_values("priority_score", ascending=False).head(batch_size)

    return selected.sort_values("priority_score", ascending=False).copy()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build high-priority annotation batch from marketplace reviews.")
    parser.add_argument("--input-csv", default="data/marketplace_reviews_weak_labeled.csv")
    parser.add_argument("--output-csv", default="data/annotation_tasks_priority.csv")
    parser.add_argument("--output-annotator-a-csv", default="data/annotation_tasks_priority_annotator_a.csv")
    parser.add_argument("--output-annotator-b-csv", default="data/annotation_tasks_priority_annotator_b.csv")
    parser.add_argument("--output-summary-json", default="data/annotation_tasks_priority_summary.json")
    parser.add_argument("--batch-size", type=int, default=400)
    parser.add_argument("--text-col", default="text")
    parser.add_argument("--label-col", default="label")
    parser.add_argument("--platform-col", default="platform")
    parser.add_argument("--source-url-col", default="source_url")
    parser.add_argument("--user-col", default="user")
    parser.add_argument("--date-col", default="date")
    parser.add_argument("--rating-col", default="rating")
    parser.add_argument("--exclude-platform", default="synthetic")
    parser.add_argument("--model-path", default="models/reviewguard_model.joblib")
    parser.add_argument("--metadata-path", default="models/reviewguard_metadata.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-text-len", type=int, default=15)
    args = parser.parse_args()

    input_path = Path(args.input_csv)
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    df = pd.read_csv(input_path).copy()
    if args.text_col not in df.columns:
        raise ValueError(f"Missing text column '{args.text_col}' in input CSV.")

    df["text"] = df[args.text_col].fillna("").astype(str)
    df = df[df["text"].str.len() >= args.min_text_len].copy()

    if args.platform_col in df.columns and args.exclude_platform:
        tokens = {tok.strip().lower() for tok in args.exclude_platform.split(",") if tok.strip()}
        if tokens:
            platform_norm = df[args.platform_col].fillna("").astype(str).str.lower()
            df = df[~platform_norm.isin(tokens)].copy()

    if df.empty:
        raise ValueError("No candidate rows remain after filtering.")

    df["text_norm"] = df["text"].map(normalize_text)
    df["duplicate_text_count"] = df["text_norm"].map(df["text_norm"].value_counts())

    if "record_id" in df.columns:
        df["record_id"] = df["record_id"].astype(str)
    else:
        df["record_id"] = [f"p{i:07d}" for i in range(1, len(df) + 1)]

    metadata_path = Path(args.metadata_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    policy = metadata.get("decision_policy", {})
    fraud_threshold = float(policy.get("fraud_threshold", 0.65))
    review_band = float(policy.get("review_band", 0.1))
    short_words = int(policy.get("short_review_words", 10))
    feature_names = metadata.get("feature_names")
    keywords = metadata.get("keywords", [])

    X = vectorize_texts(df["text"].tolist(), keywords=keywords)
    if feature_names:
        X = X[feature_names]

    model_scores = np.zeros(len(df), dtype=float)
    model_path = Path(args.model_path)
    if model_path.exists():
        model = load_model(model_path)
        model_scores = score_model(model, X)

    df["model_score"] = model_scores

    feature_cols = [
        "word_count",
        "promo_keyword_count",
        "caps_ratio",
        "exclamation_count",
    ]
    for col in feature_cols:
        df[col] = X[col].to_numpy() if col in X.columns else 0.0

    signals_all: list[str] = []
    signal_scores: list[float] = []
    duplicate_scores: list[float] = []
    uncertainty_scores: list[float] = []
    priority_scores: list[float] = []
    model_decisions: list[str] = []

    for _, row in df.iterrows():
        signals, signal_score, duplicate_score, uncertainty_score = row_signals(row, fraud_threshold)
        priority = (
            (0.35 * uncertainty_score)
            + (0.30 * signal_score)
            + (0.20 * duplicate_score)
            + (0.15 * float(row["model_score"]))
        )
        signals_all.append("|".join(signals))
        signal_scores.append(float(signal_score))
        duplicate_scores.append(float(duplicate_score))
        uncertainty_scores.append(float(uncertainty_score))
        priority_scores.append(float(priority))
        model_decisions.append(
            score_to_decision(
                score=float(row["model_score"]),
                threshold=fraud_threshold,
                review_band=review_band,
                short_words=short_words,
                word_count=int(row.get("word_count", 0)),
            )
        )

    df["risk_signals"] = signals_all
    df["signal_score"] = signal_scores
    df["duplicate_score"] = duplicate_scores
    df["uncertainty_score"] = uncertainty_scores
    df["priority_score"] = priority_scores
    df["model_decision"] = model_decisions
    df["seed_label"] = df[args.label_col].map(parse_label) if args.label_col in df.columns else ""

    selected = select_with_platform_diversity(
        df=df,
        batch_size=args.batch_size,
        seed=args.seed,
        platform_col=args.platform_col,
    )

    out = pd.DataFrame()
    out["record_id"] = selected["record_id"]
    out["text"] = selected["text"]
    out["platform"] = selected[args.platform_col].astype(str) if args.platform_col in selected.columns else ""
    out["source_url"] = selected[args.source_url_col].astype(str) if args.source_url_col in selected.columns else ""
    out["user"] = selected[args.user_col].astype(str) if args.user_col in selected.columns else ""
    out["date"] = selected[args.date_col].astype(str) if args.date_col in selected.columns else ""
    out["rating"] = selected[args.rating_col].astype(str) if args.rating_col in selected.columns else ""
    out["model_score"] = selected["model_score"].round(6)
    out["model_decision"] = selected["model_decision"]
    out["seed_label"] = selected["seed_label"]
    out["annotator_a_label"] = ""
    out["annotator_a_notes"] = ""
    out["annotator_b_label"] = ""
    out["annotator_b_notes"] = ""
    out["adjudicated_label"] = ""
    out["adjudication_notes"] = ""
    out["label_status"] = "pending"
    out["priority_score"] = selected["priority_score"].round(6)
    out["uncertainty_score"] = selected["uncertainty_score"].round(6)
    out["duplicate_text_count"] = selected["duplicate_text_count"].astype(int)
    out["risk_signals"] = selected["risk_signals"]
    out["promo_keyword_count"] = selected["promo_keyword_count"].round(4)
    out["caps_ratio"] = selected["caps_ratio"].round(4)
    out["exclamation_count"] = selected["exclamation_count"].astype(int)
    out["word_count"] = selected["word_count"].astype(int)

    output_path = Path(args.output_csv)
    output_a_path = Path(args.output_annotator_a_csv)
    output_b_path = Path(args.output_annotator_b_csv)
    summary_path = Path(args.output_summary_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_a_path.parent.mkdir(parents=True, exist_ok=True)
    output_b_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    out.to_csv(output_path, index=False)
    out.sample(frac=1.0, random_state=args.seed).to_csv(output_a_path, index=False)
    out.sample(frac=1.0, random_state=args.seed + 17).to_csv(output_b_path, index=False)

    summary = {
        "input_csv": str(input_path),
        "rows_considered": int(len(df)),
        "rows_selected": int(len(out)),
        "excluded_platforms": args.exclude_platform,
        "batch_size_requested": int(args.batch_size),
        "priority_score_range": [
            float(out["priority_score"].min()) if not out.empty else 0.0,
            float(out["priority_score"].max()) if not out.empty else 0.0,
        ],
        "platform_counts": {str(k): int(v) for k, v in out["platform"].value_counts().to_dict().items()},
        "top_signal_counts": {
            str(k): int(v)
            for k, v in (
                out["risk_signals"]
                .str.split("|", regex=False)
                .explode()
                .replace("", np.nan)
                .dropna()
                .value_counts()
                .head(20)
                .to_dict()
                .items()
            )
        },
        "policy_used": {
            "fraud_threshold": fraud_threshold,
            "review_band": review_band,
            "short_review_words": short_words,
        },
        "outputs": {
            "priority_csv": str(output_path),
            "annotator_a_csv": str(output_a_path),
            "annotator_b_csv": str(output_b_path),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Wrote: {output_path}")
    print(f"Wrote: {output_a_path}")
    print(f"Wrote: {output_b_path}")
    print(f"Wrote: {summary_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
