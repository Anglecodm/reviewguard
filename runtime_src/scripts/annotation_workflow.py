from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd
from sklearn.metrics import cohen_kappa_score


SCHEMA_COLUMNS = [
    "record_id",
    "text",
    "platform",
    "source_url",
    "user",
    "date",
    "rating",
    "model_score",
    "model_decision",
    "seed_label",
    "annotator_a_label",
    "annotator_a_notes",
    "annotator_b_label",
    "annotator_b_notes",
    "adjudicated_label",
    "adjudication_notes",
    "label_status",
]


def normalize_text(text: str) -> str:
    text = (text or "").lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def parse_label(value) -> int | None:
    if value is None:
        return None
    token = str(value).strip().lower()
    if not token:
        return None
    if token in {"0", "authentic", "real", "genuine", "legit", "clean"}:
        return 0
    if token in {"1", "fraud", "fake", "spam", "manipulated", "suspicious"}:
        return 1
    if token in {"uncertain", "unknown", "needs_review", "review", "-1", "2"}:
        return None
    return None


def bootstrap_tasks(args: argparse.Namespace) -> None:
    source = Path(args.input_csv)
    if not source.exists():
        raise FileNotFoundError(f"Input CSV not found: {source}")

    df = pd.read_csv(source)
    if args.text_col not in df.columns:
        raise ValueError(f"Missing text column '{args.text_col}' in input CSV.")

    if args.max_rows and args.max_rows > 0:
        df = df.head(args.max_rows).copy()

    df["text"] = df[args.text_col].fillna("").astype(str)
    df = df[df["text"].str.strip() != ""].copy()
    df["text_norm"] = df["text"].map(normalize_text)

    before = len(df)
    if args.dedupe:
        df = df.drop_duplicates(subset=["text_norm"], keep="first").copy()
    deduped = before - len(df)

    if args.id_col and args.id_col in df.columns:
        df["record_id"] = df[args.id_col].astype(str)
    else:
        df["record_id"] = [f"r{i:07d}" for i in range(1, len(df) + 1)]

    output = pd.DataFrame()
    output["record_id"] = df["record_id"]
    output["text"] = df["text"]
    output["platform"] = df[args.platform_col].astype(str) if args.platform_col in df.columns else ""
    output["source_url"] = df[args.source_url_col].astype(str) if args.source_url_col in df.columns else ""
    output["user"] = df[args.user_col].astype(str) if args.user_col in df.columns else ""
    output["date"] = df[args.date_col].astype(str) if args.date_col in df.columns else ""
    output["rating"] = df[args.rating_col].astype(str) if args.rating_col in df.columns else ""
    output["model_score"] = df[args.model_score_col] if args.model_score_col in df.columns else ""
    output["model_decision"] = df[args.model_decision_col] if args.model_decision_col in df.columns else ""
    output["seed_label"] = df[args.seed_label_col] if args.seed_label_col in df.columns else ""
    output["annotator_a_label"] = ""
    output["annotator_a_notes"] = ""
    output["annotator_b_label"] = ""
    output["annotator_b_notes"] = ""
    output["adjudicated_label"] = ""
    output["adjudication_notes"] = ""
    output["label_status"] = "pending"
    output = output[SCHEMA_COLUMNS]

    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(out_path, index=False)

    print(f"Wrote annotation task file: {out_path}")
    print(f"Rows: {len(output)} (dedup removed {deduped})")


def adjudicate(args: argparse.Namespace) -> None:
    source = Path(args.input_csv)
    if not source.exists():
        raise FileNotFoundError(f"Annotation CSV not found: {source}")

    df = pd.read_csv(source)
    required = {"record_id", "text", "annotator_a_label", "annotator_b_label", "adjudicated_label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required annotation columns: {sorted(missing)}")

    a_labels = df["annotator_a_label"].map(parse_label)
    b_labels = df["annotator_b_label"].map(parse_label)
    adj_labels = df["adjudicated_label"].map(parse_label)

    final = []
    unresolved = []
    agreement_rows = []

    for idx, row in df.iterrows():
        a = a_labels.iloc[idx]
        b = b_labels.iloc[idx]
        adj = adj_labels.iloc[idx]

        resolved_label = None
        status = "pending"
        route = ""

        if adj is not None:
            resolved_label = int(adj)
            status = "adjudicated"
            route = "adjudicated"
        elif a is not None and b is not None and a == b:
            resolved_label = int(a)
            status = "agreed"
            route = "double_agreement"
        elif not args.strict and a is not None and b is None:
            resolved_label = int(a)
            status = "single_accepted"
            route = "annotator_a_only"
        elif not args.strict and b is not None and a is None:
            resolved_label = int(b)
            status = "single_accepted"
            route = "annotator_b_only"
        else:
            unresolved.append(row)
            continue

        out = {
            "record_id": row["record_id"],
            "text": row["text"],
            "label": resolved_label,
            "platform": row.get("platform", ""),
            "source_url": row.get("source_url", ""),
            "user": row.get("user", ""),
            "date": row.get("date", ""),
            "rating": row.get("rating", ""),
            "label_source": route,
            "label_status": status,
            "annotator_a_label": row.get("annotator_a_label", ""),
            "annotator_b_label": row.get("annotator_b_label", ""),
            "adjudicated_label": row.get("adjudicated_label", ""),
        }
        final.append(out)

        if a is not None and b is not None:
            agreement_rows.append((a, b))

    final_df = pd.DataFrame(final)
    unresolved_df = pd.DataFrame(unresolved) if unresolved else pd.DataFrame(columns=df.columns)

    out_path = Path(args.output_csv)
    unresolved_path = Path(args.output_unresolved_csv)
    summary_path = Path(args.output_summary_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    unresolved_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    final_df.to_csv(out_path, index=False)
    unresolved_df.to_csv(unresolved_path, index=False)

    agreement = None
    kappa = None
    if agreement_rows:
        a_series = [a for a, _ in agreement_rows]
        b_series = [b for _, b in agreement_rows]
        agreement = float(sum(1 for a, b in agreement_rows if a == b) / len(agreement_rows))
        kappa = float(cohen_kappa_score(a_series, b_series))

    label_counts = (
        {str(k): int(v) for k, v in final_df["label"].value_counts(dropna=False).sort_index().to_dict().items()}
        if not final_df.empty
        else {}
    )
    summary = {
        "input_rows": int(len(df)),
        "resolved_rows": int(len(final_df)),
        "unresolved_rows": int(len(unresolved_df)),
        "strict_mode": bool(args.strict),
        "label_counts": label_counts,
        "annotator_overlap_rows": int(len(agreement_rows)),
        "annotator_raw_agreement": agreement,
        "annotator_cohen_kappa": kappa,
        "outputs": {
            "labeled_csv": str(out_path),
            "unresolved_csv": str(unresolved_path),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Wrote labeled CSV: {out_path}")
    print(f"Wrote unresolved CSV: {unresolved_path}")
    print(f"Wrote summary JSON: {summary_path}")
    print(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Annotation bootstrap + adjudication workflow.")
    sub = parser.add_subparsers(dest="command", required=True)

    boot = sub.add_parser("bootstrap", help="Create annotation task CSV with standard schema.")
    boot.add_argument("--input-csv", required=True)
    boot.add_argument("--output-csv", default="data/annotation_tasks.csv")
    boot.add_argument("--text-col", default="text")
    boot.add_argument("--id-col", default="")
    boot.add_argument("--platform-col", default="platform")
    boot.add_argument("--source-url-col", default="source_url")
    boot.add_argument("--user-col", default="user")
    boot.add_argument("--date-col", default="date")
    boot.add_argument("--rating-col", default="rating")
    boot.add_argument("--model-score-col", default="model_score")
    boot.add_argument("--model-decision-col", default="model_decision")
    boot.add_argument("--seed-label-col", default="label")
    boot.add_argument("--dedupe", action="store_true")
    boot.add_argument("--max-rows", type=int, default=0)

    adj = sub.add_parser("adjudicate", help="Resolve labels and export final training CSV.")
    adj.add_argument("--input-csv", required=True)
    adj.add_argument("--output-csv", default="data/labeled_reviews_adjudicated.csv")
    adj.add_argument("--output-unresolved-csv", default="data/labeled_reviews_unresolved.csv")
    adj.add_argument("--output-summary-json", default="data/labeled_reviews_summary.json")
    adj.add_argument(
        "--strict",
        action="store_true",
        help="Only accept adjudicated or dual-annotator agreement labels.",
    )

    args = parser.parse_args()
    if args.command == "bootstrap":
        bootstrap_tasks(args)
    elif args.command == "adjudicate":
        adjudicate(args)


if __name__ == "__main__":
    main()
