from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download


def parse_label(value) -> int | None:
    if value is None:
        return None
    token = str(value).strip().lower()
    if token in {"0", "authentic", "real", "genuine", "legit", "clean", "non-deceptive"}:
        return 0
    if token in {"1", "fraud", "fake", "spam", "manipulated", "suspicious", "deceptive"}:
        return 1
    return None


def normalize_text(text: str) -> str:
    text = (text or "").lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_jsonl(path: Path) -> list[dict]:
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def sample_per_class(df: pd.DataFrame, per_class: int, seed: int) -> pd.DataFrame:
    if per_class <= 0:
        return df.copy()
    rng = random.Random(seed)
    parts = []
    for label_value, part in df.groupby("label"):
        if len(part) <= per_class:
            parts.append(part)
            continue
        idx = list(part.index)
        rng.shuffle(idx)
        take = idx[:per_class]
        parts.append(part.loc[take].copy())
    return pd.concat(parts, axis=0).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build hybrid bootstrap dataset from marketplace + external labeled product reviews.")
    parser.add_argument("--marketplace-csv", default="data/marketplace_reviews_weak_labeled.csv")
    parser.add_argument("--output-csv", default="data/hybrid_bootstrap_reviews.csv")
    parser.add_argument("--output-summary-json", default="data/hybrid_bootstrap_reviews_summary.json")
    parser.add_argument("--hf-repo-id", default="difraud/difraud")
    parser.add_argument("--hf-domain", default="product_reviews")
    parser.add_argument("--external-per-class", type=int, default=1500)
    parser.add_argument("--include-marketplace-fraud", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-text-len", type=int, default=20)
    args = parser.parse_args()

    marketplace_path = Path(args.marketplace_csv)
    if not marketplace_path.exists():
        raise FileNotFoundError(f"Marketplace CSV not found: {marketplace_path}")

    market = pd.read_csv(marketplace_path).copy()
    if "text" not in market.columns or "label" not in market.columns:
        raise ValueError("Marketplace CSV must contain text and label columns.")

    market["text"] = market["text"].fillna("").astype(str)
    market = market[market["text"].str.len() >= args.min_text_len].copy()
    market["label"] = market["label"].map(parse_label)
    market = market[market["label"].isin([0, 1])].copy()

    if "platform" in market.columns and not args.include_marketplace_fraud:
        platform_norm = market["platform"].fillna("").astype(str).str.lower()
        market = market[platform_norm != "synthetic"].copy()

    if args.include_marketplace_fraud:
        market_use = market.copy()
    else:
        market_use = market[market["label"] == 0].copy()

    if market_use.empty:
        raise ValueError("No marketplace rows available after filtering.")

    market_out = pd.DataFrame(
        {
            "text": market_use["text"],
            "label": market_use["label"].astype(int),
            "platform": market_use.get("platform", "marketplace").astype(str)
            if "platform" in market_use.columns
            else "marketplace",
            "source_url": market_use.get("source_url", "").astype(str)
            if "source_url" in market_use.columns
            else "",
            "user": market_use.get("user", "").astype(str) if "user" in market_use.columns else "",
            "date": market_use.get("date", "").astype(str) if "date" in market_use.columns else "",
            "rating": market_use.get("rating", "").astype(str) if "rating" in market_use.columns else "",
            "label_source": "marketplace_real",
        }
    )

    files = [
        f"{args.hf_domain}/train.jsonl",
        f"{args.hf_domain}/validation.jsonl",
        f"{args.hf_domain}/test.jsonl",
    ]
    external_rows: list[dict] = []
    for fname in files:
        local_file = Path(hf_hub_download(repo_id=args.hf_repo_id, repo_type="dataset", filename=fname))
        records = load_jsonl(local_file)
        for rec in records:
            text = str(rec.get("text", "")).strip()
            label = parse_label(rec.get("label"))
            if not text or label is None:
                continue
            if len(text) < args.min_text_len:
                continue
            external_rows.append(
                {
                    "text": text,
                    "label": int(label),
                    "platform": "difraud_product_reviews",
                    "source_url": "",
                    "user": "",
                    "date": "",
                    "rating": "",
                    "label_source": "difraud_product_reviews",
                }
            )

    if not external_rows:
        raise ValueError("No external DIFrauD rows loaded.")

    external_df = pd.DataFrame(external_rows)
    external_df = sample_per_class(external_df, per_class=args.external_per_class, seed=args.seed)

    combined = pd.concat([market_out, external_df], axis=0, ignore_index=True)
    combined["text_norm"] = combined["text"].map(normalize_text)
    before = len(combined)
    combined = combined.drop_duplicates(subset=["text_norm"], keep="first").copy()
    dropped_dupes = before - len(combined)
    combined = combined.drop(columns=["text_norm"])

    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out_path, index=False)

    summary = {
        "marketplace_csv": str(marketplace_path),
        "hf_repo_id": args.hf_repo_id,
        "hf_domain": args.hf_domain,
        "include_marketplace_fraud": bool(args.include_marketplace_fraud),
        "external_per_class": int(args.external_per_class),
        "rows_output": int(len(combined)),
        "dropped_duplicates": int(dropped_dupes),
        "label_counts": {str(k): int(v) for k, v in combined["label"].value_counts().sort_index().to_dict().items()},
        "label_source_counts": {
            str(k): int(v) for k, v in combined["label_source"].value_counts().to_dict().items()
        },
        "platform_counts": {str(k): int(v) for k, v in combined["platform"].value_counts().to_dict().items()},
        "output_csv": str(out_path),
    }
    summary_path = Path(args.output_summary_json)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Wrote: {out_path}")
    print(f"Wrote: {summary_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
