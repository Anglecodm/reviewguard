from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

import pandas as pd


def normalize_text(text: str) -> str:
    text = (text or "").lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def text_fingerprint(text: str) -> str:
    text = normalize_text(text)
    tokens = re.findall(r"[a-z0-9']+", text)
    tokens = [tok for tok in tokens if len(tok) >= 3]
    if not tokens:
        return ""
    uniq = sorted(set(tokens))
    return " ".join(uniq[:40])


def parse_label(value) -> int | None:
    if value is None:
        return None
    token = str(value).strip().lower()
    if token in {"0", "authentic", "real", "genuine", "legit", "clean"}:
        return 0
    if token in {"1", "fraud", "fake", "spam", "manipulated", "suspicious"}:
        return 1
    return None


def assign_groups_time_aware(
    df: pd.DataFrame,
    group_col: str,
    time_col: str,
    test_size: float,
    val_size: float,
    seed: int,
) -> pd.Series:
    group_sizes = df.groupby(group_col).size().to_dict()
    total_rows = len(df)
    target_test = int(round(total_rows * test_size))
    target_val = int(round(total_rows * val_size))

    if time_col in df.columns:
        parsed = pd.to_datetime(df[time_col], errors="coerce", utc=True, dayfirst=True)
    else:
        parsed = pd.Series([pd.NaT] * len(df), index=df.index)

    by_group = pd.DataFrame({group_col: df[group_col], "_time": parsed}).groupby(group_col)["_time"].max()
    dated = [g for g, t in by_group.items() if pd.notna(t)]
    undated = [g for g, t in by_group.items() if pd.isna(t)]

    dated_sorted_newest = sorted(dated, key=lambda g: by_group[g], reverse=True)
    rng = random.Random(seed)
    rng.shuffle(undated)

    test_groups: set[str] = set()
    val_groups: set[str] = set()

    count_test = 0
    for g in dated_sorted_newest:
        if count_test >= target_test:
            break
        test_groups.add(g)
        count_test += int(group_sizes.get(g, 0))

    count_val = 0
    for g in dated_sorted_newest:
        if g in test_groups:
            continue
        if count_val >= target_val:
            break
        val_groups.add(g)
        count_val += int(group_sizes.get(g, 0))

    # Backfill with undated groups if temporal groups were insufficient.
    for g in undated:
        if count_test < target_test:
            test_groups.add(g)
            count_test += int(group_sizes.get(g, 0))
            continue
        if count_val < target_val and g not in test_groups:
            val_groups.add(g)
            count_val += int(group_sizes.get(g, 0))

    split = pd.Series("train", index=df.index)
    split[df[group_col].isin(test_groups)] = "test"
    split[df[group_col].isin(val_groups)] = "val"
    return split


def split_label_gaps(df: pd.DataFrame, label_col: str = "label_parsed") -> dict[str, list[int]]:
    labels = sorted(int(x) for x in df[label_col].dropna().unique().tolist())
    gaps: dict[str, list[int]] = {}
    for split_name in ("val", "test"):
        counts = (
            df[df["split"] == split_name][label_col]
            .value_counts()
            .to_dict()
        )
        missing = [label for label in labels if int(counts.get(label, 0)) == 0]
        if missing:
            gaps[split_name] = missing
    return gaps


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare leakage-safe train/val/test splits (group + time + dedup).")
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--outdir", default="data/splits")
    parser.add_argument("--text-col", default="text")
    parser.add_argument("--label-col", default="label")
    parser.add_argument("--group-col", default="source_url")
    parser.add_argument("--time-col", default="date")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--val-size", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-text-len", type=int, default=20)
    parser.add_argument("--keep-columns", default="")
    args = parser.parse_args()

    if not 0 < args.test_size < 1:
        raise ValueError("--test-size must be between 0 and 1.")
    if not 0 <= args.val_size < 1:
        raise ValueError("--val-size must be between 0 and 1.")
    if args.test_size + args.val_size >= 1:
        raise ValueError("--test-size + --val-size must be less than 1.")

    input_path = Path(args.input_csv)
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    df = pd.read_csv(input_path)
    if args.text_col not in df.columns or args.label_col not in df.columns:
        raise ValueError(f"CSV must contain '{args.text_col}' and '{args.label_col}'.")

    df = df.copy()
    df["text"] = df[args.text_col].fillna("").astype(str)
    df["label_parsed"] = df[args.label_col].map(parse_label)
    df = df[df["label_parsed"].isin([0, 1])].copy()
    df = df[df["text"].str.len() >= args.min_text_len].copy()

    df["text_norm"] = df["text"].map(normalize_text)
    before_exact = len(df)
    df = df.drop_duplicates(subset=["text_norm"], keep="first").copy()
    dropped_exact = before_exact - len(df)

    df["text_fp"] = df["text"].map(text_fingerprint)
    before_fp = len(df)
    df = df.drop_duplicates(subset=["text_fp"], keep="first").copy()
    dropped_fp = before_fp - len(df)

    if args.group_col in df.columns:
        df["group_key"] = df[args.group_col].fillna("").astype(str).str.strip()
    else:
        df["group_key"] = ""
    missing_group = df["group_key"] == ""
    if missing_group.any():
        df.loc[missing_group, "group_key"] = df.loc[missing_group, "text_fp"].replace("", pd.NA)
        missing_group = df["group_key"].isna() | (df["group_key"] == "")
        if missing_group.any():
            df.loc[missing_group, "group_key"] = [f"row_{i}" for i in df.index[missing_group]]

    warnings: list[str] = []
    split = assign_groups_time_aware(
        df=df,
        group_col="group_key",
        time_col=args.time_col,
        test_size=args.test_size,
        val_size=args.val_size,
        seed=args.seed,
    )
    df["split"] = split
    initial_gaps = split_label_gaps(df)

    used_time_fallback = False
    if initial_gaps and args.time_col in df.columns:
        fallback_split = assign_groups_time_aware(
            df=df,
            group_col="group_key",
            time_col="__no_time__",
            test_size=args.test_size,
            val_size=args.val_size,
            seed=args.seed,
        )
        fallback_df = df.copy()
        fallback_df["split"] = fallback_split
        fallback_gaps = split_label_gaps(fallback_df)
        if len(fallback_gaps) < len(initial_gaps):
            df["split"] = fallback_split
            used_time_fallback = True
            warnings.append(
                "Time-aware split produced class gaps in val/test; switched to non-temporal grouped split."
            )

    remaining_gaps = split_label_gaps(df)
    if remaining_gaps:
        warnings.append(f"Class coverage gaps remain in val/test: {remaining_gaps}")

    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "val"].copy()
    test_df = df[df["split"] == "test"].copy()

    overlap_tv = set(train_df["group_key"]).intersection(set(val_df["group_key"]))
    overlap_tt = set(train_df["group_key"]).intersection(set(test_df["group_key"]))
    overlap_vt = set(val_df["group_key"]).intersection(set(test_df["group_key"]))

    keep_cols = [c.strip() for c in args.keep_columns.split(",") if c.strip()]
    base_cols = ["text", "label_parsed"]
    for col in [args.group_col, args.time_col, "platform", "source_url", "user", "rating", "record_id"]:
        if col in df.columns and col not in base_cols:
            base_cols.append(col)
    for col in keep_cols:
        if col in df.columns and col not in base_cols:
            base_cols.append(col)

    def export_frame(frame: pd.DataFrame) -> pd.DataFrame:
        out = frame[base_cols].copy()
        out = out.rename(columns={"label_parsed": "label"})
        return out

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    train_path = outdir / "train.csv"
    val_path = outdir / "val.csv"
    test_path = outdir / "test.csv"
    export_frame(train_df).to_csv(train_path, index=False)
    export_frame(val_df).to_csv(val_path, index=False)
    export_frame(test_df).to_csv(test_path, index=False)

    manifest = df[["text_norm", "group_key", "split"]].copy()
    manifest_path = outdir / "split_manifest.csv"
    manifest.to_csv(manifest_path, index=False)

    def label_counts(frame: pd.DataFrame) -> dict[str, int]:
        return {str(int(k)): int(v) for k, v in frame["label_parsed"].value_counts().sort_index().to_dict().items()}

    summary = {
        "input_csv": str(input_path),
        "rows_after_filtering": int(len(df)),
        "dropped_exact_duplicates": int(dropped_exact),
        "dropped_fingerprint_duplicates": int(dropped_fp),
        "splits": {
            "train": {"rows": int(len(train_df)), "label_counts": label_counts(train_df)},
            "val": {"rows": int(len(val_df)), "label_counts": label_counts(val_df)},
            "test": {"rows": int(len(test_df)), "label_counts": label_counts(test_df)},
        },
        "group_leakage": {
            "train_val_overlap": int(len(overlap_tv)),
            "train_test_overlap": int(len(overlap_tt)),
            "val_test_overlap": int(len(overlap_vt)),
        },
        "time_split_fallback_used": used_time_fallback,
        "warnings": warnings,
        "outputs": {
            "train_csv": str(train_path),
            "val_csv": str(val_path),
            "test_csv": str(test_path),
            "manifest_csv": str(manifest_path),
        },
    }
    summary_path = outdir / "split_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Wrote: {train_path}")
    print(f"Wrote: {val_path}")
    print(f"Wrote: {test_path}")
    print(f"Wrote: {manifest_path}")
    print(f"Wrote: {summary_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
