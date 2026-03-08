from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def run_cmd(cmd: list[str], cwd: Path) -> None:
    print("RUN:", " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd), check=True)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def update_threshold_in_metadata(metadata_path: Path, threshold: float) -> None:
    meta = read_json(metadata_path)
    policy = meta.get("decision_policy", {})
    policy["fraud_threshold"] = float(threshold)
    meta["decision_policy"] = policy
    write_json(metadata_path, meta)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full train/eval pipeline on adjudicated labels.")
    parser.add_argument("--input-csv", default="data/labeled_reviews_adjudicated.csv")
    parser.add_argument("--text-col", default="text")
    parser.add_argument("--label-col", default="label")
    parser.add_argument("--group-col", default="source_url")
    parser.add_argument("--time-col", default="date")
    parser.add_argument("--model", default="random_forest")
    parser.add_argument("--target-precision", type=float, default=0.9)
    parser.add_argument("--prefix", default="adjudicated_v1")
    parser.add_argument("--splits-dir", default="")
    parser.add_argument("--models-dir", default="")
    parser.add_argument("--reports-dir", default="reports")
    parser.add_argument("--project-root", default="")
    args = parser.parse_args()

    root = Path(args.project_root).resolve() if args.project_root else Path(__file__).resolve().parents[1]
    python = sys.executable

    input_csv = Path(args.input_csv)
    if not input_csv.is_absolute():
        input_csv = root / input_csv
    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    splits_dir = Path(args.splits_dir) if args.splits_dir else Path("data") / f"splits_{args.prefix}"
    models_dir = Path(args.models_dir) if args.models_dir else Path(f"models_{args.prefix}")
    reports_dir = Path(args.reports_dir)
    if not splits_dir.is_absolute():
        splits_dir = root / splits_dir
    if not models_dir.is_absolute():
        models_dir = root / models_dir
    if not reports_dir.is_absolute():
        reports_dir = root / reports_dir

    prepare_script = root / "scripts" / "prepare_splits.py"
    eval_script = root / "scripts" / "evaluate_thresholds.py"

    run_cmd(
        [
            python,
            str(prepare_script),
            "--input-csv",
            str(input_csv),
            "--outdir",
            str(splits_dir),
            "--text-col",
            args.text_col,
            "--label-col",
            args.label_col,
            "--group-col",
            args.group_col,
            "--time-col",
            args.time_col,
        ],
        cwd=root,
    )

    train_csv = splits_dir / "train.csv"
    val_csv = splits_dir / "val.csv"
    test_csv = splits_dir / "test.csv"
    if not train_csv.exists() or not val_csv.exists() or not test_csv.exists():
        raise FileNotFoundError("Split files missing after prepare_splits run.")

    run_cmd(
        [
            python,
            "-m",
            "reviewguard.train",
            "--data",
            str(train_csv),
            "--text-col",
            "text",
            "--label-col",
            "label",
            "--model",
            args.model,
            "--evaluate",
            "--outdir",
            str(models_dir),
            "--test-size",
            "0.2",
            "--val-size",
            "0.1",
            "--random-state",
            "42",
        ],
        cwd=root,
    )

    model_path = models_dir / "reviewguard_model.joblib"
    metadata_path = models_dir / "reviewguard_metadata.json"
    if not model_path.exists() or not metadata_path.exists():
        raise FileNotFoundError("Trained model artifacts missing.")

    run_cmd(
        [
            python,
            str(eval_script),
            "--input-csv",
            str(val_csv),
            "--model-path",
            str(model_path),
            "--metadata-path",
            str(metadata_path),
            "--target-precision",
            str(args.target_precision),
            "--outdir",
            str(reports_dir),
            "--prefix",
            f"{args.prefix}_val",
        ],
        cwd=root,
    )

    val_summary_path = reports_dir / f"{args.prefix}_val_summary.json"
    val_summary = read_json(val_summary_path)
    recommended_threshold = float(val_summary.get("recommended_threshold", 0.65))
    update_threshold_in_metadata(metadata_path, recommended_threshold)

    run_cmd(
        [
            python,
            str(eval_script),
            "--input-csv",
            str(test_csv),
            "--model-path",
            str(model_path),
            "--metadata-path",
            str(metadata_path),
            "--target-precision",
            str(args.target_precision),
            "--outdir",
            str(reports_dir),
            "--prefix",
            f"{args.prefix}_test",
        ],
        cwd=root,
    )

    test_summary_path = reports_dir / f"{args.prefix}_test_summary.json"
    split_summary_path = splits_dir / "split_summary.json"
    holdout_path = models_dir / "holdout_metrics.json"

    summary = {
        "input_csv": str(input_csv),
        "prefix": args.prefix,
        "selected_model": args.model,
        "recommended_threshold_from_val": recommended_threshold,
        "paths": {
            "splits_dir": str(splits_dir),
            "models_dir": str(models_dir),
            "reports_dir": str(reports_dir),
            "split_summary_json": str(split_summary_path),
            "val_summary_json": str(val_summary_path),
            "test_summary_json": str(test_summary_path),
            "holdout_metrics_json": str(holdout_path),
            "metadata_json": str(metadata_path),
        },
    }
    pipeline_summary_path = reports_dir / f"{args.prefix}_pipeline_summary.json"
    write_json(pipeline_summary_path, summary)
    print(f"Wrote: {pipeline_summary_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
