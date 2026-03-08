from __future__ import annotations

import argparse
from pathlib import Path

from .service import ReviewGuardService


def main():
    parser = argparse.ArgumentParser(description="Predict review authenticity.")
    parser.add_argument("--model", default="models/reviewguard_model.joblib", help="Path to trained model.")
    parser.add_argument("--metadata", default="models/reviewguard_metadata.json", help="Path to metadata JSON.")
    parser.add_argument("--text", help="Single review text to score.")
    parser.add_argument("--file", help="Path to a text file (one review per line).")
    parser.add_argument("--threshold", type=float, default=None, help="Override fraud threshold.")
    args = parser.parse_args()

    if not args.text and not args.file:
        raise SystemExit("Provide --text or --file.")

    service = ReviewGuardService(args.model, args.metadata)

    if args.text:
        texts = [args.text]
    else:
        texts = [line.strip() for line in Path(args.file).read_text(encoding="utf-8").splitlines() if line.strip()]

    predictions = service.predict(texts, threshold=args.threshold)

    for text, pred in zip(texts, predictions):
        reason = pred.reason or "-"
        prob_pct = pred.score * 100.0
        print(f"{pred.decision}\t{pred.label}\t{prob_pct:.2f}%\t{reason}\t{text[:120]}")


if __name__ == "__main__":
    main()
