from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_reviews(path: str | Path, text_col: str = "text", label_col: str = "label") -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    df = pd.read_csv(path)
    if text_col not in df.columns or label_col not in df.columns:
        raise ValueError(
            f"Dataset must include columns '{text_col}' and '{label_col}'. "
            f"Found columns: {list(df.columns)}"
        )
    df = df[[text_col, label_col]].copy()
    df = df.rename(columns={text_col: "text", label_col: "label"})
    df["text"] = df["text"].fillna("").astype(str)
    df["label"] = df["label"].astype(int)
    return df
