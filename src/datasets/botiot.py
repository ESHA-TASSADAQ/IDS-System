from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from .common import basic_clean_dataframe, encode_categoricals, normalize_01, split_xy, make_sequences
from .base import DatasetSplits

# Common BoT-IoT columns vary by source; we'll expect a 'Label' or 'label' column.
POSSIBLE_LABELS = ["Label", "label", "Attack", "attack"]


def _find_label_col(df: pd.DataFrame) -> str:
    for c in POSSIBLE_LABELS:
        if c in df.columns:
            return c
    raise KeyError("No label column found in BoT-IoT dataset.")


def _load_csvs(data_dir: Path, files: Sequence[str] | None) -> pd.DataFrame:
    if files is None:
        files = [p.name for p in data_dir.glob("*.csv")]
    frames = []
    for fname in files:
        path = data_dir / fname
        if not path.exists():
            continue
        try:
            df = pd.read_csv(path)
        except Exception:
            df = pd.read_csv(path, encoding_errors="ignore")
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No BoT-IoT CSVs found in {data_dir}")
    df = pd.concat(frames, axis=0, ignore_index=True)
    return df


def load_botiot(
    data_dir: str | Path = "data",
    files: Sequence[str] | None = None,
    timesteps: int = 1,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
    sample_rows: int | None = None,
) -> DatasetSplits:
    data_dir = Path(data_dir) / "BOTIOT"
    data_dir.mkdir(parents=True, exist_ok=True)
    df = _load_csvs(data_dir, files)
    if sample_rows is not None and len(df) > sample_rows:
        df = df.sample(n=sample_rows, random_state=random_state)

    label_col = _find_label_col(df)

    # Standardize label to binary 0/1
    df = df.copy()
    if df[label_col].dtype == object:
        df[label_col] = df[label_col].astype(str).str.lower().map({"benign": 0, "normal": 0}).fillna(1).astype(int)
    else:
        # Assume numeric, map >0 to attack
        df[label_col] = (df[label_col].astype(float) > 0).astype(int)

    # Drop obvious non-feature columns if present
    drop_cols = [c for c in ["pkSeqID", "proto", "saddr", "daddr", "stime", "ltime", "smac", "dmac", "state"] if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    df = basic_clean_dataframe(df)

    # Categorical encoding
    categorical_cols = [c for c in df.columns if df[c].dtype == object and c != label_col]
    df, _ = encode_categoricals(df, categorical_cols)
    df, _ = normalize_01(df, exclude_cols=[label_col])

    X, y, feature_names = split_xy(df, label_col)

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=val_size, random_state=random_state, stratify=y_train
    )

    X_train = make_sequences(X_train, timesteps)
    X_val = make_sequences(X_val, timesteps)
    X_test = make_sequences(X_test, timesteps)

    return DatasetSplits(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        feature_names=feature_names,
    )