from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np
import pandas as pd

from .common import basic_clean_dataframe, encode_categoricals, normalize_01, split_xy, make_sequences
from .base import DatasetSplits

# Expected CSVs names if placed manually under data/CICIDS2017
DEFAULT_FILES = [
    # These are common file names used in public mirrors; users can adjust.
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
    "Friday-WorkingHours-Morning.pcap_ISCX.csv",
    "Monday-WorkingHours.pcap_ISCX.csv",
    "Tuesday-WorkingHours.pcap_ISCX.csv",
    "Wednesday-workingHours.pcap_ISCX.csv",
    "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
    "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
]

LABEL_COL = "Label"
BINARY_ATTACK_LABEL = "BENIGN"


def _load_csvs(data_dir: Path, files: Sequence[str] | None) -> pd.DataFrame:
    if files is None:
        files = [f for f in DEFAULT_FILES if (data_dir / f).exists()]
    if not files:
        raise FileNotFoundError(
            f"No CICIDS2017 CSV files found in {data_dir}. Place CSVs or provide file list."
        )
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
        raise FileNotFoundError("No readable CICIDS2017 CSVs found.")
    df = pd.concat(frames, axis=0, ignore_index=True)
    return df


def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Strip spaces and unify column names
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _binarize_labels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df[LABEL_COL] = (df[LABEL_COL].astype(str) != BINARY_ATTACK_LABEL).astype(int)
    return df


def load_cicids2017(
    data_dir: str | Path = "data",
    files: Sequence[str] | None = None,
    drop_cols: Sequence[str] = ("Flow ID", "Source IP", "Destination IP", "Timestamp"),
    timesteps: int = 1,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
    sample_rows: int | None = None,
) -> DatasetSplits:
    data_dir = Path(data_dir) / "CICIDS2017"
    data_dir.mkdir(parents=True, exist_ok=True)
    df = _load_csvs(data_dir, files)
    if sample_rows is not None and len(df) > sample_rows:
        df = df.sample(n=sample_rows, random_state=random_state)
    df = _standardize_columns(df)
    # Drop non-numeric IDs and timestamps if present
    for col in drop_cols:
        if col in df.columns:
            df = df.drop(columns=[col])
    df = basic_clean_dataframe(df)
    # Identify categorical columns
    categorical_cols = [c for c in df.columns if df[c].dtype == object and c != LABEL_COL]
    df, _ = encode_categoricals(df, categorical_cols)
    df, _ = normalize_01(df, exclude_cols=[LABEL_COL])

    X, y, feature_names = split_xy(df, LABEL_COL)

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