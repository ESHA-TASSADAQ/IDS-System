from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd
import numpy as np
import urllib.request
import zipfile

from .common import basic_clean_dataframe, encode_categoricals, normalize_01, split_xy, make_sequences
from .base import DatasetSplits

NSL_MIRROR = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/"
TRAIN_FILE = "KDDTrain+.txt"
TEST_FILE = "KDDTest+.txt"

NSL_COLUMNS = [
    "duration","protocol_type","service","flag","src_bytes","dst_bytes","land","wrong_fragment","urgent",
    "hot","num_failed_logins","logged_in","num_compromised","root_shell","su_attempted","num_root","num_file_creations",
    "num_shells","num_access_files","num_outbound_cmds","is_host_login","is_guest_login","count","srv_count","serror_rate",
    "srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate","diff_srv_rate","srv_diff_host_rate","dst_host_count",
    "dst_host_srv_count","dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate","dst_host_rerror_rate",
    "dst_host_srv_rerror_rate","label","difficulty"
]

CATEGORICAL_COLS = ["protocol_type", "service", "flag"]
LABEL_COL = "label"

ATTACK_LABELS_NORMAL = {"normal"}


def _ensure_download(data_dir: Path) -> Tuple[Path, Path]:
    data_dir.mkdir(parents=True, exist_ok=True)
    train_path = data_dir / TRAIN_FILE
    test_path = data_dir / TEST_FILE
    if not train_path.exists():
        urllib.request.urlretrieve(NSL_MIRROR + TRAIN_FILE, train_path)
    if not test_path.exists():
        urllib.request.urlretrieve(NSL_MIRROR + TEST_FILE, test_path)
    return train_path, test_path


def _load_raw_nsl(train_path: Path, test_path: Path) -> pd.DataFrame:
    train_df = pd.read_csv(train_path, names=NSL_COLUMNS)
    test_df = pd.read_csv(test_path, names=NSL_COLUMNS)
    df = pd.concat([train_df, test_df], axis=0, ignore_index=True)
    return df


def _binarize_labels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df[LABEL_COL] = (df[LABEL_COL].astype(str).str.lower().apply(lambda x: 0 if x in ATTACK_LABELS_NORMAL else 1)).astype(int)
    return df


def load_nsl_kdd(
    data_dir: str | Path = "data",
    timesteps: int = 1,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
) -> DatasetSplits:
    data_dir = Path(data_dir) / "NSL-KDD"
    train_path, test_path = _ensure_download(data_dir)
    df = _load_raw_nsl(train_path, test_path)
    df = _binarize_labels(df)
    df = basic_clean_dataframe(df)
    df, _ = encode_categoricals(df, CATEGORICAL_COLS)
    df, _ = normalize_01(df, exclude_cols=[LABEL_COL, "difficulty"])
    # Drop difficulty
    if "difficulty" in df.columns:
        df = df.drop(columns=["difficulty"]) 
    X, y, feature_names = split_xy(df, LABEL_COL)

    # Split
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=val_size, random_state=random_state, stratify=y_train
    )

    # Sequence shaping
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