from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


def basic_clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(axis=0)
    df = df.drop_duplicates()
    return df


def encode_categoricals(df: pd.DataFrame, categorical_cols: Iterable[str]) -> Tuple[pd.DataFrame, dict]:
    df = df.copy()
    maps: dict[str, dict] = {}
    for col in categorical_cols:
        if col not in df.columns:
            continue
        df[col] = df[col].astype(str)
        uniques = sorted(df[col].unique())
        mapping = {val: idx for idx, val in enumerate(uniques)}
        df[col] = df[col].map(mapping).astype(np.int32)
        maps[col] = mapping
    return df, maps


def normalize_01(df: pd.DataFrame, exclude_cols: Iterable[str]) -> Tuple[pd.DataFrame, MinMaxScaler]:
    df = df.copy()
    feature_cols = [c for c in df.columns if c not in set(exclude_cols)]
    scaler = MinMaxScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols].astype(float))
    return df, scaler


def split_xy(df: pd.DataFrame, label_col: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    feature_cols = [c for c in df.columns if c != label_col]
    X = df[feature_cols].to_numpy(dtype=np.float32)
    y = df[label_col].to_numpy(dtype=np.int32)
    return X, y, feature_cols


def make_sequences(X: np.ndarray, timesteps: int) -> np.ndarray:
    # Reshape tabular features to sequences of length timesteps with feature_dim / timesteps per step if divisible,
    # otherwise expand dims as channels for Conv1D then LSTM over timesteps=1.
    if timesteps <= 1:
        # shape: (samples, steps=1, features)
        return np.expand_dims(X, axis=1)
    n_samples, n_features = X.shape
    if n_features % timesteps != 0:
        # fallback: repeat along time dimension
        X_repeat = np.repeat(np.expand_dims(X, axis=1), timesteps, axis=1)
        return X_repeat
    step_features = n_features // timesteps
    return X.reshape(n_samples, timesteps, step_features)


def train_val_test_split(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
    stratify: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y if stratify else None
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=val_size, random_state=random_state, stratify=y_train if stratify else None
    )
    return X_train, X_val, X_test, y_train, y_val, y_test