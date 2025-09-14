from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np


@dataclass
class DatasetSplits:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    feature_names: list[str] | None = None


def load_dataset(name: str, data_dir: str | Path = "data", **kwargs) -> DatasetSplits:
    name_key = name.strip().lower()
    if name_key in {"nsl-kdd", "nsl_kdd", "nslkdd", "nsl"}:
        from .nsl_kdd import load_nsl_kdd
        return load_nsl_kdd(data_dir=data_dir, **kwargs)
    if name_key in {"cicids2017", "cic-ids2017", "cicids"}:
        from .cicids2017 import load_cicids2017
        return load_cicids2017(data_dir=data_dir, **kwargs)
    if name_key in {"bot-iot", "botiot", "bot_iot"}:
        from .botiot import load_botiot
        return load_botiot(data_dir=data_dir, **kwargs)
    raise ValueError(f"Unknown dataset name: {name}")