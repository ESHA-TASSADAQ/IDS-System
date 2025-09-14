import os
import random
import numpy as np

os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
os.environ.setdefault("PYTHONHASHSEED", "0")


def set_global_seed(seed: int = 42) -> None:
    """Set seeds for Python, NumPy, and TensorFlow.
    Lazily import TensorFlow to avoid overhead when not needed.
    """
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf  # lazy
        tf.random.set_seed(seed)
    except Exception:
        pass