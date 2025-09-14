import os
import random
import numpy as np
import tensorflow as tf


def set_global_seed(seed: int = 42) -> None:
	"""Set seeds for reproducibility across Python, NumPy, and TensorFlow."""
	random.seed(seed)
	np.random.seed(seed)
	os.environ["PYTHONHASHSEED"] = str(seed)
	# TensorFlow determinism best-effort
	try:
		from tensorflow.keras import backend as K  # noqa: F401
		tf.random.set_seed(seed)
		os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
		os.environ.setdefault("TF_CUDNN_DETERMINISTIC", "1")
	except Exception:
		pass