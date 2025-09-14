import os
import time
from typing import Dict, Any, Tuple, Callable

import numpy as np
import tensorflow as tf

from ..models.lstm_ids import LSTMIDSConfig, build_lstm_ids_model
from .metrics import compute_classification_metrics, plot_roc_curve, save_json


def train_and_evaluate(
	X_train: np.ndarray,
	X_test: np.ndarray,
	y_train: np.ndarray,
	y_test: np.ndarray,
	hparams: Dict[str, Any],
	seed: int = 42,
	epochs: int = 10,
	results_dir: str = "results",
	verbose: int = 0,
) -> Tuple[float, Dict[str, Any]]:
	os.makedirs(results_dir, exist_ok=True)

	config = LSTMIDSConfig(
		input_timesteps=X_train.shape[1],
		input_features=X_train.shape[2],
		conv_filters=int(hparams.get("conv_filters", 64)),
		conv_kernel_size=int(hparams.get("conv_kernel_size", 3)),
		pool_size=int(hparams.get("pool_size", 2)),
		lstm_units=int(hparams.get("lstm_units", 64)),
		dropout_rate=float(hparams.get("dropout", 0.3)),
		learning_rate=float(hparams.get("learning_rate", 1e-3)),
	)
	model = build_lstm_ids_model(config)

	batch_size = int(hparams.get("batch_size", 128))

	history = model.fit(
		X_train,
		y_train,
		validation_split=0.1,
		epochs=epochs,
		batch_size=batch_size,
		verbose=verbose,
	)
	# Use validation accuracy of last epoch as score for optimizers
	val_acc = float(history.history.get("val_accuracy", [0.0])[-1])

	y_pred_prob = model.predict(X_test, batch_size=batch_size, verbose=0).squeeze()
	metrics = compute_classification_metrics(y_test, y_pred_prob)
	metrics["val_accuracy_last"] = val_acc

	# Save basic artifacts
	timestamp = time.strftime("%Y%m%d_%H%M%S")
	run_dir = os.path.join(results_dir, f"run_{timestamp}")
	os.makedirs(run_dir, exist_ok=True)
	plot_roc_curve(y_test, y_pred_prob, os.path.join(run_dir, "roc_curve.png"))
	save_json({"hparams": hparams, "metrics": metrics}, os.path.join(run_dir, "results.json"))
	try:
		model.save(os.path.join(run_dir, "model.keras"))
	except Exception:
		pass

	return val_acc, metrics