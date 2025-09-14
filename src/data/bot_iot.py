import os
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


DEFAULT_BOTIOT_DIR = "data/BoT-IoT"
DEFAULT_FILENAME = "BoT_IoT_consolidated.csv"


def load_botiot(root_dir: str = DEFAULT_BOTIOT_DIR, filename: str = DEFAULT_FILENAME, test_size: float = 0.2, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
	os.makedirs(root_dir, exist_ok=True)
	file_path = os.path.join(root_dir, filename)
	if not os.path.exists(file_path):
		raise FileNotFoundError(
			f"Expected consolidated BoT-IoT CSV at {file_path}. Please place the dataset CSVs or consolidated file as documented."
		)

	df = pd.read_csv(file_path)

	label_col = None
	for candidate in ["label", "Label", "Attack", "class"]:
		if candidate in df.columns:
			label_col = candidate
			break
	if label_col is None:
		raise ValueError("Could not find label column in BoT-IoT dataset.")

	labels = df[label_col].astype(str).str.lower()
	binary = (~labels.isin(["benign", "normal", "0"]))
	y = binary.astype(np.int32).values
	df = df.drop(columns=[label_col])

	categorical_cols = [c for c in df.columns if df[c].dtype == "object"]
	if categorical_cols:
		df = pd.get_dummies(df, columns=categorical_cols)

	df = df.replace([np.inf, -np.inf], np.nan).fillna(0)

	X = df.values.astype(np.float32)
	scaler = MinMaxScaler()
	X = scaler.fit_transform(X).astype(np.float32)
	X = X.reshape((X.shape[0], X.shape[1], 1))

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
	return X_train, X_test, y_train, y_test