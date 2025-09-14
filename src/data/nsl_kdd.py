import os
import io
import zipfile
import requests
from typing import Tuple, Dict, Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


NSL_KDD_URLS = {
	"train": "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain+.txt",
	"test": "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTest+.txt",
}

# Columns list from NSL-KDD definition
NSL_KDD_COLUMNS = [
	"duration","protocol_type","service","flag","src_bytes","dst_bytes","land","wrong_fragment","urgent",
	"hot","num_failed_logins","logged_in","num_compromised","root_shell","su_attempted","num_root","num_file_creations",
	"num_shells","num_access_files","num_outbound_cmds","is_host_login","is_guest_login","count","srv_count","serror_rate",
	"srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate","diff_srv_rate","srv_diff_host_rate","dst_host_count",
	"dst_host_srv_count","dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate","dst_host_srv_diff_host_rate",
	"dst_host_serror_rate","dst_host_srv_serror_rate","dst_host_rerror_rate","dst_host_srv_rerror_rate","label","difficulty"
]

BINARY_ATTACK_MAPPING = {
	"normal": 0,
}


def _download_file(url: str, dest_path: str) -> None:
	os.makedirs(os.path.dirname(dest_path), exist_ok=True)
	resp = requests.get(url, timeout=60)
	resp.raise_for_status()
	with open(dest_path, "wb") as f:
		f.write(resp.content)


def load_nsl_kdd(root_dir: str = "data/NSL-KDD", test_size: float = 0.2, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
	os.makedirs(root_dir, exist_ok=True)
	train_path = os.path.join(root_dir, "KDDTrain+.txt")
	test_path = os.path.join(root_dir, "KDDTest+.txt")
	if not os.path.exists(train_path):
		_download_file(NSL_KDD_URLS["train"], train_path)
	if not os.path.exists(test_path):
		_download_file(NSL_KDD_URLS["test"], test_path)

	train_df = pd.read_csv(train_path, names=NSL_KDD_COLUMNS)
	test_df = pd.read_csv(test_path, names=NSL_KDD_COLUMNS)
	full_df = pd.concat([train_df, test_df], ignore_index=True)

	# Binary label: normal -> 0, attack -> 1
	full_df["binary_label"] = (full_df["label"].str.lower() != "normal").astype(int)
	full_df.drop(columns=["label","difficulty"], inplace=True)

	# Categorical columns
	categorical_cols = ["protocol_type", "service", "flag"]
	full_df = pd.get_dummies(full_df, columns=categorical_cols)

	# Split features and labels
	X = full_df.drop(columns=["binary_label"]).values.astype(np.float32)
	y = full_df["binary_label"].values.astype(np.int32)

	# Normalize features to [0,1]
	scaler = MinMaxScaler()
	X = scaler.fit_transform(X).astype(np.float32)

	# For sequential modeling: treat features as sequence steps with 1 channel
	X = X.reshape((X.shape[0], X.shape[1], 1))

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
	return X_train, X_test, y_train, y_test