from typing import Dict, Any
import os
import json
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
import matplotlib.pyplot as plt


def compute_classification_metrics(y_true: np.ndarray, y_pred_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
	pred_labels = (y_pred_prob >= threshold).astype(int)
	metrics = {
		"accuracy": float(accuracy_score(y_true, pred_labels)),
		"precision": float(precision_score(y_true, pred_labels, zero_division=0)),
		"recall": float(recall_score(y_true, pred_labels, zero_division=0)),
		"f1": float(f1_score(y_true, pred_labels, zero_division=0)),
	}
	fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
	metrics["roc_auc"] = float(auc(fpr, tpr))
	return metrics


def plot_roc_curve(y_true: np.ndarray, y_pred_prob: np.ndarray, save_path: str) -> None:
	fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
	roc_auc = auc(fpr, tpr)
	plt.figure()
	plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.3f})')
	plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('Receiver Operating Characteristic')
	plt.legend(loc="lower right")
	os.makedirs(os.path.dirname(save_path), exist_ok=True)
	plt.savefig(save_path, bbox_inches='tight')
	plt.close()


def save_json(obj: Any, path: str) -> None:
	os.makedirs(os.path.dirname(path), exist_ok=True)
	with open(path, 'w') as f:
		json.dump(obj, f, indent=2)