import os
from typing import List
import matplotlib.pyplot as plt


def plot_convergence(history: List[float], save_path: str) -> None:
	os.makedirs(os.path.dirname(save_path), exist_ok=True)
	plt.figure()
	plt.plot(history, marker='o')
	plt.xlabel('Iteration')
	plt.ylabel('Best Validation Accuracy')
	plt.title('Optimizer Convergence')
	plt.grid(True, ls='--', alpha=0.4)
	plt.savefig(save_path, bbox_inches='tight')
	plt.close()