from __future__ import annotations

from pathlib import Path
from typing import List, Sequence

import matplotlib.pyplot as plt


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_roc_curve(fpr, tpr, auc: float | None, out_path: str | Path, title: str = "ROC Curve") -> None:
    out_dir = ensure_dir(Path(out_path).parent)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"ROC AUC={auc:.4f}" if auc is not None else "ROC")
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    if auc is not None:
        plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def save_convergence_plot(values: Sequence[float], out_path: str | Path, title: str = "Convergence") -> None:
    out_dir = ensure_dir(Path(out_path).parent)
    plt.figure(figsize=(6, 5))
    plt.plot(list(range(1, len(values) + 1)), values, marker='o')
    plt.xlabel("Iteration")
    plt.ylabel("Best Fitness")
    plt.title(title)
    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()