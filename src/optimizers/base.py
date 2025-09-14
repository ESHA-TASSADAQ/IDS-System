from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Callable


class HyperparameterOptimizer(ABC):
	"""Abstract base class for hyperparameter optimizers."""

	def __init__(self, objective_fn: Callable[[Dict[str, Any]], Tuple[float, Dict[str, Any]]], param_space: Dict[str, Tuple[Any, Any]], max_iters: int = 20, seed: int = 42):
		self.objective_fn = objective_fn
		self.param_space = param_space
		self.max_iters = max_iters
		self.seed = seed

	@abstractmethod
	def optimize(self) -> Tuple[Dict[str, Any], float, list]:
		"""Run optimization, return (best_params, best_score, history). Higher score is better."""
		raise NotImplementedError

	def clip_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
		clipped = {}
		for k, v in params.items():
			low, high = self.param_space[k]
			if isinstance(low, float) or isinstance(high, float):
				clipped[k] = float(min(max(v, low), high))
			else:
				clipped[k] = int(min(max(int(round(v)), low), high))
		return clipped