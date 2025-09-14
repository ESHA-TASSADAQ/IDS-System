from typing import Dict, Any, Tuple, List
import random

from .base import HyperparameterOptimizer


class JAYAOptimizer(HyperparameterOptimizer):
	def __init__(self, *args, population_size: int = 10, **kwargs):
		super().__init__(*args, **kwargs)
		self.population_size = population_size

	def _random_params(self) -> Dict[str, Any]:
		params = {}
		for k, (low, high) in self.param_space.items():
			if isinstance(low, float) or isinstance(high, float):
				params[k] = random.uniform(float(low), float(high))
			else:
				params[k] = random.randint(int(low), int(high))
		return params

	def optimize(self) -> Tuple[Dict[str, Any], float, List[float]]:
		random.seed(self.seed)
		population = [self._random_params() for _ in range(self.population_size)]
		scores = []
		for p in population:
			score, _ = self.objective_fn(self.clip_params(p))
			scores.append(score)
		best_idx = max(range(self.population_size), key=lambda i: scores[i])
		best = population[best_idx].copy()
		best_score = scores[best_idx]
		worst_idx = min(range(self.population_size), key=lambda i: scores[i])
		worst = population[worst_idx].copy()

		history = [best_score]
		for _ in range(self.max_iters):
			new_population = []
			for i in range(self.population_size):
				candidate = {}
				for k in self.param_space.keys():
					r1, r2 = random.random(), random.random()
					candidate[k] = population[i][k] + r1 * (best[k] - abs(population[i][k])) - r2 * (worst[k] - abs(population[i][k]))
				candidate = self.clip_params(candidate)
				new_population.append(candidate)
			# Evaluate new population
			population = new_population
			scores = []
			for p in population:
				score, _ = self.objective_fn(p)
				scores.append(score)
			best_idx = max(range(self.population_size), key=lambda i: scores[i])
			best = population[best_idx].copy()
			best_score = scores[best_idx]
			worst_idx = min(range(self.population_size), key=lambda i: scores[i])
			worst = population[worst_idx].copy()
			history.append(best_score)
		return best, best_score, history