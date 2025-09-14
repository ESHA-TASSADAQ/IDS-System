from typing import Dict, Any, Tuple, List
import random
import math

from .base import HyperparameterOptimizer


class SSAOptimizer(HyperparameterOptimizer):
	def __init__(self, *args, population_size: int = 20, discoverer_ratio: float = 0.2, danger_ratio: float = 0.1, **kwargs):
		super().__init__(*args, **kwargs)
		self.population_size = population_size
		self.discoverer_ratio = discoverer_ratio
		self.danger_ratio = danger_ratio

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
		pop = [self._random_params() for _ in range(self.population_size)]
		scores = []
		for p in pop:
			s, _ = self.objective_fn(self.clip_params(p))
			scores.append(s)

		history = [max(scores)]
		for t in range(self.max_iters):
			# Sort by fitness descending
			indices = list(range(self.population_size))
			indices.sort(key=lambda i: scores[i], reverse=True)
			sorted_pop = [pop[i] for i in indices]
			sorted_scores = [scores[i] for i in indices]

			n_discoverers = max(1, int(self.discoverer_ratio * self.population_size))
			n_warners = max(1, int(self.danger_ratio * self.population_size))

			new_pop = []
			best = sorted_pop[0]
			worst = sorted_pop[-1]
			for i, indiv in enumerate(sorted_pop):
				candidate = indiv.copy()
				if i < n_discoverers:
					# discoverers update
					candidate = {k: indiv[k] * (1 + random.uniform(-1, 1)) for k in indiv}
				elif i >= self.population_size - n_warners:
					# warners (aware of danger)
					candidate = {k: indiv[k] + random.uniform(-1, 1) * abs(indiv[k] - best[k]) for k in indiv}
				else:
					# followers move towards best, away from worst
					candidate = {k: indiv[k] + random.random() * (best[k] - abs(indiv[k])) - random.random() * (worst[k] - abs(indiv[k])) for k in indiv}
				candidate = self.clip_params(candidate)
				new_pop.append(candidate)

			pop = new_pop
			scores = []
			for p in pop:
				s, _ = self.objective_fn(p)
				scores.append(s)
			history.append(max(scores))

		best_idx = max(range(self.population_size), key=lambda i: scores[i])
		return pop[best_idx], scores[best_idx], history