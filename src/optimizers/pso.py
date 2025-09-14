from typing import Dict, Any, Tuple, List
import random
import math

from .base import HyperparameterOptimizer


class PSOOptimizer(HyperparameterOptimizer):
	def __init__(self, *args, num_particles: int = 10, inertia: float = 0.7, cognitive: float = 1.5, social: float = 1.5, **kwargs):
		super().__init__(*args, **kwargs)
		self.num_particles = num_particles
		self.inertia = inertia
		self.cognitive = cognitive
		self.social = social

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
		# Initialize particles
		particles = [self._random_params() for _ in range(self.num_particles)]
		velocities = [{k: 0.0 for k in self.param_space.keys()} for _ in range(self.num_particles)]
		personal_best_params = list(particles)
		personal_best_scores = []
		history = []

		# Evaluate initial particles
		for p in particles:
			score, _ = self.objective_fn(self.clip_params(p))
			personal_best_scores.append(score)

		global_best_idx = max(range(self.num_particles), key=lambda i: personal_best_scores[i])
		global_best_params = personal_best_params[global_best_idx].copy()
		global_best_score = personal_best_scores[global_best_idx]
		history.append(global_best_score)

		for t in range(self.max_iters):
			for i in range(self.num_particles):
				for k in self.param_space.keys():
					r1, r2 = random.random(), random.random()
					velocities[i][k] = (
						self.inertia * velocities[i][k]
						+ self.cognitive * r1 * (personal_best_params[i][k] - particles[i][k])
						+ self.social * r2 * (global_best_params[k] - particles[i][k])
					)
					particles[i][k] += velocities[i][k]
				particles[i] = self.clip_params(particles[i])

				score, _ = self.objective_fn(particles[i])
				if score > personal_best_scores[i]:
					personal_best_scores[i] = score
					personal_best_params[i] = particles[i].copy()
			if max(personal_best_scores) > global_best_score:
				global_best_idx = max(range(self.num_particles), key=lambda j: personal_best_scores[j])
				global_best_params = personal_best_params[global_best_idx].copy()
				global_best_score = personal_best_scores[global_best_idx]
			history.append(global_best_score)

		return global_best_params, global_best_score, history