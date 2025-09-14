import os
import argparse
import time
from typing import Dict, Any, Tuple

import numpy as np

from src.utils.seed import set_global_seed
from src.utils.system_info import save_system_info
from src.data.nsl_kdd import load_nsl_kdd
from src.data.cicids2017 import load_cicids2017
from src.data.bot_iot import load_botiot
from src.training.train_eval import train_and_evaluate
from src.training.plots import plot_convergence
from src.optimizers.base import HyperparameterOptimizer
from src.optimizers.pso import PSOOptimizer
from src.optimizers.jaya import JAYAOptimizer
from src.optimizers.ssa import SSAOptimizer


DATASET_LOADERS = {
	"NSL-KDD": load_nsl_kdd,
	"CICIDS2017": load_cicids2017,
	"BoT-IoT": load_botiot,
}


def build_param_space() -> Dict[str, tuple]:
	return {
		"lstm_units": (32, 256),
		"conv_filters": (16, 256),
		"learning_rate": (1e-5, 1e-2),
		"batch_size": (32, 512),
		"dropout": (0.1, 0.6),
	}


def make_objective(X_train, X_test, y_train, y_test, epochs: int, results_dir: str, seed: int):
	def objective(hparams: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
		val_acc, metrics = train_and_evaluate(
			X_train, X_test, y_train, y_test,
			hparams=hparams,
			seed=seed,
			epochs=epochs,
			results_dir=results_dir,
			verbose=0,
		)
		return val_acc, metrics
	return objective


def main(args: argparse.Namespace) -> None:
	set_global_seed(args.seed)

	# Ensure results directory and record system info
	os.makedirs(args.results_dir, exist_ok=True)
	timestamp = time.strftime("%Y%m%d_%H%M%S")
	run_root = os.path.join(args.results_dir, f"exp_{args.dataset}_{args.optimizer}_{timestamp}")
	os.makedirs(run_root, exist_ok=True)
	save_system_info(os.path.join(run_root, "system_info.json"))

	# Load dataset
	if args.dataset == "NSL-KDD":
		X_train, X_test, y_train, y_test = load_nsl_kdd()
	elif args.dataset == "CICIDS2017":
		X_train, X_test, y_train, y_test = load_cicids2017(filename=args.cicids_file)
	elif args.dataset == "BoT-IoT":
		X_train, X_test, y_train, y_test = load_botiot(filename=args.botiot_file)
	else:
		raise ValueError(f"Unsupported dataset {args.dataset}.")

	if args.optimizer == "NONE":
		# direct training with provided hparams
		hparams = {
			"lstm_units": args.lstm_units,
			"conv_filters": args.conv_filters,
			"learning_rate": args.learning_rate,
			"batch_size": args.batch_size,
			"dropout": args.dropout,
		}
		_, metrics = train_and_evaluate(
			X_train, X_test, y_train, y_test,
			hparams=hparams,
			seed=args.seed,
			epochs=args.epochs,
			results_dir=run_root,
			verbose=1,
		)
		print("Metrics:", metrics)
		return

	param_space = build_param_space()
	objective = make_objective(X_train, X_test, y_train, y_test, args.epochs, run_root, args.seed)

	optimizer: HyperparameterOptimizer
	if args.optimizer == "PSO":
		optimizer = PSOOptimizer(objective, param_space, max_iters=args.max_iters, seed=args.seed, num_particles=args.population)
	elif args.optimizer == "JAYA":
		optimizer = JAYAOptimizer(objective, param_space, max_iters=args.max_iters, seed=args.seed, population_size=args.population)
	elif args.optimizer == "SSA":
		optimizer = SSAOptimizer(objective, param_space, max_iters=args.max_iters, seed=args.seed, population_size=args.population)
	else:
		raise ValueError("Unsupported optimizer. Choose from PSO, JAYA, SSA, or NONE.")

	best_params, best_score, history = optimizer.optimize()
	print("Best score:", best_score)
	print("Best params:", best_params)

	# Save optimizer artifacts
	with open(os.path.join(run_root, "best.txt"), 'w') as f:
		f.write(f"best_score: {best_score}\n")
		f.write(f"best_params: {best_params}\n")
	plot_convergence(history, os.path.join(run_root, "convergence.png"))


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Train LSTM-based IDS with optional hyperparameter optimization")
	parser.add_argument("--dataset", type=str, default="NSL-KDD", choices=list(DATASET_LOADERS.keys()))
	parser.add_argument("--optimizer", type=str, default="NONE", choices=["NONE", "PSO", "JAYA", "SSA"])
	parser.add_argument("--epochs", type=int, default=10)
	parser.add_argument("--results_dir", type=str, default="results")
	parser.add_argument("--seed", type=int, default=42)
	parser.add_argument("--population", type=int, default=10, help="Population size for optimizers")
	parser.add_argument("--max_iters", type=int, default=10)
	# Direct hparams
	parser.add_argument("--lstm_units", type=int, default=64)
	parser.add_argument("--conv_filters", type=int, default=64)
	parser.add_argument("--learning_rate", type=float, default=1e-3)
	parser.add_argument("--batch_size", type=int, default=128)
	parser.add_argument("--dropout", type=float, default=0.3)
	# Dataset file options
	parser.add_argument("--cicids_file", type=str, default="CICIDS2017_consolidated.csv")
	parser.add_argument("--botiot_file", type=str, default="BoT_IoT_consolidated.csv")
	args = parser.parse_args()
	main(args)