# LSTM-based IDS with PSO/JAYA/SSA Optimization

This repository implements the paper "An optimized LSTM-based deep learning model for anomaly network intrusion detection" (Scientific Reports, 2025).

It provides an end-to-end, reproducible pipeline:
- Dataset download/prep: NSL-KDD, CICIDS2017, BoT-IoT
- Model: Conv1D → MaxPool → LSTM → Dense (binary)
- Hyperparameter optimization: PSO, JAYA, SSA
- Training and evaluation: Accuracy, Precision, Recall, F1, ROC; convergence plots

## Repository Structure

```
src/
  datasets/
  models/
  optimizers/
  utils/
data/
models/
notebooks/
docs/
results/
```

## Environment

- Python 3.9+
- TensorFlow/Keras 2.15 (CPU or GPU)

Create env via Conda:
```bash
conda env create -f environment.yml
conda activate lstmids
```
Or via pip:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Datasets

- NSL-KDD: auto-downloaded from a public mirror.
- CICIDS2017: requires dataset files (CSV) placed under `data/CICIDS2017/` or available via Kaggle API.
- BoT-IoT: requires dataset files placed under `data/BOTIOT/` or available via Kaggle API.

See `docs/datasets.md` for details on expected file names.

## Quickstart

Train and evaluate on NSL-KDD with PSO optimization:
```bash
python train.py --optimizer PSO --dataset NSL-KDD --epochs 10 --iterations 5 --population 8
```

Train with SSA on CICIDS2017 (requires data present):
```bash
python train.py --optimizer SSA --dataset CICIDS2017 --epochs 10 --iterations 5 --population 8
```

## Reproducibility

- Fixed random seeds across Python, NumPy, and TensorFlow.
- Results, configs, and plots are stored under `results/`.

## Notes

- Optimizers tune: LSTM units, Conv1D filters, learning rate, batch size, dropout.
- For large datasets (CICIDS2017/BoT-IoT), the loader supports reading multiple CSVs and downsampling via CLI flags.

## Citation

If you use this code, please cite the original paper and this implementation.