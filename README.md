# LSTM-based Intrusion Detection System (IDS)

This repository implements the paper "An optimized LSTM-based deep learning model for anomaly network intrusion detection" (Scientific Reports, 2025).

It provides an end-to-end, reproducible pipeline:
- Dataset download and preprocessing (NSL-KDD, CICIDS2017, BoT-IoT)
- Configurable Conv1D + MaxPooling + LSTM + Dense model
- Hyperparameter optimization with PSO, JAYA, and SSA
- Training, evaluation, ROC curves, and convergence plots

## Quickstart

Create a Python 3.9 environment using Conda:

```bash
conda env create -f environment.yml
conda activate lstm-ids
```

Alternatively with pip:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run training (example):

```bash
python train.py --optimizer SSA --dataset CICIDS2017 --epochs 20
```

Results are saved under `results/` with timestamped folders.

## Datasets

- NSL-KDD: Auto-download from a public mirror.
- CICIDS2017: Large; place extracted CSVs under `data/CICIDS2017/` or provide the consolidated CSV.
- BoT-IoT: Large; place CSVs under `data/BoT-IoT/`.

See `docs/README.md` for details, expected file names, and manual download links where automated download is restricted.

## Reproducibility

- Random seeds are fixed across NumPy, Python, and TensorFlow.
- Hardware/software versions are recorded in results metadata.

## Reference

- Scientific Reports (2025). "An optimized LSTM-based deep learning model for anomaly network intrusion detection".