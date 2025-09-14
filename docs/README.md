# Documentation

## Datasets

- NSL-KDD: Auto-downloaded on first run to `data/NSL-KDD/` from a public GitHub mirror.
- CICIDS2017: Obtain from the Canadian Institute for Cybersecurity. Place a consolidated CSV as `data/CICIDS2017/CICIDS2017_consolidated.csv` or adjust paths when calling the loader.
- BoT-IoT: Obtain from UNSW Canberra. Place a consolidated CSV as `data/BoT-IoT/BoT_IoT_consolidated.csv`.

Consolidation scripts can be added later if needed; for large datasets, we avoid auto-download to respect license and bandwidth.

## Reproducing Experiments

Examples:

```bash
python train.py --dataset NSL-KDD --optimizer NONE --epochs 10
python train.py --dataset NSL-KDD --optimizer PSO --population 12 --max_iters 8 --epochs 5
python train.py --dataset CICIDS2017 --optimizer SSA --population 20 --max_iters 10 --epochs 10
```

Results are saved under `results/` including `results.json` and `roc_curve.png`.