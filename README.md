# Probabilistic Bird Trajectory Forecasting

Research code for **Probabilistic Bird Trajectory Forecasting with Heavy-Tailed Uncertainty Modeling for Low-Altitude Airspace Monitoring**, by Feiyang Song, Zhonghe Liu, Yuyang Zhao, and Jingguo Zhu.

**[Published paper — Sensors 26(4), 1270, 15 February 2026](https://doi.org/10.3390/s26041270)** · [Data format](DATA.md) · [Reproduction guide](REPRODUCE.md) · [Code review](docs/VALIDATION.md)

## Overview

The research investigates bird trajectory prediction with a compact Transformer and a Student-t mixture-density output. Heavy-tailed predictive distributions represent multiple plausible futures and large motion deviations. The repository also preserves detection/tracking, UAV-awareness, baseline and figure-generation experiments.

The runnable reference workflow uses the existing Transformer encoder-decoder and Student-t head. It trains on the bundled trajectory export, holds out source groups, saves the best validation checkpoint, and reports measured test metrics. It is a practical way to inspect the modeling code; it does **not** reconstruct the paper's complete experiment.

## Quick start

Use Python 3.10 or newer, from the repository root. A CPU is sufficient.

```bash
git clone https://github.com/songfy0118/birds_detection.git
cd birds_detection
python -m pip install -r requirements-core.txt
python -m unittest discover -s tests -v
python run_final_experiment.py --smoke --out output/smoke
```

The smoke check trains a small model for one epoch on up to 16 trajectories per partition. It writes:

- `output/smoke/best.pt`: weights, model settings, split indices and data fingerprint.
- `output/smoke/metrics.json`: measured NLL, ADE, FDE and minADE, with metric definitions and run scope.

To use all bundled trajectories, then evaluate the saved model again:

```bash
python forecast.py --epochs 5 --out output/reference
python forecast.py --checkpoint output/reference/best.pt --out output/reevaluation
```

CUDA is optional: append `--device cuda`. A changed dataset or incompatible checkpoint raises an error rather than falling back to random weights.

## What is included

| Component | Entry point | Status |
|---|---|---|
| Training and held-out evaluation | [forecast.py](forecast.py) | Supported reference workflow |
| Transformer forecaster | [models/transformer.py](models/transformer.py) | Existing encoder-decoder with future queries |
| Heavy-tailed mixture head and losses | [models/heads/student_t.py](models/heads/student_t.py) | Likelihood checked against PyTorch distributions |
| Data parsing and normalization | [utils/data.py](utils/data.py) | Bundled and pixel-coordinate JSONL schemas |
| Trajectories | [data/bird_trajs.jsonl](data/bird_trajs.jsonl) | 2,197 windows, 44 track IDs, 8 observed / 12 future steps |
| Measured inference timing | [measure_speed.py](measure_speed.py) | Requires a reference-workflow checkpoint |
| Historical experiments | `train/`, `predict/`, `tracking/`, `uav/`, `baselines/`, `figure_scripts/` | Research archive; see reproduction guide |
| Manuscript and figures | `paper/`, `figures/` | Research artifacts |

## Reading the results

ADE/FDE use the mixture-weighted mean prediction. minADE selects the lowest-error component trajectory using the target; it is an oracle metric, not an online prediction rule. NLL evaluates the full mixture distribution per two-dimensional step.

Distances in the reference report use image coordinates normalized to `[-1, 1]`, **not metres**. The bundled data lacks camera calibration and original source-video identifiers. Its track-based partition prevents the same track ID crossing partitions, but cannot establish cross-video independence.

The paper's reported accuracy, calibration and speed belong to its experimental protocol. The bundled 8/12-step export, new reference split and full-mixture training loss do not reproduce that protocol or the 60-step research checkpoints. See [REPRODUCE.md](REPRODUCE.md) for the exact boundary.

## Citation

```bibtex
@article{song2026probabilistic,
  title = {Probabilistic Bird Trajectory Forecasting with Heavy-Tailed Uncertainty Modeling for Low-Altitude Airspace Monitoring},
  author = {Song, Feiyang and Liu, Zhonghe and Zhao, Yuyang and Zhu, Jingguo},
  journal = {Sensors},
  year = {2026},
  volume = {26},
  number = {4},
  pages = {1270},
  doi = {10.3390/s26041270}
}
```

## Licenses

Third-party code and weights retain their upstream licenses. The paper's publication license does not automatically license every code or data artifact in this repository; no new blanket license is asserted here.
