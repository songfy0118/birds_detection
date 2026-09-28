# Running and interpreting the reference experiment

## Installation

Python 3.10+ and `requirements-core.txt` are sufficient for forecasting and tests. `requirements.txt` lists the broader historical detection, plotting and UAV dependencies; they are not needed for the quick start.

Commands assume the current directory is the repository root.

```bash
python -m pip install -r requirements-core.txt
python -m unittest discover -s tests -v
python forecast.py --smoke --out output/smoke
python forecast.py --epochs 5 --batch-size 32 --seed 42 --out output/reference
python forecast.py --checkpoint output/reference/best.pt --out output/reevaluation
python measure_speed.py --checkpoint output/reference/best.pt --iterations 100
```

Training uses AdamW, learning rate 0.0003, weight decay 0.02 and full Student-t mixture NLL. Validation NLL selects the saved checkpoint. The test partition is evaluated only after selection. The default model has width 96, two encoder layers, two decoder layers and five mixture components. Smoke mode uses a smaller model and subset.

The seed fixes initial weights and partition assignment. Exact numerical reproducibility across PyTorch versions, accelerators or hardware is not guaranteed.

## Outputs

`metrics.json` records scope, model settings, parameter count, dataset fingerprint, partition sizes, seed, selected epoch, validation history and final test metrics. Re-evaluation has an empty training history and reuses the checkpoint's exact test partition.

The checkpoint stores row indices tied to the exact file fingerprint. Reordering a JSONL file therefore requires a new training run.

Timing uses a synthetic input and a trained reference checkpoint with explicit warmup, evaluation mode and CUDA synchronization. It measures model forward passes only, not detection, tracking, I/O or video FPS.

## Reference run versus paper reproduction

| Requirement | Public reference workflow |
|---|---|
| Existing Transformer and Student-t implementation | Included |
| Bundled data reader, training, checkpoint, held-out metrics | Included and tested |
| Original 60-step trajectories and exact paper partitions | Not supplied by the bundled 8/12-step export |
| Original WTA/auxiliary-loss schedule and calibration procedure | Not reconstructed by the full-mixture-NLL reference trainer |
| Camera-to-metre conversion | Not available in the bundled export |
| Published baseline and UAV experiment reproduction | Not established by this workflow |
| Paper speed/accuracy results | Consult the published paper; do not substitute smoke metrics |

## Historical code

The `train/`, `predict/`, `tracking/`, `uav/`, `generator/`, `baselines/` and figure directories preserve research development. Some scripts require external datasets, local paths, optional packages or their own checkpoint conventions. They are not collectively validated by the reference tests.

The previous `run_final_experiment.py` mixed illustrative inputs, fixed plotting values and fallback random weights. It is retained in Git history; the current file delegates to measured training/evaluation. Four empty model/training placeholders were removed; the real Student-t implementation is `models/heads/student_t.py`. No dual-transformer implementation is claimed.

The older `train/train_transformer.py` now disables its redundant model-side normalization. Use `forecast.py` for the supported end-to-end path.
