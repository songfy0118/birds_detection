# Validation record — 28 September 2026

This record covers the public reference workflow added during portfolio maintenance, not a reproduction of the published benchmark.

## Pass 1: source and scientific scope

- Checked the bundled JSONL against the existing dataset reader: the export uses `hist/fut`, whereas the reader expected `past/future`.
- Identified fixed-value plots and random-weight fallback in the former advertised experiment entry.
- Replaced that entry with measured training/evaluation, documented archive boundaries, and removed four empty placeholders.
- Kept paper results separate from newly measured reference metrics.

## Pass 2: execution and regression checks

Environment: Windows, Python 3.14.3, PyTorch 2.13.0+cpu. No new model or raw dataset downloads were required.

```bash
python -m unittest discover -s tests -v
python run_final_experiment.py --smoke --out output/portfolio_smoke
python forecast.py --epochs 2 --out output/portfolio_reference
python forecast.py --checkpoint output/portfolio_reference/best.pt --out output/portfolio_reload
python measure_speed.py --checkpoint output/portfolio_reference/best.pt --iterations 20
```

All **6 tests passed**. They cover both coordinate schemas, invalid records, video/track grouping, partition overlap rejection, the likelihood against `torch.distributions.StudentT`, and checkpoint round-trip/data-mismatch behavior.

The all-data run used 1,561 training, 421 validation and 215 test windows. Validation selected epoch 2. Reloading reproduced the same test metrics:

| Reference metric | Measured value |
|---|---:|
| Full-mixture NLL | -2.425481 |
| Mixture-mean ADE | 0.076969 |
| Mixture-mean FDE | 0.100976 |
| Best-component minADE | 0.065843 |

Distance units are normalized image coordinates, not metres. These values are from a new track-grouped reference experiment, not the publication. The forecaster had 1,041,498 parameters. The timing command completed; its hardware-specific forward-pass throughput is not reported as video FPS or the paper's efficiency result.

## Pass 3: consistency and release review

- Reviewed dataset dimensions, model normalization, coordinate units, metric definitions and checkpoint metadata.
- Added defensive validation of loaded partitions and checked strict parameter loading.
- Checked documented local links and CLI arguments, removed stale entry-point claims, and checked the Git diff for whitespace errors.
- Kept generated weights/results and local research material outside the published change.

## Remaining boundaries

Original paper split manifests, full 60-step data/calibration and complete baseline/UAV reproduction are not supplied by the supported quick start. Legacy research scripts are retained as an archive and are not covered by the six reference tests. Cross-machine numerical identity is not guaranteed.
