# Probabilistic Bird Trajectory Forecasting

Code, figures, and manuscript for the paper:

> **Probabilistic Bird Trajectory Forecasting with Heavy-Tailed Uncertainty
> Modeling for Low-Altitude Airspace Monitoring**
> Feiyang Song, Zhonghe Liu, Yuyang Zhao, Jingguo Zhu.

A vision-based framework for monitoring shared low-altitude airspace. Its core,
**Mini-BirdFormer**, couples a lightweight Transformer encoder with a
**Student-t Mixture Density Network (MDN)** head to forecast bird-flock
trajectories *with calibrated, heavy-tailed uncertainty* — using only ~1.05 M
parameters (minADE 0.785 m, NLL 1.25 → −2.01 vs. a Gaussian-LSTM baseline,
616 FPS). A plug-and-play **UAV awareness module** adds zero-shot drone detection
via an open-vocabulary model (OWL-ViT) and a synthetic-data pipeline (92% recall,
no false alarms).

> **Note.** This repo holds the code + small derived data + figures + paper. The
> raw videos, full datasets, and heavy pretrained weights (~30+ GB) are **not** on
> GitHub — see [`DATA.md`](DATA.md) for what they are and how to obtain them.

---

## Get the code / 获取代码

```bash
git clone https://github.com/songfy0118/birds_detection.git
cd birds_detection
pip install -r requirements.txt
```

(Or download the ZIP from the green **Code** button on GitHub.)

---

## Repository structure / 仓库结构

```
birds_detection/
├── models/              # network definitions
│   ├── transformer.py       # Mini-BirdFormer lightweight Transformer encoder  ← core
│   ├── heads/student_t.py   # Student-t mixture-density head (heavy-tailed)     ← core contribution
│   ├── heads/gaussian.py    # Gaussian MDN head (baseline)
│   ├── heads/intent_head.py # auxiliary intent head
│   ├── lstm.py              # Gaussian-LSTM baseline
│   └── build_variants.py    # assembles encoder + head into model variants
├── train/               # training loops (train_transformer.py, train_lstm.py, …)
├── predict/             # inference (predict.py, predict_online.py)
├── metrics/             # ADE/FDE, NLL/uncertainty, collision rate, HOTA, …
├── utils/               # data / geometry / visualization helpers
├── tracking/            # YOLOv8 detection → tracking → tracklet sequences
│   ├── run_tracker.py       # run the detector+tracker on video
│   ├── mot_to_sequences.py  # convert MOT tracks → trajectory sequences
│   └── stabilize.py         # camera-motion stabilization
├── uav/                 # UAV awareness module (zero-shot detection + synthesis)
│   ├── presence_owlvit_*.py # OWL-ViT open-vocabulary drone detection
│   ├── presence_gdino_*.py  # GroundingDINO variant
│   ├── presence_yolo_*.py   # YOLO comparison
│   └── synthesize_uav.py / batch_synthesize_uav*.py  # synthetic UAV injection
├── generator/           # synthetic trajectory / scene generators
├── bird_forecast/       # export tracklets → HiVT baseline format
├── hivt_data/           # small pre-split .npz tensors for the HiVT baseline
│
├── run_final_experiment.py   # ★ main paper experiment (train + eval + compare)
├── run_all_baselines.py      # run every baseline end-to-end
├── main.py                   # end-to-end pipeline entry
├── eval_noise_final.py       # ★ robustness-to-input-noise study (Fig. 4)
├── eval_noise.py / eval_noise_direct.py   # earlier noise-eval variants
├── measure_efficiency.py / measure_speed.py / benchmark_speed.py  # FPS / params (Fig. 5)
│
├── figure_scripts/      # scripts that render the paper figures
│   ├── plot_fig1_teaser.py      # Fig. 1 system overview
│   ├── plot_fig2_arch.py        # Fig. 2 Mini-BirdFormer architecture
│   ├── plot_fig3_uav.py         # Fig. 3 UAV detection / synthesis
│   ├── plot_fig4_robustness.py  # Fig. 4 robustness to noise
│   ├── plot_fig5_efficiency.py  # Fig. 5 efficiency
│   └── generate_*.py / create_*.py / picture_introduction.py
├── figures/             # rendered figure outputs used in the manuscript
│
├── baselines/           # third-party baselines compared in the paper
│   ├── Social-STGCNN-master/
│   ├── Trajectron-plus-plus-master/
│   ├── sgan-master/            # Social-GAN
│   └── stgcnn_baseline/
│
├── data/bird_trajs.jsonl     # derived bird tracklets (input to forecasting)
├── checkpoints/              # lstm_forecast.pt, yolov8n.pt
│
├── paper/
│   ├── Probabilistic_Bird_Trajectory_Forecasting.pdf   # latest compiled paper (v12)
│   └── latex/                # LaTeX source (MDPI template) + figures
│
├── requirements.txt
├── DATA.md              # where the 30+ GB of data/weights live
└── .gitignore
```

> **Heads-up on a couple of empty files.** `models/studentT.py`,
> `models/dual_transformer.py`, `train/train_studentT.py`, and
> `train/train_dual_transformer.py` are 0-byte placeholders left over from
> development. The *actual* Student-t model is `models/heads/student_t.py` plus
> the definitions inside `run_final_experiment.py`. Kept for a faithful backup.

---

## Workflow / 工作流程

1. **Detect + track** birds in video → MOT tracks
   (`tracking/run_tracker.py`, uses `checkpoints/yolov8n.pt`).
2. **Build sequences**: MOT tracks → trajectory tracklets
   (`tracking/mot_to_sequences.py`) → `data/bird_trajs.jsonl`.
3. **Train forecasting** models (`run_final_experiment.py`, or `train/train_*.py`):
   Mini-BirdFormer (Student-t) + Gaussian-LSTM / Transformer baselines.
4. **Evaluate**: `metrics/` (minADE/FDE, NLL, collision), robustness via
   `eval_noise_final.py`, efficiency via `measure_efficiency.py`.
5. **UAV awareness** (independent module, not part of forecasting training):
   `uav/presence_owlvit_*.py` + `uav/synthesize_uav.py`.
6. **Figures**: scripts in `figure_scripts/` regenerate the paper figures.

Because step 2's output (`bird_trajs.jsonl`) is included, you can run the
forecasting experiments (steps 3–4 + figures) **without** the raw video or the
large datasets.

---

## Reproducing the headline result / 复现主要结果

```bash
python run_final_experiment.py
```

This trains/loads Mini-BirdFormer and the baselines on `data/bird_trajs.jsonl`
and reports the metrics quoted in the paper (minADE, FDE, NLL, FPS).

## License

See the paper for citation. Third-party code under `baselines/` retains its
original license.
