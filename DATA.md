# Data, Weights & Large Outputs (not stored in this repo)

GitHub rejects files larger than 100 MB and is not meant to hold large
datasets. This repository therefore contains **only the code, the small derived
trajectory data, the small checkpoints, the figures, and the paper**. The raw
videos, full datasets, pretrained heavy weights, and rendered output videos
(~30+ GB in total) are **kept locally** on the original machine.

本仓库只放代码、小体量轨迹数据、小 checkpoint、图、论文。原始视频、完整数据集、
大模型权重、渲染输出视频（总计 30+ GB）不在 GitHub 上，保存在本地原始机器。

## What IS in the repo / 仓库里有的

| Path | Size | Description |
|---|---|---|
| `data/bird_trajs.jsonl` | 3.5 MB | Derived bird tracklets (the actual input to forecasting). 由视频跟踪导出的鸟类轨迹。 |
| `hivt_data/{train,val,test}.npz` | 52 KB | Pre-split trajectory tensors for the HiVT baseline export. |
| `checkpoints/lstm_forecast.pt` | 207 KB | Trained Gaussian-LSTM baseline checkpoint. |
| `checkpoints/yolov8n.pt` | 6.5 MB | YOLOv8-nano detector used by the tracking front-end. |

## What is NOT in the repo (kept locally) / 不在仓库、保存在本地的

Original location: `Desktop/bird_detection/` on the author's machine.

| Item | Size | Original path |
|---|---|---|
| FBD-SV-2024 training videos + labels | ~3.4 GB | `day1/training_data_base2_FBD-SV-2024/`, `day2/FBD_SV_2024_train/` |
| Intermediate dataset (frames/crops) | ~2 GB | `day1/dataset/` |
| Rendered tracking/result videos | ~6.7 GB | `day1/output/`, `day2/output_version1/` (`*_track.mp4`) |
| Packaged final outputs | ~6.8 GB | `day1/aaafinal_output/` (+ `aaafinal_output.zip` 6 GB) |
| GroundingDINO Swin-B weights | 895 MB | `day1/groundingdino_swinb_cogcoor.pth` |
| OWL-ViT base weights | 585 MB | `day1/models/owlvit-base-patch32/` (auto-downloaded by `transformers`) |
| Earlier experiment archive | ~1.7 GB | `day1/archive/` |

## How to regenerate / 如何重建

1. **OWL-ViT weights** download automatically the first time `transformers` loads
   `google/owlvit-base-patch32`.
2. **GroundingDINO weights**: get `groundingdino_swinb_cogcoor.pth` from the
   official GroundingDINO release.
3. **FBD-SV-2024** is a public bird-video dataset; place it where the tracking
   scripts expect it, then run the front-end (see README "Workflow").
4. **`bird_trajs.jsonl`** (already included) is the output of the detection +
   tracking stage, so forecasting can be trained/evaluated without the raw video.
