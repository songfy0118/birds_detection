# utils/data.py
# ------------------------------------------------------------
# IO helpers + Trajectory Dataset (jsonl) + Dataloader builder
# - 读取 tracking/mot_to_sequences.py 生成的 jsonl（含 past/future/W/H/...）
# - 统一归一化到 [-1,1]（按 W/H），同时保留像素坐标（可选）
# - 返回：past_xy [B,P,2]、future_xy [B,F,2]、past_wh [B,P,2]、meta
# ------------------------------------------------------------
from __future__ import annotations
from pathlib import Path
import json, math, random
from typing import List, Dict, Tuple, Optional
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# ------------------- small utils -------------------
def ensure_dir(p: str | Path):
    Path(p).mkdir(parents=True, exist_ok=True)

def _norm_xy(xy: np.ndarray, W: float, H: float) -> np.ndarray:
    # [-1,1] 归一化： (x - W/2)/(W/2) ; (y - H/2)/(H/2)
    sW, sH = W/2.0, H/2.0
    out = xy.copy()
    out[...,0] = (xy[...,0] - W/2.0) / sW
    out[...,1] = (xy[...,1] - H/2.0) / sH
    return out

def _rep_wh(W: int, H: int, L: int) -> np.ndarray:
    return np.tile(np.array([[W, H]], dtype=np.float32), (L,1))

# ------------------- dataset -------------------
class TrajJsonlDataset(Dataset):
    def __init__(self, jsonl_path: str | Path, normalize=True, return_pixels=False):
        self.path = Path(jsonl_path)
        self.normalize = normalize
        self.return_pixels = return_pixels
        self.items: List[dict] = []
        with open(self.path, "r", encoding="utf-8") as f:
            for line_number, line in enumerate(f, 1):
                if not line.strip():
                    continue
                try:
                    d = json.loads(line)
                    if not isinstance(d, dict):
                        raise ValueError("each record must be a JSON object")
                    unit_image = "hist" in d and "fut" in d
                    if unit_image:
                        hist = np.asarray(d["hist"], dtype=np.float32)
                        fut = np.asarray(d["fut"], dtype=np.float32)
                        if hist.ndim != 2 or fut.ndim != 2 or hist.shape[1] != 5 or fut.shape[1] != 5:
                            raise ValueError("hist/fut must have rows [frame, x, y, width, height]")
                        if self.return_pixels:
                            raise ValueError("hist/fut has unit-image coordinates, not calibrated pixels")
                        past, future = hist[:, 1:3].copy(), fut[:, 1:3].copy()
                        W = H = 1
                    else:
                        W = d.get("W", d.get("w", d.get("W0")))
                        H = d.get("H", d.get("h", d.get("H0")))
                        if W is None or H is None or not np.isfinite([W, H]).all() or min(W, H) <= 0:
                            raise ValueError("pixel-coordinate records require positive W and H")
                        past = np.asarray(d["past"], dtype=np.float32)
                        future = np.asarray(d["future"], dtype=np.float32)
                    for name, points in (("past", past), ("future", future)):
                        if points.ndim != 2 or points.shape[1] != 2 or len(points) == 0 or not np.isfinite(points).all():
                            raise ValueError(f"{name} must be a non-empty finite [length, 2] array")
                    if self.items and (len(past), len(future)) != (len(self.items[0]["past"]), len(self.items[0]["future"])):
                        raise ValueError("all records must have the same past/future lengths")
                    tid = int(d.get("track_id", d.get("tid", -1)))
                    video = d.get("video", "")
                    if not isinstance(video, str):
                        raise ValueError("video must be a string")
                except (ValueError, TypeError, KeyError) as exc:
                    raise ValueError(f"{self.path}:{line_number}: {exc}") from exc
                agent_type = d.get("agent_type","unknown")
                self.items.append({
                    "past": past, "future": future, "W": float(W), "H": float(H),
                    "agent_type": agent_type, "video": video, "track_id": tid,
                    "unit_image": unit_image,
                })

        # 统计 P/F
        if not self.items:
            raise ValueError(f"{self.path}: no trajectory records")
        self.P = self.items[0]["past"].shape[0]
        self.F = self.items[0]["future"].shape[0]

    def __len__(self): return len(self.items)

    def __getitem__(self, idx: int):
        it = self.items[idx]
        W, H = it["W"], it["H"]
        past, future = it["past"], it["future"]

        past_wh = _rep_wh(W, H, past.shape[0]).astype(np.float32)
        if self.normalize:
            if it["unit_image"]:
                past_n, future_n = 2 * past - 1, 2 * future - 1
            else:
                past_n = _norm_xy(past, W, H)
                future_n = _norm_xy(future, W, H)
        else:
            past_n, future_n = past, future

        sample = {
            "past_xy":   torch.from_numpy(past_n),      # [P,2]
            "future_xy": torch.from_numpy(future_n),    # [F,2]
            "past_wh":   torch.from_numpy(past_wh),     # [P,2]
            "W": W, "H": H,
            "agent_type": it["agent_type"], "video": it["video"], "track_id": it["track_id"]
        }
        if self.return_pixels:
            sample["past_px"] = torch.from_numpy(past)     # [P,2] 像素坐标
            sample["future_px"] = torch.from_numpy(future) # [F,2]
        return sample

def traj_collate(batch: List[dict]) -> dict:
    keys = ["past_xy","future_xy","past_wh"]
    out = {k: torch.stack([b[k] for b in batch], dim=0) for k in keys}
    out["W"] = torch.tensor([b["W"] for b in batch], dtype=torch.float32)
    out["H"] = torch.tensor([b["H"] for b in batch], dtype=torch.float32)
    out["meta"] = [{"video":b["video"], "track_id":b["track_id"], "agent_type":b["agent_type"]} for b in batch]
    if "past_px" in batch[0]:
        out["past_px"]   = torch.stack([b["past_px"]   for b in batch], 0)
        out["future_px"] = torch.stack([b["future_px"] for b in batch], 0)
    return out

def build_loaders(train_jsonl: str|Path,
                  val_jsonl:   str|Path,
                  batch_size=256, num_workers=4,
                  normalize=True, return_pixels=False):
    ds_tr = TrajJsonlDataset(train_jsonl, normalize=normalize, return_pixels=return_pixels)
    ds_va = TrajJsonlDataset(val_jsonl,   normalize=normalize, return_pixels=return_pixels)
    dl_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True,  num_workers=num_workers,
                       pin_memory=True, drop_last=False, collate_fn=traj_collate)
    dl_va = DataLoader(ds_va, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                       pin_memory=True, drop_last=False, collate_fn=traj_collate)
    return ds_tr, ds_va, dl_tr, dl_va
