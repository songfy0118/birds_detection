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
    sW, sH = max(1.0, W/2.0), max(1.0, H/2.0)
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
            for line in f:
                d = json.loads(line)
                # 兼容字段命名差异
                W = d.get("W") or d.get("w") or d.get("W0") or 1920
                H = d.get("H") or d.get("h") or d.get("H0") or 1080
                past   = np.array(d["past"],   dtype=np.float32)  # [P,2]
                future = np.array(d["future"], dtype=np.float32)  # [F,2]
                agent_type = d.get("agent_type","unknown")
                video = d.get("video","")
                tid   = int(d.get("track_id",-1))
                self.items.append({
                    "past": past, "future": future, "W": int(W), "H": int(H),
                    "agent_type": agent_type, "video": video, "track_id": tid
                })

        # 统计 P/F
        self.P = self.items[0]["past"].shape[0]
        self.F = self.items[0]["future"].shape[0]

    def __len__(self): return len(self.items)

    def __getitem__(self, idx: int):
        it = self.items[idx]
        W, H = it["W"], it["H"]
        past, future = it["past"], it["future"]

        past_wh = _rep_wh(W, H, past.shape[0]).astype(np.float32)
        if self.normalize:
            past_n   = _norm_xy(past, W, H)
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
                       pin_memory=True, drop_last=True, collate_fn=traj_collate)
    dl_va = DataLoader(ds_va, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                       pin_memory=True, drop_last=False, collate_fn=traj_collate)
    return ds_tr, ds_va, dl_tr, dl_va
