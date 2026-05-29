# metrics/collision_rate.py
# ------------------------------------------------------------
# Collision rate over predicted trajectories.
# 输入:
#   per_video_pred: dict[video] -> Tensor [N, M, 2] (同窗对齐的 N个主体)
#   per_video_WH:   dict[video] -> (W,H)
# 阈值:
#   thr: 以像素为单位；若提供 norm=True 则用 min(W,H)*ratio。
# 输出:
#   每个视频一个 CR，再对视频平均。
# ------------------------------------------------------------
from __future__ import annotations
from typing import Dict, Tuple
import torch

def _pairwise_dist(a: torch.Tensor) -> torch.Tensor:
    # a: [N,2] -> [N,N]
    diff = a[:,None,:] - a[None,:,:]
    return torch.linalg.vector_norm(diff, dim=-1)

def collision_rate(per_video_pred: Dict[str, torch.Tensor],
                   per_video_WH: Dict[str, Tuple[int,int]],
                   thr: float = 12.0, norm: bool = False, ratio: float = 0.01) -> float:
    rates = []
    for v, traj in per_video_pred.items():
        # traj: [N,M,2]
        if traj.ndim != 3 or traj.size(0) < 2:
            continue
        W,H = per_video_WH.get(v, (1920,1080))
        T = thr if not norm else min(W,H) * ratio
        N,M,_ = traj.shape
        col = 0; tot = 0
        for t in range(M):
            D = _pairwise_dist(traj[:,t])  # [N,N]
            mask = torch.triu(torch.ones_like(D, dtype=torch.bool), diagonal=1)
            d = D[mask]
            col += (d < T).sum().item()
            tot += d.numel()
        if tot>0:
            rates.append(col/tot)
    return float(sum(rates)/len(rates)) if rates else 0.0
