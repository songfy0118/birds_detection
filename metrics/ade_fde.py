# metrics/ade_fde.py
# ------------------------------------------------------------
# ADE / FDE utilities with robust handling
# pred, gt: [B,M,2] (pixel or normalized)
# ------------------------------------------------------------
from __future__ import annotations
import torch
from typing import Dict, List, Tuple

def ade(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    # [B,M,2] -> scalar
    # 计算所有点误差的平均值
    return torch.linalg.vector_norm(pred - gt, dim=-1).mean()

def fde(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    # 只计算最后一个点的误差
    return torch.linalg.vector_norm(pred[:, -1] - gt[:, -1], dim=-1).mean()

def raw_displacement_errors(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """
    新增函数：返回原始的逐帧误差，不求平均。
    用于绘制 Time Horizon 分析图。
    Returns: [B, M] 张量，代表每个样本在每个时间步的欧氏距离
    """
    # pred: [B, T, 2], gt: [B, T, 2]
    # norm -> [B, T]
    return torch.linalg.vector_norm(pred - gt, dim=-1)

def ade_fde(pred: torch.Tensor, gt: torch.Tensor) -> Dict[str, float]:
    a = float(ade(pred, gt))
    f = float(fde(pred, gt))
    return {"ADE": a, "FDE": f}

def multi_horizon_ade_fde(pred: torch.Tensor,
                          gt: torch.Tensor,
                          horizons: List[int]) -> Dict[str, float]:
    """
    horizons: 以"步"为单位，例如 [10,20,30,60]
    """
    out = {}
    for h in horizons:
        h = max(1, min(int(h), pred.size(1)))
        # 截取前 h 帧计算 ADE
        a = torch.linalg.vector_norm(pred[:, :h] - gt[:, :h], dim=-1).mean()
        # 取第 h-1 帧计算 FDE
        f = torch.linalg.vector_norm(pred[:, h-1] - gt[:, h-1], dim=-1).mean()
        out[f"ADE@{h}"] = float(a)
        out[f"FDE@{h}"] = float(f)
    return out