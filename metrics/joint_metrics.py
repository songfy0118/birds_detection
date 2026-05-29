# metrics/joint_metrics.py
# Final stable version

from __future__ import annotations
import torch

def jade_jfde(pred: torch.Tensor, gt: torch.Tensor):
    """
    pred, gt: [N, M, 2]
    单主体预测：JADE = ADE，JFDE = FDE（评估无冲突）
    """
    diff = pred - gt
    ade = diff.norm(dim=-1).mean()
    fde = diff[:, -1].norm(dim=-1).mean()
    return ade.item(), fde.item()
