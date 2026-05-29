# utils/geometry.py
# ------------------------------------------------------------
# Normalization <-> Pixel, simple camera-motion token embedder,
# Dot distance (center distance normalized by frame size)
# ------------------------------------------------------------
from __future__ import annotations
import torch
import math

def to_norm(xy: torch.Tensor, W: torch.Tensor, H: torch.Tensor) -> torch.Tensor:
    # xy: [B,L,2]; W/H: [B] or [B,1]
    sW = (W[...,None] / 2.0).clamp_min(1.0)
    sH = (H[...,None] / 2.0).clamp_min(1.0)
    out = xy.clone()
    out[...,0] = (xy[...,0] - sW) / sW
    out[...,1] = (xy[...,1] - sH) / sH
    return out

def to_pixel(xy_n: torch.Tensor, W: torch.Tensor, H: torch.Tensor) -> torch.Tensor:
    sW = (W[...,None] / 2.0).clamp_min(1.0)
    sH = (H[...,None] / 2.0).clamp_min(1.0)
    out = xy_n.clone()
    out[...,0] = xy_n[...,0] * sW + sW
    out[...,1] = xy_n[...,1] * sH + sH
    return out

def dot_distance(p: torch.Tensor, q: torch.Tensor, W: torch.Tensor, H: torch.Tensor) -> torch.Tensor:
    # p,q:[B,L,2]; 返回归一化中心距离（L2），对小目标更稳健
    sW = W[...,None].clamp_min(1.0)
    sH = H[...,None].clamp_min(1.0)
    dx = (p[...,0]-q[...,0]) / sW
    dy = (p[...,1]-q[...,1]) / sH
    return torch.sqrt(dx*dx + dy*dy)  # [B,L]

class CameraToken(nn.Module):
    """
    把一段相机运动参数序列（如 dx,dy,da）编码成 d_model 维 token。
    接口：forward(motion_seq) -> [B,d_model]
    """
    def __init__(self, in_dim=3, d_model=256, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, d_model)
        )
    def forward(self, motion_seq: torch.Tensor):
        # motion_seq: [B,T,3] (dx,dy,da)；简单取均值后映射
        x = motion_seq.mean(dim=1)  # [B,3]
        return self.net(x)
