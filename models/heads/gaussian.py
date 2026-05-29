# models/heads/gaussian.py
# ------------------------------------------------------------
# 2D Diagonal Gaussian + MDN-Gaussian Heads
# ------------------------------------------------------------
from __future__ import annotations
import torch
import torch.nn as nn
import math

LOG2PI = math.log(2.0 * math.pi)

# ------------ NLL ----------------

def nll_gauss_2d(mu, log_std, target, reduction="mean"):
    log_std = torch.clamp(log_std, min=-7, max=7)
    z = (target - mu) * torch.exp(-log_std)
    nll = 0.5*(z*z).sum(-1) + log_std.sum(-1) + LOG2PI
    if reduction=="mean": return nll.mean()
    if reduction=="sum":  return nll.sum()
    return nll

def nll_gauss_mixture_2d(logits, mu, log_std, target, reduction="mean"):
    log_std = torch.clamp(log_std, min=-7, max=7)
    B,M,K,_ = mu.shape
    t = target.unsqueeze(2)      # [B,M,1,2] -> [B,M,K,2]
    z = (t - mu) * torch.exp(-log_std)
    log_comp = -0.5*(z*z).sum(-1) - log_std.sum(-1) - LOG2PI   # [B,M,K]
    log_mix = torch.log_softmax(logits, dim=-1) + log_comp
    nll = -torch.logsumexp(log_mix, dim=-1)      # [B,M]
    return nll.mean() if reduction=="mean" else nll.sum()

def wta_nll_gauss_mixture_2d(logits, mu, log_std, target, reduction="mean"):
    with torch.no_grad():
        t = target.unsqueeze(2)
        ade = torch.linalg.vector_norm(mu - t, dim=-1).mean(1) # [B,K]
        pick = ade.argmin(1)                                   # [B]
    B,M,K,_ = mu.shape
    idx = pick.view(B,1,1,1).expand(-1,M,1,2)
    mu_w = mu.gather(2, idx).squeeze(2)
    ls_w = log_std.gather(2, idx).squeeze(2)
    return nll_gauss_2d(mu_w, ls_w, target, reduction=reduction)

# ------------ Heads ---------------

class GaussianHead2D(nn.Module):
    def __init__(self, d_in, hidden=0):
        super().__init__()
        if hidden>0:
            self.net = nn.Sequential(
                nn.Linear(d_in, hidden), nn.ReLU(True),
                nn.Linear(hidden, 4)
            )
        else:
            self.net = nn.Linear(d_in, 4)

    def forward(self, x):
        out = self.net(x)
        mu = out[..., :2]
        log_std = out[..., 2:]
        return {"mu": mu, "log_std": log_std}

class MDNGaussianHead2D(nn.Module):
    def __init__(self, d_in, K=3, hidden=0):
        super().__init__()
        self.K = K
        out_dim = K + K*4     # logits + K*(mu2 + std2)
        if hidden>0:
            self.net = nn.Sequential(
                nn.Linear(d_in, hidden), nn.ReLU(),
                nn.Linear(hidden, out_dim)
            )
        else:
            self.net = nn.Linear(d_in, out_dim)

    def forward(self, x):
        B,M,D = x.shape
        out = self.net(x)
        logits = out[..., :self.K]
        rest   = out[..., self.K:].view(B,M,self.K,4)
        mu = rest[..., :2]
        log_std = rest[..., 2:4]
        return {"logits": logits, "mu": mu, "log_std": log_std}
