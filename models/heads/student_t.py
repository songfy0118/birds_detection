# models/heads/student_t.py
# ------------------------------------------------------------
# 2D Student-t (single + MDN) + WTA
# ------------------------------------------------------------
from __future__ import annotations
import torch
import torch.nn as nn
import math

LOGPI = math.log(math.pi)

def nll_student_t_2d(mu, log_scale, dof, target, reduction="mean"):
    log_scale = torch.clamp(log_scale, -7, 7)
    s = torch.exp(log_scale)
    z = (target - mu) / s

    if not torch.is_tensor(dof):
        dof = torch.tensor(float(dof), device=mu.device, dtype=mu.dtype)
    dof = dof.clamp(min=1.0)

    c0 = torch.log(s*torch.sqrt(dof*math.pi))
    c1 = torch.lgamma(dof/2) - torch.lgamma((dof+1)/2)
    c2 = 0.5*(dof+1)*torch.log1p((z*z)/dof)
    nll = (c0 + c1 + c2).sum(-1)
    return nll.mean() if reduction=="mean" else nll.sum()

def nll_student_t_mixture_2d(logits, mu, log_scale, dof, target, reduction="mean"):
    log_scale = torch.clamp(log_scale, -7, 7)
    if not torch.is_tensor(dof):
        dof = torch.tensor(float(dof), device=mu.device, dtype=mu.dtype)
    dof = dof.clamp(min=1.0)

    B,M,K,_ = mu.shape
    t = target.unsqueeze(2)

    s = torch.exp(log_scale)
    z = (t - mu) / s

    c0 = torch.log(s*torch.sqrt(dof*math.pi))
    c1 = torch.lgamma(dof/2) - torch.lgamma((dof+1)/2)
    c2 = 0.5*(dof+1)*torch.log1p((z*z)/dof)
    log_p = -(c0 + c1 + c2).sum(-1)   # log p(y|k)
    log_mix = torch.log_softmax(logits, -1) + log_p
    nll = -torch.logsumexp(log_mix, dim=-1)
    return nll.mean() if reduction=="mean" else nll.sum()

def wta_nll_student_t_mixture_2d(logits, mu, log_scale, dof, target, reduction="mean"):
    with torch.no_grad():
        t = target.unsqueeze(2)
        ade = torch.linalg.vector_norm(mu - t, dim=-1).mean(1)
        pick = ade.argmin(1)
    B,M,K,_ = mu.shape
    idx = pick.view(B,1,1,1).expand(-1,M,1,2)
    mu_w = mu.gather(2, idx).squeeze(2)
    ls_w = log_scale.gather(2, idx).squeeze(2)
    return nll_student_t_2d(mu_w, ls_w, dof, target, reduction=reduction)

class StudentTHead2D(nn.Module):
    def __init__(self, d_in, hidden=0, dof_init=3.0, learnable_dof=True):
        super().__init__()
        self.learnable_dof = learnable_dof
        if hidden>0:
            self.net = nn.Sequential(nn.Linear(d_in, hidden), nn.ReLU(), nn.Linear(hidden, 4))
        else:
            self.net = nn.Linear(d_in, 4)
        if learnable_dof:
            self.log_dof = nn.Parameter(torch.log(torch.tensor(dof_init)))
        else:
            self.register_buffer("log_dof", torch.log(torch.tensor(dof_init)))

    def forward(self, x):
        out = self.net(x)
        mu = out[..., :2]
        log_sc = out[..., 2:4]
        dof = torch.exp(self.log_dof) + 1.0
        return {"mu": mu, "log_scale": log_sc, "dof": dof}

class MDNStudentTHead2D(nn.Module):
    def __init__(self, d_in, K=3, hidden=0, dof_init=3.0, learnable_dof=True):
        super().__init__()
        self.K = K
        out_dim = K + 4*K
        if hidden>0:
            self.net = nn.Sequential(nn.Linear(d_in, hidden), nn.ReLU(), nn.Linear(hidden, out_dim))
        else:
            self.net = nn.Linear(d_in, out_dim)
        if learnable_dof:
            self.log_dof = nn.Parameter(torch.log(torch.tensor(dof_init)))
        else:
            self.register_buffer("log_dof", torch.log(torch.tensor(dof_init)))

    def forward(self, x):
        B,M,D = x.shape
        out = self.net(x)
        logits = out[..., :self.K]
        rest   = out[..., self.K:].view(B,M,self.K,4)
        mu = rest[..., :2]
        log_sc = rest[..., 2:4]
        dof = torch.exp(self.log_dof) + 1.0
        return {"logits": logits, "mu": mu, "log_scale": log_sc, "dof": dof}
