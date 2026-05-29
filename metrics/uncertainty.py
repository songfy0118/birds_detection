# metrics/uncertainty.py
# ------------------------------------------------------------
# Monte-Carlo uncertainty coverage (UC@q) for 2D trajectories
# Supports: Gaussian / Student-t / MDN-(Gaussian|Student-t)
# ------------------------------------------------------------
from __future__ import annotations
import torch
import torch.distributions as D

def _ensure_device(t, device):
    return t if (t is not None and t.device == device) else (t.to(device) if t is not None else None)

def sample_gaussian(mu, log_scale, num_samples: int):
    # mu: [B,M,2], log_scale: [B,M,2] (diag)
    B, M, _ = mu.shape
    std = log_scale.exp()
    eps = torch.randn(num_samples, B, M, 2, device=mu.device)
    return mu[None] + eps * std[None]  # [S,B,M,2]

def sample_student_t(mu, log_scale, dof, num_samples: int):
    # mu: [B,M,2], log_scale: [B,M,2], dof: [B,M] or scalar
    B, M, _ = mu.shape
    nu = dof if dof.ndim > 0 else torch.full((B,M), float(dof), device=mu.device)
    std = log_scale.exp()
    # Student-t: x = mu + z * std * sqrt(nu / v), z~N(0,1), v~ChiSq(nu)
    z = torch.randn(num_samples, B, M, 2, device=mu.device)
    v = torch.distributions.Chi2(nu).sample((num_samples,)).to(mu.device)  # [S,B,M]
    scale = torch.sqrt(nu / v).unsqueeze(-1)  # [S,B,M,1]
    return mu[None] + z * std[None] * scale  # [S,B,M,2]

def _cat_sample(logits):
    # logits: [...,K] -> indices [...,]
    probs = logits.softmax(dim=-1)
    flat = probs.reshape(-1, probs.size(-1))
    idx = torch.multinomial(flat, 1).squeeze(-1)
    return idx.view(*probs.shape[:-1])

def sample_mdn_gauss(logits, mu, log_scale, num_samples: int):
    # logits: [B,M,K], mu: [B,M,K,2], log_scale: [B,M,K,2]
    B,M,K,_ = mu.shape
    S = num_samples
    # repeat mixture along samples, draw a comp for each (S,B,M)
    comp = _cat_sample(logits.expand(S,-1,-1,-1))  # [S,B,M]
    # gather chosen component
    idx = comp.unsqueeze(-1).unsqueeze(-1).expand(-1,-1,-1,1,2)  # [S,B,M,1,2]
    mu_sel  = mu[None].expand(S,-1,-1,-1,-1).gather(3, idx).squeeze(3)        # [S,B,M,2]
    ls_sel  = log_scale[None].expand(S,-1,-1,-1,-1).gather(3, idx).squeeze(3) # [S,B,M,2]
    eps = torch.randn(S,B,M,2, device=mu.device)
    return mu_sel + eps * ls_sel.exp()

def sample_mdn_student_t(logits, mu, log_scale, dof, num_samples: int):
    # logits: [B,M,K], mu/log_scale: [B,M,K,2], dof: [B,M,K] or scalar
    B,M,K,_ = mu.shape
    S = num_samples
    comp = _cat_sample(logits.expand(S,-1,-1,-1))  # [S,B,M]
    idx = comp.unsqueeze(-1).unsqueeze(-1).expand(-1,-1,-1,1,2)
    mu_sel = mu[None].expand(S,-1,-1,-1,-1).gather(3, idx).squeeze(3)         # [S,B,M,2]
    ls_sel = log_scale[None].expand(S,-1,-1,-1,-1).gather(3, idx).squeeze(3)  # [S,B,M,2]
    if isinstance(dof, torch.Tensor) and dof.ndim==3:  # [B,M,K] -> select
        dof_idx = comp.unsqueeze(-1)  # [S,B,M,1]
        dof_sel = dof[None].expand(S,-1,-1,-1).gather(3, dof_idx).squeeze(3)  # [S,B,M]
    else:
        dof_sel = torch.full((S,B,M), float(dof), device=mu.device)
    z = torch.randn(S,B,M,2, device=mu.device)
    v = torch.distributions.Chi2(dof_sel).sample().to(mu.device)  # [S,B,M]
    scale = torch.sqrt(dof_sel / v).unsqueeze(-1)
    return mu_sel + z * ls_sel.exp() * scale  # [S,B,M,2]

def uc_coverage_at_q(gt_future_xy, mu, samples, q: float = 0.95):
    """
    gt_future_xy: [B,M,2] (normalized/pixel consistent with model output)
    mu:           [B,M,2]  (mean used for distance baseline)
    samples:      [S,B,M,2]
    Return: coverage (scalar), per-step boolean mask [B,M]
    """
    # distances of samples to mean
    dists = torch.linalg.norm(samples - mu[None], dim=-1)  # [S,B,M]
    thr = torch.quantile(dists, q, dim=0)                  # [B,M]
    gt_d = torch.linalg.norm(gt_future_xy - mu, dim=-1)    # [B,M]
    covered = (gt_d <= thr)                                # [B,M]
    return covered.float().mean().item(), covered

def compute_uc(args, head_type, out_dict, gt_future_xy, num_samples=200, q=0.95):
    """
    out_dict: model forward outputs (keys depend on head_type)
    Returns: uc95(float)
    """
    with torch.no_grad():
        if head_type == "gauss":
            S = sample_gaussian(out_dict["mu"], out_dict.get("log_scale", out_dict.get("log_sigma")), num_samples)
            cov,_ = uc_coverage_at_q(gt_future_xy, out_dict["mu"], S, q)
            return cov
        elif head_type == "student_t":
            S = sample_student_t(out_dict["mu"], out_dict.get("log_scale", out_dict.get("log_sigma")), out_dict["dof"], num_samples)
            cov,_ = uc_coverage_at_q(gt_future_xy, out_dict["mu"], S, q)
            return cov
        elif head_type == "mdn_gauss":
            S = sample_mdn_gauss(out_dict["logits"], out_dict["mu"], out_dict.get("log_scale", out_dict.get("log_sigma")), num_samples)
            # use mixture mean as baseline
            mix_p = out_dict["logits"].softmax(-1)  # [B,M,K]
            mix_mu = (mix_p.unsqueeze(-1) * out_dict["mu"]).sum(dim=2)  # [B,M,2]
            cov,_ = uc_coverage_at_q(gt_future_xy, mix_mu, S, q)
            return cov
        elif head_type == "mdn_student_t":
            S = sample_mdn_student_t(out_dict["logits"], out_dict["mu"], out_dict.get("log_scale", out_dict.get("log_sigma")), out_dict["dof"], num_samples)
            mix_p = out_dict["logits"].softmax(-1)
            mix_mu = (mix_p.unsqueeze(-1) * out_dict["mu"]).sum(dim=2)
            cov,_ = uc_coverage_at_q(gt_future_xy, mix_mu, S, q)
            return cov
        else:
            raise ValueError(f"Unknown head type: {head_type}")
