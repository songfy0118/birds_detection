"""
Mini-BirdFormer Ablation: Gaussian Head vs Student-t Head
==========================================================
用你自己的 Mini-BirdFormer 架构，分别用 Gaussian 和 Student-t head 训练，
对比两者的 minADE / minFDE / NLL，用于 Ablation Table。

用法:
    python run_ablation.py --train_jsonl path/to/train.jsonl \
                           --val_jsonl   path/to/val.jsonl \
                           --test_jsonl  path/to/test.jsonl \
                           --device cuda:0

注意: 这个脚本实现了一个简化版的 Mini-BirdFormer。
如果你想用你 day1 里的完整代码跑 ablation (推荐)，
只需要在你原有的 eval-model 命令里改 --head gauss / --head mdn_student_t 即可:

    # Gaussian ablation (你已有的 trans_gauss.pt 权重)
    python main.py eval-model --jsonl output/jsonl_fbd/test.jsonl \
        --model transformer --head gauss \
        --past_len 8 --future_len 60 \
        --d_model 256 --nhead 8 --layers 4 \
        --weights output/weights/trans_gauss.pt \
        --device cuda:0 --batch_size 256 --horizons 30 60

    # MDN Gaussian ablation
    python main.py eval-model --jsonl output/jsonl_fbd/test.jsonl \
        --model transformer --head mdn_gauss --mdn_k 5 \
        --past_len 8 --future_len 60 \
        --d_model 256 --nhead 8 --layers 4 \
        --weights output/weights/trans_mdn_gauss.pt \
        --device cuda:0 --batch_size 256 --horizons 30 60

    # Student-t (你的主模型)
    python main.py eval-model --jsonl output/jsonl_fbd/test.jsonl \
        --model transformer --head mdn_student_t --mdn_k 5 \
        --past_len 8 --future_len 60 \
        --d_model 256 --nhead 8 --layers 4 \
        --weights output/weights/trans_mdn_student_t.pt \
        --device cuda:0 --batch_size 256 --horizons 30 60

如果你已经有这些权重文件 (trans_gauss.pt, trans_mdn_gauss.pt, trans_mdn_student_t.pt),
那你直接用上面的命令跑 eval 就行了，不需要重新训练！

===========================================================
下面是独立实现（如果你找不到原有权重或想重新跑）:
"""

import argparse
import json
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


# ============================================================
# Data
# ============================================================

class BirdTrajectoryDataset(Dataset):
    def __init__(self, jsonl_path):
        self.samples = []
        with open(jsonl_path) as f:
            for line in f:
                d = json.loads(line.strip())
                past = np.array(d["past"], dtype=np.float32)
                future = np.array(d["future"], dtype=np.float32)
                self.samples.append((past, future))
        print(f"  Loaded {len(self.samples)} from {jsonl_path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        p, f = self.samples[idx]
        return torch.from_numpy(p), torch.from_numpy(f)


# ============================================================
# Mini-BirdFormer (simplified standalone version)
# ============================================================

class MiniBirdFormer(nn.Module):
    """
    Simplified Mini-BirdFormer: 2-layer Transformer encoder + MDN head.
    d_model=96, nhead=4, ff=192 (matching your paper).
    """

    def __init__(self, d_model=96, nhead=4, num_layers=2, ff_dim=192,
                 obs_len=8, pred_len=60, head_type="student_t", mdn_k=5, dropout=0.1):
        super().__init__()
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.head_type = head_type
        self.mdn_k = mdn_k
        self.d_model = d_model

        # Input embedding
        self.input_proj = nn.Linear(2, d_model)

        # Positional encoding
        pe = torch.zeros(obs_len, d_model)
        pos = torch.arange(0, obs_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, T, d)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=ff_dim,
            dropout=dropout, batch_first=True, activation="gelu"
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # MDN head
        if head_type == "gaussian":
            # Single Gaussian: predict mu + sigma per timestep
            # Output: pred_len * 2 (mu_x, mu_y) + pred_len * 2 (log_sigma) = pred_len * 4
            self.head = nn.Sequential(
                nn.Linear(d_model, 192), nn.GELU(),
                nn.Linear(192, 192), nn.GELU(),
                nn.Linear(192, pred_len * 4)  # mu_x, mu_y, log_sx, log_sy
            )
        elif head_type == "mdn_gaussian":
            # Gaussian MDN with K components
            out_per_k = pred_len * 4  # mu_x, mu_y, log_sx, log_sy
            self.head = nn.Sequential(
                nn.Linear(d_model, 192), nn.GELU(),
                nn.Linear(192, 192), nn.GELU(),
                nn.Linear(192, mdn_k * out_per_k + mdn_k)  # params + weights
            )
        elif head_type == "student_t":
            # Student-t MDN with K components
            out_per_k = pred_len * 5  # mu_x, mu_y, log_sx, log_sy, log_nu
            self.head = nn.Sequential(
                nn.Linear(d_model, 192), nn.GELU(),
                nn.Linear(192, 192), nn.GELU(),
                nn.Linear(192, mdn_k * out_per_k + mdn_k)  # params + weights
            )

    def forward(self, past_disp):
        """
        past_disp: (B, obs_len, 2)
        Returns dict with prediction parameters
        """
        B = past_disp.shape[0]

        # Embed + positional encoding
        h = self.input_proj(past_disp) + self.pe[:, :self.obs_len, :]  # (B, T, d)

        # Transformer encode
        h = self.encoder(h)  # (B, T, d)

        # Average pooling
        h_hist = h.mean(dim=1)  # (B, d)

        # MDN head
        raw = self.head(h_hist)  # (B, ?)

        if self.head_type == "gaussian":
            raw = raw.view(B, self.pred_len, 4)
            return {
                "mu": raw[..., :2],  # (B, pred_len, 2)
                "log_sigma": raw[..., 2:4],
            }

        elif self.head_type == "mdn_gaussian":
            K = self.mdn_k
            T = self.pred_len
            weights_raw = raw[:, :K]
            params_raw = raw[:, K:].view(B, K, T, 4)
            return {
                "weights": torch.softmax(weights_raw, dim=-1),  # (B, K)
                "mu": params_raw[..., :2],  # (B, K, T, 2)
                "log_sigma": params_raw[..., 2:4],
            }

        elif self.head_type == "student_t":
            K = self.mdn_k
            T = self.pred_len
            weights_raw = raw[:, :K]
            params_raw = raw[:, K:].view(B, K, T, 5)
            return {
                "weights": torch.softmax(weights_raw, dim=-1),
                "mu": params_raw[..., :2],
                "log_sigma": params_raw[..., 2:4],
                "log_nu": params_raw[..., 4],  # degrees of freedom
            }


# ============================================================
# Loss functions
# ============================================================

def gaussian_nll_loss(output, target_disp):
    """Simple Gaussian NLL for single-head model."""
    mu = output["mu"]  # (B, T, 2)
    log_s = output["log_sigma"].clamp(-6, 6)  # (B, T, 2)
    sigma = torch.exp(log_s)

    diff = target_disp - mu
    nll = 0.5 * (diff / sigma.clamp(min=1e-6))**2 + log_s + 0.5 * math.log(2 * math.pi)
    return nll.mean()


def mdn_gaussian_nll_loss(output, target_disp):
    """Gaussian MDN NLL with WTA."""
    weights = output["weights"]  # (B, K)
    mu = output["mu"]  # (B, K, T, 2)
    log_s = output["log_sigma"].clamp(-6, 6)
    sigma = torch.exp(log_s)
    B, K, T, _ = mu.shape

    target = target_disp.unsqueeze(1).expand_as(mu)  # (B, K, T, 2)
    diff = target - mu

    nll_per_dim = 0.5 * (diff / sigma.clamp(min=1e-6))**2 + log_s + 0.5 * math.log(2 * math.pi)
    nll_per_k = nll_per_dim.sum(dim=-1).mean(dim=-1)  # (B, K)

    # WTA
    best_k = nll_per_k.argmin(dim=1)  # (B,)
    best_nll = nll_per_k[torch.arange(B), best_k]
    best_log_w = torch.log(weights[torch.arange(B), best_k].clamp(min=1e-8))

    return (best_nll - best_log_w).mean()


def student_t_nll_loss(output, target_disp):
    """Student-t MDN NLL with WTA (matching your paper's Eq. 10)."""
    weights = output["weights"]  # (B, K)
    mu = output["mu"]  # (B, K, T, 2)
    log_s = output["log_sigma"].clamp(-6, 6)
    log_nu_raw = output["log_nu"]  # (B, K, T)
    B, K, T, _ = mu.shape

    sigma = torch.exp(log_s)
    nu = torch.nn.functional.softplus(log_nu_raw) + 2.0  # ensure nu > 2

    target = target_disp.unsqueeze(1).expand_as(mu)
    diff = target - mu

    # Factored Student-t NLL (per dimension, independent)
    d = 2  # dimensions
    z_sq = (diff / sigma.clamp(min=1e-6))**2  # (B, K, T, 2)
    z_sum = z_sq.sum(dim=-1)  # (B, K, T)

    # Log Student-t density (up to constants)
    nll = 0.5 * (nu + d).unsqueeze(-1) * torch.log1p(z_sq / nu.unsqueeze(-1))  # (B, K, T, 2)
    nll = nll.sum(dim=-1)  # (B, K, T)
    nll = nll + log_s.sum(dim=-1)  # add log scale terms
    nll = nll + 0.5 * torch.log(nu) + math.log(math.pi)  # normalizing constant approx

    nll_per_k = nll.mean(dim=-1)  # (B, K)

    # WTA
    best_k = nll_per_k.argmin(dim=1)
    best_nll = nll_per_k[torch.arange(B), best_k]
    best_log_w = torch.log(weights[torch.arange(B), best_k].clamp(min=1e-8))

    return (best_nll - best_log_w).mean()


# ============================================================
# Evaluation
# ============================================================

@torch.no_grad()
def evaluate_model(model, dataloader, device, num_samples=20):
    model.eval()
    all_ade_30, all_fde_30, all_ade_60, all_fde_60 = [], [], [], []

    for pasts, futures in dataloader:
        pasts, futures = pasts.to(device), futures.to(device)
        B = pasts.shape[0]

        # Displacements
        past_disp = pasts[:, 1:, :] - pasts[:, :-1, :]
        past_disp_padded = torch.cat([torch.zeros(B, 1, 2, device=device), past_disp], dim=1)
        first_fd = futures[:, 0:1, :] - pasts[:, -1:, :]
        future_disp = torch.cat([first_fd, futures[:, 1:, :] - futures[:, :-1, :]], dim=1)

        output = model(past_disp_padded)

        # Sample trajectories
        all_trajs = []
        for _ in range(num_samples):
            if model.head_type == "gaussian":
                mu = output["mu"]  # (B, T, 2)
                log_s = output["log_sigma"].clamp(-6, 6)
                sigma = torch.exp(log_s)
                sampled_disp = mu + sigma * torch.randn_like(mu)
                sampled_pos = sampled_disp.cumsum(dim=1) + pasts[:, -1:, :]
                all_trajs.append(sampled_pos.unsqueeze(1))
            else:
                # MDN: pick from each component
                mu = output["mu"]  # (B, K, T, 2)
                log_s = output["log_sigma"].clamp(-6, 6)
                sigma = torch.exp(log_s)

                for k in range(mu.shape[1]):
                    sampled_disp = mu[:, k] + sigma[:, k] * torch.randn_like(mu[:, k])
                    sampled_pos = sampled_disp.cumsum(dim=1) + pasts[:, -1:, :]
                    all_trajs.append(sampled_pos.unsqueeze(1))

        all_trajs = torch.cat(all_trajs, dim=1)  # (B, total_K, 60, 2)
        gt = futures.unsqueeze(1).expand_as(all_trajs)
        errors = torch.norm(all_trajs - gt, dim=-1)  # (B, total_K, 60)

        # 60-frame
        ade_k = errors.mean(dim=-1)
        fde_k = errors[:, :, -1]
        all_ade_60.append(ade_k.min(dim=1).values)
        all_fde_60.append(fde_k.min(dim=1).values)

        # 30-frame
        ade_30_k = errors[:, :, :30].mean(dim=-1)
        fde_30_k = errors[:, :, 29]
        all_ade_30.append(ade_30_k.min(dim=1).values)
        all_fde_30.append(fde_30_k.min(dim=1).values)

    return {
        "minADE_30": torch.cat(all_ade_30).mean().item(),
        "minFDE_30": torch.cat(all_fde_30).mean().item(),
        "minADE_60": torch.cat(all_ade_60).mean().item(),
        "minFDE_60": torch.cat(all_fde_60).mean().item(),
    }


# ============================================================
# Train + Ablation
# ============================================================

def train_and_evaluate(head_type, train_loader, val_loader, test_loader,
                       device, epochs=80, lr=5e-4, save_dir="./output_ablation"):
    """Train a Mini-BirdFormer variant and evaluate."""

    model = MiniBirdFormer(
        d_model=96, nhead=4, num_layers=2, ff_dim=192,
        obs_len=8, pred_len=60,
        head_type=head_type, mdn_k=5, dropout=0.1
    ).to(device)

    params = sum(p.numel() for p in model.parameters())
    print(f"\n  [{head_type}] Parameters: {params/1e6:.3f} M")

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.02)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Choose loss
    if head_type == "gaussian":
        loss_fn = gaussian_nll_loss
    elif head_type == "mdn_gaussian":
        loss_fn = mdn_gaussian_nll_loss
    else:
        loss_fn = student_t_nll_loss

    best_val = float("inf")
    best_path = os.path.join(save_dir, f"best_{head_type}.pt")

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, cnt = 0, 0
        for pasts, futures in train_loader:
            pasts, futures = pasts.to(device), futures.to(device)
            B = pasts.shape[0]

            past_disp = pasts[:, 1:, :] - pasts[:, :-1, :]
            past_disp_padded = torch.cat([torch.zeros(B, 1, 2, device=device), past_disp], dim=1)
            first_fd = futures[:, 0:1, :] - pasts[:, -1:, :]
            future_disp = torch.cat([first_fd, futures[:, 1:, :] - futures[:, :-1, :]], dim=1)

            output = model(past_disp_padded)
            loss = loss_fn(output, future_disp)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item() * B
            cnt += B

        scheduler.step()
        avg_loss = total_loss / max(cnt, 1)

        if epoch % 10 == 0 or epoch == epochs:
            val_res = evaluate_model(model, val_loader, device, num_samples=20)
            val_ade = val_res["minADE_60"]
            print(f"    Epoch {epoch:3d} | loss={avg_loss:.4f} | val_ADE60={val_ade:.4f}")
            if val_ade < best_val:
                best_val = val_ade
                torch.save(model.state_dict(), best_path)

    # Test
    model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
    test_res = evaluate_model(model, test_loader, device, num_samples=20)
    test_res["params_M"] = round(params / 1e6, 3)

    return test_res


def main():
    parser = argparse.ArgumentParser(description="Mini-BirdFormer Ablation Study")
    parser.add_argument("--train_jsonl", type=str, required=True)
    parser.add_argument("--val_jsonl", type=str, required=True)
    parser.add_argument("--test_jsonl", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--save_dir", type=str, default="./output_ablation")
    args = parser.parse_args()

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    os.makedirs(args.save_dir, exist_ok=True)

    # Load data
    print("Loading data...")
    from run_stgcnn import BirdTrajectoryDataset, collate_fn
    train_ds = BirdTrajectoryDataset(args.train_jsonl)
    val_ds = BirdTrajectoryDataset(args.val_jsonl)
    test_ds = BirdTrajectoryDataset(args.test_jsonl)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_fn, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            collate_fn=collate_fn, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             collate_fn=collate_fn, num_workers=0)

    # Run ablation for each head type
    head_types = ["gaussian", "mdn_gaussian", "student_t"]
    all_results = {}

    for ht in head_types:
        print(f"\n{'='*50}")
        print(f"  ABLATION: {ht}")
        print(f"{'='*50}")
        res = train_and_evaluate(ht, train_loader, val_loader, test_loader,
                                 device, epochs=args.epochs, save_dir=args.save_dir)
        all_results[ht] = res

    # Print comparison table
    print(f"\n\n{'='*70}")
    print(f"  ABLATION RESULTS — Mini-BirdFormer (2L-96d)")
    print(f"{'='*70}")
    print(f"{'Head Type':<16} {'minADE(30)':<12} {'minFDE(30)':<12} {'minADE(60)':<12} {'minFDE(60)':<12} {'Params(M)':<10}")
    print("-" * 70)
    for ht in head_types:
        r = all_results[ht]
        print(f"{ht:<16} {r['minADE_30']:<12.4f} {r['minFDE_30']:<12.4f} "
              f"{r['minADE_60']:<12.4f} {r['minFDE_60']:<12.4f} {r['params_M']:<10.3f}")
    print("=" * 70)

    # Save
    with open(os.path.join(args.save_dir, "ablation_results.json"), "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {args.save_dir}/ablation_results.json")


if __name__ == "__main__":
    main()
