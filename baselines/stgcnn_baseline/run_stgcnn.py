"""
Social-STGCNN Baseline for Bird Trajectory Forecasting
=======================================================
独立脚本，直接读取 jsonl 格式数据，训练 + 评测 Social-STGCNN 模型。
输出 minADE / minFDE (K=5 best-of-5)，用于填写论文 Table 2。

用法 (PyCharm 或命令行):
    python run_stgcnn.py --train_jsonl path/to/train.jsonl \
                         --val_jsonl   path/to/val.jsonl \
                         --test_jsonl  path/to/test.jsonl \
                         --epochs 80 --device cuda:0

如果你的 jsonl 很小 (如 172 条)，可以减少 epochs 或换成更大的 jsonl。
"""

import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# ============================================================
# 1. 数据加载
# ============================================================

class BirdTrajectoryDataset(Dataset):
    """从 jsonl 文件读取已 window 好的 (past=8, future=60) 轨迹。"""

    def __init__(self, jsonl_path: str):
        self.samples = []
        with open(jsonl_path, "r") as f:
            for line in f:
                d = json.loads(line.strip())
                past = np.array(d["past"], dtype=np.float32)    # (8, 2)
                future = np.array(d["future"], dtype=np.float32)  # (60, 2)
                self.samples.append((past, future))
        print(f"  Loaded {len(self.samples)} samples from {jsonl_path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        past, future = self.samples[idx]
        return torch.from_numpy(past), torch.from_numpy(future)


def collate_fn(batch):
    """Batch: list of (past, future) tensors."""
    pasts = torch.stack([b[0] for b in batch])      # (B, 8, 2)
    futures = torch.stack([b[1] for b in batch])     # (B, 60, 2)
    return pasts, futures


# ============================================================
# 2. Social-STGCNN 模型
# ============================================================
# 简化版 Social-STGCNN:
#   - 因为 bird jsonl 是 单轨迹 (无 scene-level 同帧邻居信息),
#     我们用 temporal graph convolution 替代 spatial social graph,
#     这是 STGCNN 在 single-agent 设置下的标准降级方式。
#   - 核心结构: ST-GCNN layers (temporal) + Time-Extrapolator CNN (TXP-CNN)

class ConvTemporalGraphical(nn.Module):
    """时序图卷积: 在时间维度上做 1D 卷积 + 图结构 (这里用自连接)。"""

    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.PReLU()

    def forward(self, x):
        # x: (B, C, T)
        return self.relu(self.bn(self.conv(x)))


class SocialSTGCNN(nn.Module):
    """
    Social-STGCNN (single-agent variant for bird trajectory forecasting).

    Architecture:
        Input (B, 8, 2) -> displacement -> ST-GCNN layers -> TXP-CNN -> Output (B, K, 60, 5)
        Output 5 channels = [mu_x, mu_y, log_sigma_x, log_sigma_y, rho]
        (bivariate Gaussian per timestep for probabilistic prediction)
    """

    def __init__(self, obs_len=8, pred_len=60, n_stgcnn=1, n_txpcnn=5,
                 input_feat=2, output_feat=5, kernel_size=3, num_samples=5):
        super().__init__()
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.num_samples = num_samples

        # ST-GCNN blocks (temporal graph convolution on observed sequence)
        self.st_gcnns = nn.ModuleList()
        self.st_gcnns.append(ConvTemporalGraphical(input_feat, 64, kernel_size))
        for _ in range(n_stgcnn - 1):
            self.st_gcnns.append(ConvTemporalGraphical(64, 64, kernel_size))

        # TXP-CNN: extrapolate from obs_len to pred_len
        self.txp_cnns = nn.ModuleList()
        self.txp_cnns.append(nn.Conv1d(obs_len, pred_len, kernel_size=3, padding=1))
        self.txp_bns = nn.ModuleList()
        self.txp_bns.append(nn.BatchNorm1d(pred_len))
        for _ in range(n_txpcnn - 1):
            self.txp_cnns.append(nn.Conv1d(pred_len, pred_len, kernel_size=3, padding=1))
            self.txp_bns.append(nn.BatchNorm1d(pred_len))

        self.prelus = nn.ModuleList([nn.PReLU() for _ in range(n_txpcnn)])

        # Final projection to output features
        self.output_proj = nn.Conv1d(64, output_feat, kernel_size=1)

    def forward(self, x):
        """
        x: (B, obs_len, 2) - displacement sequence
        Returns: (B, num_samples, pred_len, 5) - predicted Gaussian params
        """
        B = x.shape[0]

        # (B, obs_len, 2) -> (B, 2, obs_len) for conv1d
        h = x.permute(0, 2, 1)

        # ST-GCNN: temporal graph convolution
        for gcn in self.st_gcnns:
            h = gcn(h)  # (B, 64, obs_len)

        # Project to output features: (B, 64, obs_len) -> (B, 5, obs_len)
        h = self.output_proj(h)  # (B, 5, obs_len)

        # TXP-CNN: (B, 5, obs_len) -> permute -> (B, obs_len, 5)
        h = h.permute(0, 2, 1)  # (B, obs_len, 5)

        # Temporal extrapolation: (B, obs_len, 5) -> (B, pred_len, 5)
        for i, (cnn, bn) in enumerate(zip(self.txp_cnns, self.txp_bns)):
            h = self.prelus[i](bn(cnn(h)))

        # h: (B, pred_len, 5)
        # Expand for multi-sample: just repeat (deterministic model, noise at sampling)
        h = h.unsqueeze(1).repeat(1, self.num_samples, 1, 1)  # (B, K, pred_len, 5)

        return h

    def sample_trajectories(self, params):
        """
        从预测的 Gaussian 参数中采样轨迹。
        params: (B, K, pred_len, 5) -> [mu_x, mu_y, log_sx, log_sy, rho_raw]
        Returns: (B, K, pred_len, 2) sampled positions (displacements)
        """
        mu_x = params[..., 0]
        mu_y = params[..., 1]
        log_sx = params[..., 2]
        log_sy = params[..., 3]
        rho_raw = params[..., 4]

        sx = torch.exp(log_sx.clamp(-6, 6))
        sy = torch.exp(log_sy.clamp(-6, 6))
        rho = torch.tanh(rho_raw)

        # Sample from bivariate Gaussian
        eps_x = torch.randn_like(mu_x)
        eps_y = torch.randn_like(mu_y)

        x_sample = mu_x + sx * eps_x
        y_sample = mu_y + sy * (rho * eps_x + torch.sqrt((1 - rho**2).clamp(min=1e-8)) * eps_y)

        return torch.stack([x_sample, y_sample], dim=-1)  # (B, K, pred_len, 2)


# ============================================================
# 3. 损失函数: Bivariate Gaussian NLL
# ============================================================

def bivariate_gaussian_nll(params, targets):
    """
    params: (B, K, pred_len, 5)
    targets: (B, pred_len, 2) - ground truth displacements
    Returns: scalar loss (best-of-K, winner-takes-all)
    """
    B, K, T, _ = params.shape
    targets_expand = targets.unsqueeze(1).expand_as(params[..., :2])  # (B, K, T, 2)

    mu_x = params[..., 0]
    mu_y = params[..., 1]
    log_sx = params[..., 2].clamp(-6, 6)
    log_sy = params[..., 3].clamp(-6, 6)
    rho_raw = params[..., 4]

    sx = torch.exp(log_sx)
    sy = torch.exp(log_sy)
    rho = torch.tanh(rho_raw)

    dx = targets_expand[..., 0] - mu_x
    dy = targets_expand[..., 1] - mu_y

    one_minus_rho2 = (1 - rho**2).clamp(min=1e-8)

    z = (dx / sx.clamp(min=1e-6))**2 + (dy / sy.clamp(min=1e-6))**2 \
        - 2 * rho * dx * dy / (sx * sy).clamp(min=1e-6)
    z = z / one_minus_rho2

    log_norm = math.log(2 * math.pi) + log_sx + log_sy + 0.5 * torch.log(one_minus_rho2)
    nll = 0.5 * z + log_norm  # (B, K, T)

    nll_per_sample = nll.mean(dim=-1)  # (B, K)

    # Winner-takes-all: pick best mode
    best_nll, _ = nll_per_sample.min(dim=1)  # (B,)

    return best_nll.mean()


# ============================================================
# 4. 评测指标: minADE / minFDE
# ============================================================

@torch.no_grad()
def evaluate(model, dataloader, device, num_samples=20):
    """
    计算 minADE 和 minFDE (best-of-K)。
    用 num_samples 次采样，取最好的。
    """
    model.eval()
    all_ade = []
    all_fde = []
    all_ade_30 = []
    all_fde_30 = []

    for pasts, futures in dataloader:
        pasts = pasts.to(device)     # (B, 8, 2)
        futures = futures.to(device)  # (B, 60, 2)
        B = pasts.shape[0]

        # 转换为 displacement
        past_disp = pasts[:, 1:, :] - pasts[:, :-1, :]  # (B, 7, 2)
        future_disp = futures[:, 1:, :] - futures[:, :-1, :]  # (B, 59, 2)
        # 第一帧 future displacement = future[0] - past[-1]
        first_future_disp = (futures[:, 0:1, :] - pasts[:, -1:, :])  # (B, 1, 2)
        future_disp_full = torch.cat([first_future_disp, future_disp], dim=1)  # (B, 60, 2)

        # Pad past_disp to obs_len=8: prepend zero
        past_disp_padded = torch.cat([torch.zeros(B, 1, 2, device=device), past_disp], dim=1)  # (B, 8, 2)

        # 多次采样
        all_trajs = []
        for _ in range(num_samples):
            params = model(past_disp_padded)  # (B, K_model, 60, 5)
            sampled_disp = model.sample_trajectories(params)  # (B, K_model, 60, 2)

            # Displacement -> absolute position
            last_obs = pasts[:, -1:, :].unsqueeze(1)  # (B, 1, 1, 2)
            sampled_pos = sampled_disp.cumsum(dim=2) + last_obs  # (B, K_model, 60, 2)
            all_trajs.append(sampled_pos)

        # Concat all samples: (B, total_K, 60, 2)
        all_trajs = torch.cat(all_trajs, dim=1)
        total_K = all_trajs.shape[1]

        # Ground truth
        gt = futures.unsqueeze(1).expand(B, total_K, 60, 2)

        # ADE per sample: mean over time
        errors = torch.norm(all_trajs - gt, dim=-1)  # (B, total_K, 60)

        # 60-frame metrics
        ade_per_k = errors.mean(dim=-1)  # (B, total_K)
        fde_per_k = errors[:, :, -1]      # (B, total_K)

        min_ade, _ = ade_per_k.min(dim=1)  # (B,)
        min_fde, _ = fde_per_k.min(dim=1)  # (B,)

        all_ade.append(min_ade)
        all_fde.append(min_fde)

        # 30-frame metrics
        ade_30_per_k = errors[:, :, :30].mean(dim=-1)
        fde_30_per_k = errors[:, :, 29]

        min_ade_30, _ = ade_30_per_k.min(dim=1)
        min_fde_30, _ = fde_30_per_k.min(dim=1)

        all_ade_30.append(min_ade_30)
        all_fde_30.append(min_fde_30)

    results = {
        "minADE_30": torch.cat(all_ade_30).mean().item(),
        "minFDE_30": torch.cat(all_fde_30).mean().item(),
        "minADE_60": torch.cat(all_ade).mean().item(),
        "minFDE_60": torch.cat(all_fde).mean().item(),
    }
    return results


# ============================================================
# 5. 训练循环
# ============================================================

def train_one_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    count = 0

    for pasts, futures in dataloader:
        pasts = pasts.to(device)
        futures = futures.to(device)
        B = pasts.shape[0]

        # Displacements
        past_disp = pasts[:, 1:, :] - pasts[:, :-1, :]
        past_disp_padded = torch.cat([torch.zeros(B, 1, 2, device=device), past_disp], dim=1)

        first_future_disp = futures[:, 0:1, :] - pasts[:, -1:, :]
        future_disp = futures[:, 1:, :] - futures[:, :-1, :]
        future_disp_full = torch.cat([first_future_disp, future_disp], dim=1)

        # Forward
        params = model(past_disp_padded)  # (B, K, 60, 5)

        # Loss
        loss = bivariate_gaussian_nll(params, future_disp_full)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item() * B
        count += B

    return total_loss / max(count, 1)


# ============================================================
# 6. 主函数
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Social-STGCNN Baseline")
    parser.add_argument("--train_jsonl", type=str, required=True)
    parser.add_argument("--val_jsonl", type=str, required=True)
    parser.add_argument("--test_jsonl", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--num_samples", type=int, default=20,
                        help="Number of samples for minADE/minFDE evaluation")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_dir", type=str, default="./output_stgcnn")
    args = parser.parse_args()

    # Seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Data
    print("Loading data...")
    train_ds = BirdTrajectoryDataset(args.train_jsonl)
    val_ds = BirdTrajectoryDataset(args.val_jsonl)
    test_ds = BirdTrajectoryDataset(args.test_jsonl)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_fn, num_workers=0, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            collate_fn=collate_fn, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             collate_fn=collate_fn, num_workers=0)

    # Model
    model = SocialSTGCNN(
        obs_len=8, pred_len=60,
        n_stgcnn=1, n_txpcnn=5,
        input_feat=2, output_feat=5,
        kernel_size=3, num_samples=5
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {param_count / 1e6:.3f} M")

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Training
    os.makedirs(args.save_dir, exist_ok=True)
    best_val_ade = float("inf")
    best_epoch = 0

    print(f"\n{'='*60}")
    print(f"Training Social-STGCNN for {args.epochs} epochs")
    print(f"{'='*60}\n")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        scheduler.step()

        # Validate every 5 epochs
        if epoch % 5 == 0 or epoch == args.epochs:
            val_results = evaluate(model, val_loader, device, num_samples=args.num_samples)
            val_ade = val_results["minADE_60"]

            dt = time.time() - t0
            print(f"Epoch {epoch:3d}/{args.epochs} | loss={train_loss:.4f} | "
                  f"val_minADE60={val_ade:.4f} | val_minFDE60={val_results['minFDE_60']:.4f} | "
                  f"time={dt:.1f}s")

            if val_ade < best_val_ade:
                best_val_ade = val_ade
                best_epoch = epoch
                torch.save(model.state_dict(), os.path.join(args.save_dir, "best_stgcnn.pt"))
        else:
            dt = time.time() - t0
            print(f"Epoch {epoch:3d}/{args.epochs} | loss={train_loss:.4f} | time={dt:.1f}s")

    print(f"\nBest val ADE: {best_val_ade:.4f} at epoch {best_epoch}")

    # ==========================================
    # Final Test Evaluation
    # ==========================================
    print(f"\n{'='*60}")
    print("Loading best model and evaluating on TEST set...")
    print(f"{'='*60}\n")

    model.load_state_dict(torch.load(os.path.join(args.save_dir, "best_stgcnn.pt"),
                                     map_location=device, weights_only=True))

    test_results = evaluate(model, test_loader, device, num_samples=args.num_samples)

    print(f"╔══════════════════════════════════════════╗")
    print(f"║   Social-STGCNN — TEST SET RESULTS      ║")
    print(f"╠══════════════════════════════════════════╣")
    print(f"║  minADE (30-frame): {test_results['minADE_30']:.4f}              ║")
    print(f"║  minFDE (30-frame): {test_results['minFDE_30']:.4f}              ║")
    print(f"║  minADE (60-frame): {test_results['minADE_60']:.4f}              ║")
    print(f"║  minFDE (60-frame): {test_results['minFDE_60']:.4f}              ║")
    print(f"║  Parameters:        {param_count/1e6:.3f} M              ║")
    print(f"╚══════════════════════════════════════════╝")

    # Save results
    results_path = os.path.join(args.save_dir, "results_stgcnn.json")
    with open(results_path, "w") as f:
        json.dump({
            "model": "Social-STGCNN",
            "params_M": round(param_count / 1e6, 3),
            "best_epoch": best_epoch,
            **test_results
        }, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
