"""
Social-STGCNN Baseline for Bird Trajectory Forecasting (v2 - 分辨率归一化)
=========================================================================
修复: train(1280x720) vs test(混合1920x1080+1280x720) 坐标尺度不一致问题。
所有坐标归一化到 [0,1] 范围，评测时再转回像素空间计算ADE/FDE。

用法:
    python run_stgcnn_v2.py --train_jsonl ..\output\jsonl_base_8_60\train.jsonl ^
                            --val_jsonl   ..\output\jsonl_base_8_60\val.jsonl ^
                            --test_jsonl  ..\output\jsonl_base_8_60\test.jsonl ^
                            --epochs 120 --device cuda:0
"""

import argparse, json, math, os, random, time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


# ============================================================
# 1. 数据加载 (带分辨率归一化)
# ============================================================

class BirdTrajectoryDataset(Dataset):
    def __init__(self, jsonl_path: str, normalize: bool = True):
        self.samples = []
        self.normalize = normalize
        with open(jsonl_path, "r") as f:
            for line in f:
                d = json.loads(line.strip())
                past = np.array(d["past"], dtype=np.float32)
                future = np.array(d["future"], dtype=np.float32)
                W = d.get("W", 1280)
                H = d.get("H", 720)
                self.samples.append((past, future, W, H))
        print(f"  Loaded {len(self.samples)} samples from {jsonl_path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        past, future, W, H = self.samples[idx]
        past_t = torch.from_numpy(past)
        future_t = torch.from_numpy(future)

        if self.normalize:
            scale = torch.tensor([W, H], dtype=torch.float32)
            past_norm = past_t / scale
            future_norm = future_t / scale
        else:
            past_norm = past_t
            future_norm = future_t
            scale = torch.tensor([1.0, 1.0])

        return past_norm, future_norm, torch.tensor([W, H], dtype=torch.float32)


def collate_fn(batch):
    pasts = torch.stack([b[0] for b in batch])
    futures = torch.stack([b[1] for b in batch])
    scales = torch.stack([b[2] for b in batch])
    return pasts, futures, scales


# ============================================================
# 2. Social-STGCNN 模型 (同之前)
# ============================================================

class ConvTemporalGraphical(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.PReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class SocialSTGCNN(nn.Module):
    def __init__(self, obs_len=8, pred_len=60, n_stgcnn=1, n_txpcnn=5,
                 input_feat=2, output_feat=5, kernel_size=3, num_samples=5):
        super().__init__()
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.num_samples = num_samples

        self.st_gcnns = nn.ModuleList()
        self.st_gcnns.append(ConvTemporalGraphical(input_feat, 64, kernel_size))
        for _ in range(n_stgcnn - 1):
            self.st_gcnns.append(ConvTemporalGraphical(64, 64, kernel_size))

        self.txp_cnns = nn.ModuleList()
        self.txp_bns = nn.ModuleList()
        self.txp_cnns.append(nn.Conv1d(obs_len, pred_len, kernel_size=3, padding=1))
        self.txp_bns.append(nn.BatchNorm1d(pred_len))
        for _ in range(n_txpcnn - 1):
            self.txp_cnns.append(nn.Conv1d(pred_len, pred_len, kernel_size=3, padding=1))
            self.txp_bns.append(nn.BatchNorm1d(pred_len))

        self.prelus = nn.ModuleList([nn.PReLU() for _ in range(n_txpcnn)])
        self.output_proj = nn.Conv1d(64, output_feat, kernel_size=1)

    def forward(self, x):
        B = x.shape[0]
        h = x.permute(0, 2, 1)
        for gcn in self.st_gcnns:
            h = gcn(h)
        h = self.output_proj(h)
        h = h.permute(0, 2, 1)
        for i, (cnn, bn) in enumerate(zip(self.txp_cnns, self.txp_bns)):
            h = self.prelus[i](bn(cnn(h)))
        h = h.unsqueeze(1).repeat(1, self.num_samples, 1, 1)
        return h

    def sample_trajectories(self, params):
        mu_x = params[..., 0]
        mu_y = params[..., 1]
        log_sx = params[..., 2].clamp(-6, 6)
        log_sy = params[..., 3].clamp(-6, 6)
        rho_raw = params[..., 4]

        sx = torch.exp(log_sx)
        sy = torch.exp(log_sy)
        rho = torch.tanh(rho_raw)

        eps_x = torch.randn_like(mu_x)
        eps_y = torch.randn_like(mu_y)

        x_sample = mu_x + sx * eps_x
        y_sample = mu_y + sy * (rho * eps_x + torch.sqrt((1 - rho**2).clamp(min=1e-8)) * eps_y)

        return torch.stack([x_sample, y_sample], dim=-1)


# ============================================================
# 3. 损失函数
# ============================================================

def bivariate_gaussian_nll(params, targets):
    B, K, T, _ = params.shape
    targets_expand = targets.unsqueeze(1).expand_as(params[..., :2])

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
    nll = 0.5 * z + log_norm

    nll_per_sample = nll.mean(dim=-1)
    best_nll, _ = nll_per_sample.min(dim=1)
    return best_nll.mean()


# ============================================================
# 4. 评测 (转回像素空间计算ADE/FDE)
# ============================================================

@torch.no_grad()
def evaluate(model, dataloader, device, num_samples=20):
    model.eval()
    all_ade_30, all_fde_30, all_ade_60, all_fde_60 = [], [], [], []

    for pasts, futures, scales in dataloader:
        pasts = pasts.to(device)
        futures = futures.to(device)
        scales = scales.to(device)  # (B, 2) = [W, H]
        B = pasts.shape[0]

        # Displacements (in normalized space)
        past_disp = pasts[:, 1:, :] - pasts[:, :-1, :]
        past_disp_padded = torch.cat([torch.zeros(B, 1, 2, device=device), past_disp], dim=1)
        first_fd = futures[:, 0:1, :] - pasts[:, -1:, :]
        future_disp = torch.cat([first_fd, futures[:, 1:, :] - futures[:, :-1, :]], dim=1)

        # Sample multiple trajectories
        all_trajs = []
        for _ in range(num_samples):
            params = model(past_disp_padded)
            sampled_disp = model.sample_trajectories(params)  # (B, K, 60, 2) normalized
            # Cumsum + last observed (normalized)
            sampled_pos = sampled_disp.cumsum(dim=2) + pasts[:, -1:, :].unsqueeze(1)
            all_trajs.append(sampled_pos)

        all_trajs = torch.cat(all_trajs, dim=1)  # (B, total_K, 60, 2) normalized

        # Convert back to pixel space for ADE/FDE
        scale_expand = scales.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, 2)
        all_trajs_px = all_trajs * scale_expand
        futures_px = futures * scales.unsqueeze(1)  # (B, 60, 2) in pixels

        gt = futures_px.unsqueeze(1).expand_as(all_trajs_px)
        errors = torch.norm(all_trajs_px - gt, dim=-1)  # (B, total_K, 60)

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
# 5. 训练
# ============================================================

def train_one_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss, count = 0, 0

    for pasts, futures, scales in dataloader:
        pasts = pasts.to(device)
        futures = futures.to(device)
        B = pasts.shape[0]

        past_disp = pasts[:, 1:, :] - pasts[:, :-1, :]
        past_disp_padded = torch.cat([torch.zeros(B, 1, 2, device=device), past_disp], dim=1)
        first_fd = futures[:, 0:1, :] - pasts[:, -1:, :]
        future_disp = torch.cat([first_fd, futures[:, 1:, :] - futures[:, :-1, :]], dim=1)

        params = model(past_disp_padded)
        loss = bivariate_gaussian_nll(params, future_disp)

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
    parser = argparse.ArgumentParser(description="Social-STGCNN Baseline v2")
    parser.add_argument("--train_jsonl", type=str, required=True)
    parser.add_argument("--val_jsonl", type=str, required=True)
    parser.add_argument("--test_jsonl", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--num_samples", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_dir", type=str, default="./output_stgcnn")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading data (with resolution normalization)...")
    train_ds = BirdTrajectoryDataset(args.train_jsonl, normalize=True)
    val_ds = BirdTrajectoryDataset(args.val_jsonl, normalize=True)
    test_ds = BirdTrajectoryDataset(args.test_jsonl, normalize=True)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_fn, num_workers=0, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            collate_fn=collate_fn, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             collate_fn=collate_fn, num_workers=0)

    model = SocialSTGCNN(
        obs_len=8, pred_len=60, n_stgcnn=1, n_txpcnn=5,
        input_feat=2, output_feat=5, kernel_size=3, num_samples=5
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {param_count / 1e6:.3f} M")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

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

    # Final test
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

    results_path = os.path.join(args.save_dir, "results_stgcnn.json")
    with open(results_path, "w") as f:
        json.dump({"model": "Social-STGCNN", "params_M": round(param_count / 1e6, 3),
                   "best_epoch": best_epoch, **test_results}, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
