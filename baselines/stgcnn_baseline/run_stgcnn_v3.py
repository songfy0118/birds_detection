"""
Social-STGCNN Baseline v3 — 确定性版本
======================================
简化为确定性预测（直接输出mu），避免概率采样在不同尺度下的数值问题。
用 MSE loss 训练，输出 ADE/FDE（不是 minADE，因为是单一确定性预测）。

对于论文 Table 2，Social-STGCNN 作为 GNN 范式的代表，确定性结果完全合理。
"""

import argparse, json, math, os, random, time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


# ============================================================
# 1. Data
# ============================================================

class BirdTrajectoryDataset(Dataset):
    def __init__(self, jsonl_path):
        self.samples = []
        with open(jsonl_path) as f:
            for line in f:
                d = json.loads(line.strip())
                past = np.array(d["past"], dtype=np.float32)
                future = np.array(d["future"], dtype=np.float32)
                W, H = d.get("W", 1280), d.get("H", 720)
                self.samples.append((past, future, W, H))
        print(f"  Loaded {len(self.samples)} samples from {jsonl_path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        past, future, W, H = self.samples[idx]
        # 转为 displacement（像素空间），再除以参考尺度归一化
        # 用固定参考尺度 1280 避免不同分辨率的问题
        REF = 1280.0
        past_t = torch.from_numpy(past) / REF
        future_t = torch.from_numpy(future) / REF
        return past_t, future_t, torch.tensor(REF, dtype=torch.float32)


def collate_fn(batch):
    return (torch.stack([b[0] for b in batch]),
            torch.stack([b[1] for b in batch]),
            torch.stack([b[2] for b in batch]))


# ============================================================
# 2. Model: Social-STGCNN (deterministic)
# ============================================================

class ConvTemporalBlock(nn.Module):
    def __init__(self, in_ch, out_ch, ks=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, ks, padding=ks // 2),
            nn.BatchNorm1d(out_ch),
            nn.PReLU()
        )
    def forward(self, x):
        return self.net(x)


class SocialSTGCNN(nn.Module):
    def __init__(self, obs_len=8, pred_len=60, hidden=64, n_stgcnn=2, n_txpcnn=5):
        super().__init__()
        self.pred_len = pred_len

        # Spatio-temporal graph conv (temporal only for single-agent)
        layers = [ConvTemporalBlock(2, hidden)]
        for _ in range(n_stgcnn - 1):
            layers.append(ConvTemporalBlock(hidden, hidden))
        self.st_gcn = nn.Sequential(*layers)

        # Project features to 2D output
        self.feat_proj = nn.Conv1d(hidden, 2, 1)

        # Time extrapolation CNN: obs_len -> pred_len
        txp = [nn.Conv1d(obs_len, pred_len, 3, padding=1), nn.BatchNorm1d(pred_len), nn.PReLU()]
        for _ in range(n_txpcnn - 1):
            txp += [nn.Conv1d(pred_len, pred_len, 3, padding=1), nn.BatchNorm1d(pred_len), nn.PReLU()]
        self.txp_cnn = nn.Sequential(*txp)

        # Final output: pred_len -> pred_len, 2 -> 2
        self.out = nn.Linear(2, 2)

    def forward(self, past_disp):
        """
        past_disp: (B, obs_len, 2) displacement sequence (normalized)
        Returns: (B, pred_len, 2) predicted future displacements
        """
        # (B, 2, obs_len)
        h = past_disp.permute(0, 2, 1)
        h = self.st_gcn(h)          # (B, hidden, obs_len)
        h = self.feat_proj(h)       # (B, 2, obs_len)
        h = h.permute(0, 2, 1)     # (B, obs_len, 2)
        h = self.txp_cnn(h)         # (B, pred_len, 2)
        h = self.out(h)             # (B, pred_len, 2)
        return h


# ============================================================
# 3. Train
# ============================================================

def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss, cnt = 0, 0
    for pasts, futures, refs in loader:
        pasts, futures = pasts.to(device), futures.to(device)
        B = pasts.shape[0]

        # Displacement input
        past_disp = pasts[:, 1:, :] - pasts[:, :-1, :]
        past_disp = torch.cat([torch.zeros(B, 1, 2, device=device), past_disp], dim=1)

        # Target displacement
        first_fd = futures[:, 0:1, :] - pasts[:, -1:, :]
        future_disp = torch.cat([first_fd, futures[:, 1:, :] - futures[:, :-1, :]], dim=1)

        pred_disp = model(past_disp)
        loss = nn.functional.mse_loss(pred_disp, future_disp)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item() * B
        cnt += B
    return total_loss / max(cnt, 1)


# ============================================================
# 4. Evaluate (pixel-space ADE/FDE)
# ============================================================

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_ade_30, all_fde_30, all_ade_60, all_fde_60 = [], [], [], []

    for pasts, futures, refs in loader:
        pasts, futures, refs = pasts.to(device), futures.to(device), refs.to(device)
        B = pasts.shape[0]
        REF = refs[0].item()  # all same = 1280

        past_disp = pasts[:, 1:, :] - pasts[:, :-1, :]
        past_disp = torch.cat([torch.zeros(B, 1, 2, device=device), past_disp], dim=1)

        pred_disp = model(past_disp)  # (B, 60, 2) normalized

        # Reconstruct absolute position (normalized)
        pred_pos = pred_disp.cumsum(dim=1) + pasts[:, -1:, :]

        # Convert to pixel space
        pred_px = pred_pos * REF
        gt_px = futures * REF

        errors = torch.norm(pred_px - gt_px, dim=-1)  # (B, 60)

        all_ade_60.append(errors.mean(dim=-1))
        all_fde_60.append(errors[:, -1])
        all_ade_30.append(errors[:, :30].mean(dim=-1))
        all_fde_30.append(errors[:, 29])

    return {
        "ADE_30": torch.cat(all_ade_30).mean().item(),
        "FDE_30": torch.cat(all_fde_30).mean().item(),
        "ADE_60": torch.cat(all_ade_60).mean().item(),
        "FDE_60": torch.cat(all_fde_60).mean().item(),
    }


# ============================================================
# 5. Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_jsonl", required=True)
    parser.add_argument("--val_jsonl", required=True)
    parser.add_argument("--test_jsonl", required=True)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--save_dir", type=str, default="./output_stgcnn")
    args = parser.parse_args()

    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading data...")
    train_ds = BirdTrajectoryDataset(args.train_jsonl)
    val_ds = BirdTrajectoryDataset(args.val_jsonl)
    test_ds = BirdTrajectoryDataset(args.test_jsonl)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    model = SocialSTGCNN(obs_len=8, pred_len=60, hidden=64, n_stgcnn=2, n_txpcnn=5).to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {params/1e6:.3f} M")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    os.makedirs(args.save_dir, exist_ok=True)
    best_val, best_ep = float("inf"), 0

    print(f"\n{'='*60}\nTraining Social-STGCNN (deterministic) for {args.epochs} epochs\n{'='*60}\n")

    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        loss = train_one_epoch(model, train_loader, optimizer, device)
        scheduler.step()

        if ep % 5 == 0 or ep == args.epochs:
            vr = evaluate(model, val_loader, device)
            dt = time.time() - t0
            print(f"Epoch {ep:3d}/{args.epochs} | loss={loss:.6f} | val_ADE60={vr['ADE_60']:.4f} | val_FDE60={vr['FDE_60']:.4f} | {dt:.1f}s")
            if vr["ADE_60"] < best_val:
                best_val = vr["ADE_60"]
                best_ep = ep
                torch.save(model.state_dict(), os.path.join(args.save_dir, "best.pt"))
        else:
            print(f"Epoch {ep:3d}/{args.epochs} | loss={loss:.6f} | {time.time()-t0:.1f}s")

    print(f"\nBest val ADE60: {best_val:.4f} at epoch {best_ep}")
    print(f"\n{'='*60}\nEvaluating on TEST set...\n{'='*60}\n")

    model.load_state_dict(torch.load(os.path.join(args.save_dir, "best.pt"), map_location=device, weights_only=True))
    tr = evaluate(model, test_loader, device)

    print(f"╔══════════════════════════════════════════════╗")
    print(f"║   Social-STGCNN — TEST RESULTS (pixels)     ║")
    print(f"╠══════════════════════════════════════════════╣")
    print(f"║  ADE  (30-frame):  {tr['ADE_30']:>8.3f}                  ║")
    print(f"║  FDE  (30-frame):  {tr['FDE_30']:>8.3f}                  ║")
    print(f"║  ADE  (60-frame):  {tr['ADE_60']:>8.3f}                  ║")
    print(f"║  FDE  (60-frame):  {tr['FDE_60']:>8.3f}                  ║")
    print(f"║  Parameters:       {params/1e6:.3f} M                  ║")
    print(f"╚══════════════════════════════════════════════╝")

    with open(os.path.join(args.save_dir, "results.json"), "w") as f:
        json.dump({"model": "Social-STGCNN", "params_M": round(params/1e6, 3),
                   "best_epoch": best_ep, **tr}, f, indent=2)
    print(f"Saved to {args.save_dir}/results.json")


if __name__ == "__main__":
    main()
