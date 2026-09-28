from __future__ import annotations

import sys
from pathlib import Path

# --------- 把 day1 根目录加入 sys.path ----------
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# -------------------------------------------------

import argparse
import time
import math
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
import torch.optim.lr_scheduler as lr_scheduler  # <--- 新增导入

from utils.data import build_loaders, ensure_dir
from models.transformer import TransformerForecaster
from models.heads import (
    nll_gauss_2d, nll_gauss_mixture_2d, wta_nll_gauss_mixture_2d,
    nll_student_t_2d, nll_student_t_mixture_2d, wta_nll_student_t_mixture_2d
)


def ade_l2(pred, gt):
    return torch.linalg.vector_norm(pred - gt, dim=-1).mean()


def fde_l2(pred, gt):
    return torch.linalg.vector_norm(pred[:, -1] - gt[:, -1], dim=-1).mean()


def joint_ade(pred, gt):
    err = torch.linalg.vector_norm(pred - gt, dim=-1)
    return err.mean(dim=1).mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_jsonl", required=True)
    ap.add_argument("--val_jsonl", required=True)
    ap.add_argument("--save_dir", default="output/weights")

    ap.add_argument("--past_len", type=int, default=8)
    ap.add_argument("--future_len", type=int, default=60)
    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--nhead", type=int, default=8)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--head", choices=["gauss", "student_t", "mdn_gauss", "mdn_student_t"],
                    default="mdn_student_t")
    ap.add_argument("--mdn_k", type=int, default=5)
    ap.add_argument("--dof_init", type=float, default=3.0)
    ap.add_argument("--learnable_dof", action="store_true")

    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--amp", action="store_true")

    ap.add_argument("--lambda_ade", type=float, default=0.2)
    ap.add_argument("--lambda_fde", type=float, default=0.5)
    ap.add_argument("--lambda_joint", type=float, default=0.2)
    ap.add_argument("--mdn_mode", choices=["nll", "wta"], default="wta")
    ap.add_argument("--out_name", default="trans_xxx.pt")

    args = ap.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    ensure_dir(args.save_dir)

    ds_tr, ds_va, dl_tr, dl_va = build_loaders(
        args.train_jsonl, args.val_jsonl,
        batch_size=args.batch_size, num_workers=4,
        normalize=True, return_pixels=False
    )

    model = TransformerForecaster(
        past_len=args.past_len, future_len=args.future_len,
        d_model=args.d_model, nhead=args.nhead,
        enc_layers=args.layers, dec_layers=args.layers,
        dropout=args.dropout, head=args.head,
        mdn_K=args.mdn_k, student_t_dof_init=args.dof_init,
        learnable_dof=args.learnable_dof, normalize_xy=False
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = GradScaler(enabled=args.amp)

    # ----------------------------------------------------
    # Q1 Turbo 核心：Warmup + Cosine Annealing 调度器
    # ----------------------------------------------------
    T_max = args.epochs * len(dl_tr)  # 总步数
    warmup_steps = int(0.1 * T_max)  # 前 10% 步数用于 Warmup

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            # Warmup: 学习率线性增加
            return float(current_step) / float(max(1, warmup_steps))
        # Cosine Annealing: 学习率从 1 衰减到 0
        progress = float(current_step - warmup_steps) / float(max(1, T_max - warmup_steps))
        return max(0.0, 0.5 * (1. + math.cos(math.pi * progress)))

    scheduler = lr_scheduler.LambdaLR(opt, lr_lambda)
    # ----------------------------------------------------

    best_val = math.inf
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        total = 0.0
        for batch in dl_tr:
            past_xy = batch["past_xy"].to(device)
            past_wh = batch["past_wh"].to(device)
            future = batch["future_xy"].to(device)

            opt.zero_grad(set_to_none=True)
            with autocast(enabled=args.amp):
                out = model(past_xy, past_wh, return_hidden=False)
                loss = 0.0

                if args.head == "gauss":
                    loss += nll_gauss_2d(out["mu"], out["log_std"], future, reduction="mean")
                    pred = out["mu"]
                elif args.head == "mdn_gauss":
                    if args.mdn_mode == "wta":
                        loss += wta_nll_gauss_mixture_2d(
                            out["logits"], out["mu"], out["log_std"], future, reduction="mean"
                        )
                    else:
                        loss += nll_gauss_mixture_2d(
                            out["logits"], out["mu"], out["log_std"], future, reduction="mean"
                        )
                    pred = out["mu"].mean(dim=2)
                elif args.head == "student_t":
                    loss += nll_student_t_2d(out["mu"], out["log_scale"], out["dof"],
                                             future, reduction="mean")
                    pred = out["mu"]
                else:  # mdn_student_t
                    if args.mdn_mode == "wta":
                        loss += wta_nll_student_t_mixture_2d(
                            out["logits"], out["mu"], out["log_scale"], out["dof"],
                            future, reduction="mean"
                        )
                    else:
                        loss += nll_student_t_mixture_2d(
                            out["logits"], out["mu"], out["log_scale"], out["dof"],
                            future, reduction="mean"
                        )
                    pred = out["mu"].mean(dim=2)

                if args.lambda_ade > 0:
                    loss += args.lambda_ade * ade_l2(pred, future)
                if args.lambda_fde > 0:
                    loss += args.lambda_fde * fde_l2(pred, future)
                if args.lambda_joint > 0:
                    loss += args.lambda_joint * joint_ade(pred, future)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            scheduler.step()  # <--- 调度器步进
            total += float(loss)

        model.eval()
        with torch.no_grad():
            val_loss, val_ade, val_fde = 0.0, 0.0, 0.0
            for batch in dl_va:
                past_xy = batch["past_xy"].to(device)
                past_wh = batch["past_wh"].to(device)
                future = batch["future_xy"].to(device)
                out = model(past_xy, past_wh, return_hidden=False)

                if args.head == "gauss":
                    ll = nll_gauss_2d(out["mu"], out["log_std"], future, reduction="mean")
                    pred = out["mu"]
                elif args.head == "mdn_gauss":
                    if args.mdn_mode == "wta":
                        ll = wta_nll_gauss_mixture_2d(
                            out["logits"], out["mu"], out["log_std"], future, reduction="mean"
                        )
                    else:
                        ll = nll_gauss_mixture_2d(
                            out["logits"], out["mu"], out["log_std"], future, reduction="mean"
                        )
                    pred = out["mu"].mean(dim=2)
                elif args.head == "student_t":
                    ll = nll_student_t_2d(out["mu"], out["log_scale"], out["dof"],
                                          future, reduction="mean")
                    pred = out["mu"]
                else:
                    if args.mdn_mode == "wta":
                        ll = wta_nll_student_t_mixture_2d(
                            out["logits"], out["mu"], out["log_scale"], out["dof"],
                            future, reduction="mean"
                        )
                    else:
                        ll = nll_student_t_mixture_2d(
                            out["logits"], out["mu"], out["log_scale"], out["dof"],
                            future, reduction="mean"
                        )
                    pred = out["mu"].mean(dim=2)

                val_loss += float(ll)
                val_ade += float(ade_l2(pred, future))
                val_fde += float(fde_l2(pred, future))

            nva = max(1, len(dl_va))
            val_loss /= nva
            val_ade /= nva
            val_fde /= nva

        dt = time.time() - t0
        print(f"[E{epoch:03d}] train {total / len(dl_tr):.4f} | "
              f"val ll {val_loss:.4f} ade {val_ade:.3f} fde {val_fde:.3f} | {dt:.1f}s")

        # 原代码：
        # metric = val_loss

        # 修改为：关注 ADE
        metric = val_ade
        if metric < best_val:
            best_val = metric
            ckpt = Path(args.save_dir) / f"trans_{args.head}_{args.mdn_mode}_best_ade.pt"
            torch.save({"model": model.state_dict(), "args": vars(args)}, ckpt)
            print("  ↳ saved best ADE:", ckpt)


if __name__ == "__main__":
    main()
