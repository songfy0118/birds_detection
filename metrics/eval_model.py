# metrics/eval_model.py
# ------------------------------------------------------------
# Evaluate model directly on jsonl (test split)
# ------------------------------------------------------------

from __future__ import annotations

from pathlib import Path
import sys

# ====== 自动寻找项目根目录（包含 utils 和 models 的目录）======
p = Path(__file__).resolve()
while p != p.parent:
    if (p / "utils").exists() and (p / "models").exists():
        sys.path.insert(0, str(p))
        break
    p = p.parent
# ==========================================================

import argparse
import json
import torch
from torch.utils.data import DataLoader

from utils.data import TrajJsonlDataset, traj_collate
from models.transformer import TransformerForecaster
from models.lstm import LSTMForecaster
from metrics.ade_fde import ade_fde, multi_horizon_ade_fde


def build_model(args, device: torch.device):
    if args.model == "transformer":
        m = TransformerForecaster(
            past_len=args.past_len,
            future_len=args.future_len,
            d_model=args.d_model,
            nhead=args.nhead,
            enc_layers=args.layers,
            dec_layers=args.layers,
            head=args.head,
            mdn_K=args.mdn_k,
        ).to(device)
    else:
        m = LSTMForecaster(
            past_len=args.past_len,
            future_len=args.future_len,
            d_model=args.d_model,
            head=args.head,
            mdn_K=args.mdn_k,
        ).to(device)

    if args.weights and Path(args.weights).exists():
        ckpt = torch.load(args.weights, map_location=device)
        sd = ckpt.get("model", ckpt)
        m.load_state_dict(sd, strict=False)

    m.eval()
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--model", choices=["transformer", "lstm"], default="transformer")
    ap.add_argument("--head", choices=["gauss", "student_t", "mdn_gauss", "mdn_student_t"],
                    default="mdn_student_t")
    ap.add_argument("--mdn_k", type=int, default=5)

    ap.add_argument("--past_len", type=int, default=8)
    ap.add_argument("--future_len", type=int, default=60)
    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--nhead", type=int, default=8)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--horizons", type=int, nargs="+", default=[30, 60])
    ap.add_argument("--out", default="output/report_eval_model.json")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # 构建数据集 / dataloader
    ds = TrajJsonlDataset(args.jsonl, normalize=True, return_pixels=False)
    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        collate_fn=traj_collate,
    )

    # 构建模型并加载权重
    model = build_model(args, device)

    preds = []
    gts = []

    with torch.no_grad():
        for batch in dl:
            past_xy = batch["past_xy"].to(device)
            past_wh = batch["past_wh"].to(device)
            future = batch["future_xy"].to(device)  # [B,F,2]

            if args.model == "transformer":
                out = model(past_xy, past_wh, return_hidden=False)
            else:
                out = model(past_xy)

            # 不同 head 的输出统一成 [B,F,2]
            if args.head in ("gauss", "student_t"):
                mu = out["mu"]
            else:
                # MDN: 对 K 取均值作为代表轨迹（和训练时一致的设定）
                mu = out["mu"].mean(dim=2)

            preds.append(mu.cpu())
            gts.append(future.cpu())

    if not preds:
        print("[ERR] no data found")
        return

    prd = torch.cat(preds, dim=0)  # [B,F,2]
    gt = torch.cat(gts, dim=0)     # [B,F,2]

    base = ade_fde(prd, gt)
    mh = multi_horizon_ade_fde(prd, gt, args.horizons)
    out = {**base, **mh, "B": int(prd.size(0)), "F": int(prd.size(1))}

    print(json.dumps(out, indent=2, ensure_ascii=False))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
