from __future__ import annotations

import sys
from pathlib import Path

# --------- 把 day1 根目录加入 sys.path ----------
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# -------------------------------------------------

import argparse
import csv
import torch
from torch.utils.data import DataLoader

from utils.data import TrajJsonlDataset, traj_collate
from models.transformer import TransformerForecaster
from models.lstm import LSTMForecaster


def pick_mdn(mu, logits, mode: str = "winner"):
    """
    mu: [B,M,K,2]; logits: [B,M,K]
    return: [B,M,2]
    """
    if mode == "mean":
        return mu.mean(dim=2)
    k = logits.argmax(dim=-1)  # [B,M]
    idx = k.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, 2)
    return mu.gather(2, idx).squeeze(2)  # [B,M,2]


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
    ap.add_argument("--jsonl", required=True, help="test.jsonl from tracking")
    ap.add_argument("--out_dir", required=True, help="where to write *_pred.csv")
    ap.add_argument("--model", choices=["transformer", "lstm"], default="transformer")
    ap.add_argument("--head", choices=["gauss", "student_t", "mdn_gauss", "mdn_student_t"],
                    default="mdn_student_t")
    ap.add_argument("--mdn_k", type=int, default=5)
    ap.add_argument("--pick", choices=["mean", "winner", "top1"], default="winner")

    ap.add_argument("--past_len", type=int, default=8)
    ap.add_argument("--future_len", type=int, default=60)
    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--nhead", type=int, default=8)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--weights", default="")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    ds = TrajJsonlDataset(args.jsonl, normalize=True, return_pixels=True)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                    num_workers=0, pin_memory=False, collate_fn=traj_collate)

    model = build_model(args, device)

    writers = {}
    handles = {}

    with torch.no_grad():
        for batch in dl:
            past = batch["past_xy"].to(device)    # [B,P,2]
            wh   = batch["past_wh"].to(device)    # [B,P,2]
            meta = batch["meta"]

            # forward
            if args.model == "transformer":
                out = model(past, wh)
            else:
                out = model(past)

            # choose mean trajectory
            if args.head in ("gauss", "student_t"):
                mu = out["mu"]      # [B,M,2]
            else:
                mu = pick_mdn(out["mu"], out["logits"], mode=args.pick)

            # 反归一化到像素
            sW = (wh[:, -1, 0] / 2.0).unsqueeze(-1)
            sH = (wh[:, -1, 1] / 2.0).unsqueeze(-1)
            mu_px = mu.clone()
            mu_px[..., 0] = mu[..., 0] * sW + sW
            mu_px[..., 1] = mu[..., 1] * sH + sH

            for i, m in enumerate(meta):
                video = str(m["video"])
                tid   = m["track_id"]

                if video not in writers:
                    fout = out_dir / f"{video}_pred.csv"
                    f = open(fout, "w", newline="", encoding="utf-8")
                    handles[video] = f
                    w = csv.writer(f)
                    w.writerow(["video", "track_id", "sample_idx", "t", "x", "y"])
                    writers[video] = w
                    if args.verbose:
                        print(f"[CREATE] {fout}")

                w = writers[video]

                for t in range(mu_px.size(1)):
                    x = float(mu_px[i, t, 0])
                    y = float(mu_px[i, t, 1])
                    w.writerow([video, tid, i, t + 1, f"{x:.2f}", f"{y:.2f}"])

                if args.verbose:
                    print(f"[WRITE] video={video}, tid={tid}, seq_len={mu_px.size(1)}")

    for f in handles.values():
        f.close()

    print(f"[OK] Predictions saved to: {out_dir}")


if __name__ == "__main__":
    main()
