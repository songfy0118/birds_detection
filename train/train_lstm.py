# train/train_lstm.py
from __future__ import annotations
from pathlib import Path
import argparse, time, math
import torch
from torch.cuda.amp import autocast, GradScaler
from utils.data import build_loaders, ensure_dir
from models.lstm import LSTMForecaster
from models.heads import (
    nll_gauss_2d, nll_gauss_mixture_2d, wta_nll_gauss_mixture_2d,
    nll_student_t_2d, nll_student_t_mixture_2d, wta_nll_student_t_mixture_2d
)

def ade_l2(pred, gt): return torch.linalg.vector_norm(pred-gt, dim=-1).mean()
def fde_l2(pred, gt): return torch.linalg.vector_norm(pred[:,-1]-gt[:,-1], dim=-1).mean()
def joint_ade(pred, gt): return torch.linalg.vector_norm(pred-gt, dim=-1).mean(dim=1).mean()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_jsonl", required=True)
    ap.add_argument("--val_jsonl",   required=True)
    ap.add_argument("--save_dir", default="output/weights")
    ap.add_argument("--past_len", type=int, default=8)
    ap.add_argument("--future_len", type=int, default=60)
    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--head", choices=["gauss","student_t","mdn_gauss","mdn_student_t"], default="student_t")
    ap.add_argument("--mdn_k", type=int, default=5)
    ap.add_argument("--dof_init", type=float, default=3.0)
    ap.add_argument("--learnable_dof", action="store_true")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--lambda_ade", type=float, default=0.2)
    ap.add_argument("--lambda_fde", type=float, default=0.5)
    ap.add_argument("--lambda_joint", type=float, default=0.2)
    ap.add_argument("--mdn_mode", choices=["nll","wta"], default="wta")
    args = ap.parse_args()

    device = torch.device(args.device if (torch.cuda.is_available() or "cpu" not in args.device) else "cpu")
    ensure_dir(args.save_dir)
    ds_tr, ds_va, dl_tr, dl_va = build_loaders(args.train_jsonl, args.val_jsonl, args.batch_size, 4, True, False)

    model = LSTMForecaster(args.past_len, args.future_len, d_input=4, d_model=args.d_model,
                           head=args.head, mdn_K=args.mdn_k, student_t_dof_init=args.dof_init,
                           learnable_dof=args.learnable_dof).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scaler = GradScaler(enabled=args.amp)

    best = math.inf
    for ep in range(1, args.epochs+1):
        model.train(); t0=time.time(); tot=0.0
        for b in dl_tr:
            past = b["past_xy"].to(device); fut = b["future_xy"].to(device)
            opt.zero_grad(set_to_none=True)
            with autocast(enabled=args.amp):
                out = model(past)
                if args.head == "gauss":
                    ll = nll_gauss_2d(out["mu"], out["log_std"], fut, reduction="mean"); pred=out["mu"]
                elif args.head == "mdn_gauss":
                    ll = (wta_nll_gauss_mixture_2d if args.mdn_mode=="wta" else nll_gauss_mixture_2d)(
                        out["logits"], out["mu"], out["log_std"], fut, reduction="mean")
                    pred = out["mu"].mean(dim=2)
                elif args.head == "student_t":
                    ll = nll_student_t_2d(out["mu"], out["log_scale"], out["dof"], fut, reduction="mean"); pred=out["mu"]
                else:
                    ll = (wta_nll_student_t_mixture_2d if args.mdn_mode=="wta" else nll_student_t_mixture_2d)(
                        out["logits"], out["mu"], out["log_scale"], out["dof"], fut, reduction="mean")
                    pred = out["mu"].mean(dim=2)
                loss = ll + 0.2*ade_l2(pred,fut) + 0.5*fde_l2(pred,fut) + 0.2*joint_ade(pred,fut)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            tot += float(loss)
        # val
        model.eval()
        with torch.no_grad():
            vl=0.0; va=0.0; vf=0.0
            for b in dl_va:
                past=b["past_xy"].to(device); fut=b["future_xy"].to(device)
                out=model(past)
                if args.head=="gauss":
                    ll=nll_gauss_2d(out["mu"],out["log_std"],fut,"mean"); pred=out["mu"]
                elif args.head=="mdn_gauss":
                    fn=wta_nll_gauss_mixture_2d if args.mdn_mode=="wta" else nll_gauss_mixture_2d
                    ll=fn(out["logits"],out["mu"],out["log_std"],fut,"mean"); pred=out["mu"].mean(dim=2)
                elif args.head=="student_t":
                    ll=nll_student_t_2d(out["mu"],out["log_scale"],out["dof"],fut,"mean"); pred=out["mu"]
                else:
                    fn=wta_nll_student_t_mixture_2d if args.mdn_mode=="wta" else nll_student_t_mixture_2d
                    ll=fn(out["logits"],out["mu"],out["log_scale"],out["dof"],fut,"mean"); pred=out["mu"].mean(dim=2)
                vl+=float(ll); va+=float(ade_l2(pred,fut)); vf+=float(fde_l2(pred,fut))
            vl/=len(dl_va); va/=len(dl_va); vf/=len(dl_va)
        dt=time.time()-t0
        print(f"[E{ep:03d}] train {tot/len(dl_tr):.4f} | val ll {vl:.4f} ade {va:.3f} fde {vf:.3f} | {dt:.1f}s")
        if vl<best:
            best=vl
            p=Path(args.save_dir)/f"lstm_{args.head}_{args.mdn_mode}_best.pt"
            torch.save({"model":model.state_dict(),"args":vars(args)}, p)
            print("  ↳ saved:", p)

if __name__=="__main__":
    main()
