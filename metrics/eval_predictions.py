# metrics/eval_predictions.py
# ------------------------------------------------------------
# Evaluate offline predictions written by predict/predict.py
# Inputs:
#   --jsonl  test.jsonl (from tracking/mot_to_sequences.py)
#   --pred_dir  directory of *_pred.csv files (one per video)
# Output:
#   print ADE/FDE & multi-horizon; save metrics.json
# ------------------------------------------------------------
from __future__ import annotations
from pathlib import Path
import argparse, json, csv
import numpy as np
import torch

from metrics.ade_fde import ade_fde, multi_horizon_ade_fde

def read_jsonl(path: Path):
    items=[]
    with open(path,"r",encoding="utf-8") as f:
        for ln in f:
            items.append(json.loads(ln))
    return items

def read_preds_csv(csv_path: Path, future_len: int):
    """
    读取一个视频的 pred.csv，按每 future_len 行切成一个样本 [F,2]
    返回 list[np.ndarray(F,2)]
    """
    rows=[]
    with open(csv_path,"r",encoding="utf-8") as f:
        r=csv.DictReader(f)
        for row in r:
            rows.append((float(row["x"]), float(row["y"])))
    arr=np.array(rows, dtype=np.float32)
    if len(arr)%future_len!=0:
        # 裁掉尾部不满 F 的
        n = (len(arr)//future_len)*future_len
        arr = arr[:n]
    seqs = arr.reshape(-1, future_len, 2)
    return [s for s in seqs]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--pred_dir", required=True)
    ap.add_argument("--future_len", type=int, default=60)
    ap.add_argument("--horizons", type=int, nargs="+", default=[30,60],
                    help="以步为单位，例如 [30 60] 相当于 1.0s/2.0s 若fps=30")
    ap.add_argument("--out", default="output/report_eval_preds.json")
    args = ap.parse_args()

    items = read_jsonl(Path(args.jsonl))
    # 按 video 汇总 GT
    gt_by_video={}
    for it in items:
        v = it.get("video","")
        gt_by_video.setdefault(v, []).append(np.array(it["future"], dtype=np.float32))

    pred_by_video={}
    for csvp in Path(args.pred_dir).glob("*_pred.csv"):
        v = csvp.stem.replace("_pred","")
        pred_by_video[v] = read_preds_csv(csvp, args.future_len)

    # 顺序对齐：对每个视频，按顺序 zip
    preds=[]; gts=[]
    for v, gt_list in gt_by_video.items():
        pred_list = pred_by_video.get(v, [])
        n = min(len(gt_list), len(pred_list))
        if n==0: continue
        gts.append(np.stack(gt_list[:n], 0))
        preds.append(np.stack(pred_list[:n], 0))
    if not preds:
        print("[ERR] No matching predictions and GT found."); return

    gt  = torch.from_numpy(np.concatenate(gts,0))     # [B,F,2]
    prd = torch.from_numpy(np.concatenate(preds,0))   # [B,F,2]

    base = ade_fde(prd, gt)
    mh   = multi_horizon_ade_fde(prd, gt, args.horizons)
    out  = {**base, **mh, "B": int(prd.size(0)), "F": int(prd.size(1))}
    print(json.dumps(out, indent=2, ensure_ascii=False))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out,"w",encoding="utf-8") as f:
        json.dump(out,f,ensure_ascii=False,indent=2)

if __name__ == "__main__":
    main()
