import argparse
from pathlib import Path
import numpy as np
import pandas as pd
COLS = ["frame","id","x","y","w","h","conf","x3","y3","z3"]

def read_mot(p): return pd.read_csv(p, header=None, names=COLS)

def metrics_for_file(txt):
    df = read_mot(txt)
    g = df.groupby("id")
    smooth, frag, short, jump = [], [], 0, []
    for _, gi in g:
        gi = gi.sort_values("frame")
        if len(gi) < 20: short += 1
        cx = gi["x"].values + gi["w"].values/2
        cy = gi["y"].values + gi["h"].values/2
        vx = np.diff(cx); vy = np.diff(cy)
        ax = np.diff(vx); ay = np.diff(vy)
        if len(ax)>0: smooth.append(np.var(ax)+np.var(ay))
        frames = gi["frame"].values
        cuts = np.sum(np.diff(frames) > 1) + 1
        frag.append(cuts/len(frames))
        step = np.sqrt(vx**2+vy**2); scale = np.maximum(gi["w"].values[:-1],1)
        jump.append(np.mean(step/scale > 0.5))
    return {
        "smooth_var_mean": float(np.mean(smooth)) if smooth else None,
        "frag_ratio_mean": float(np.mean(frag)) if frag else None,
        "short_track_ratio": float(short/max(len(g),1)),
        "big_jump_ratio_mean": float(np.mean(jump)) if jump else None
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mot_dir", required=True)
    ap.add_argument("--out", default="output/proxy_tracking_quality.json")
    args = ap.parse_args()
    res = {}
    for p in sorted(Path(args.mot_dir).glob("*_stab.txt")):
        res[p.stem] = metrics_for_file(p)
    import json, os
    os.makedirs(Path(args.out).parent, exist_ok=True)
    with open(args.out,"w",encoding="utf-8") as f:
        json.dump(res,f,indent=2,ensure_ascii=False)
    print(f"[OK] Proxy tracking metrics saved to {args.out}")

if __name__=="__main__":
    main()
