# metrics/so_hota.py
# ------------------------------------------------------------
# SO-HOTA: Small-Object-friendly HOTA-like metric for MOT outputs.
# Inputs:
#   - --gt_dir    : directory of GT MOT files (*.txt)
#   - --pred_dir  : directory of predicted MOT files (*.txt)
# Format (MOT): each line "frame,id,x,y,w,h,conf,x3,y3,z3"
#   * x,y,w,h in pixels; (x,y) is top-left corner.
#   * "conf" ignored for GT; used for predictions可选过滤。
# Key features:
#   - Supports IoU-based evaluation (tau list), and/or center-distance thresholds
#   - Association via Hungarian (if scipy available) or greedy (fallback)
#   - Outputs: HOTA, DetA, AssA averaged over thresholds; plus per-tau明细
# ------------------------------------------------------------
from __future__ import annotations
from pathlib import Path
import argparse, math, json
from typing import Dict, Tuple, List, Optional
import numpy as np
import pandas as pd

try:
    from scipy.optimize import linear_sum_assignment
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False

COLS = ["frame","id","x","y","w","h","conf","x3","y3","z3"]

# ---------- IO ----------
def _read_mot_file(p: Path) -> pd.DataFrame:
    if not p.exists():
        raise FileNotFoundError(f"MOT file not found: {p}")
    df = pd.read_csv(p, header=None, names=COLS)
    # 类型规范
    for c in ["frame","id"]:
        df[c] = df[c].astype(int)
    for c in ["x","y","w","h","conf"]:
        df[c] = df[c].astype(float)
    return df

def load_mot_dir(dir_path: Path) -> Dict[str, pd.DataFrame]:
    out = {}
    for txt in sorted(dir_path.glob("*.txt")):
        out[txt.stem] = _read_mot_file(txt)
    if not out:
        raise RuntimeError(f"No *.txt found under {dir_path}")
    return out

# ---------- geometry ----------
def iou_xywh(a: np.ndarray, b: np.ndarray) -> float:
    # a,b: [4] as (x,y,w,h)
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, inter_x2 - inter_x1), max(0.0, inter_y2 - inter_y1)
    inter = iw * ih
    if inter <= 0: return 0.0
    area_a = aw * ah
    area_b = bw * bh
    return inter / (area_a + area_b - inter + 1e-12)

def center_distance(a: np.ndarray, b: np.ndarray) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    acx, acy = ax + aw/2.0, ay + ah/2.0
    bcx, bcy = bx + bw/2.0, by + bh/2.0
    dx, dy = acx - bcx, acy - bcy
    return float((dx*dx + dy*dy) ** 0.5)

# ---------- per-frame matching ----------
def match_frame(
    gt_boxes: np.ndarray, gt_ids: np.ndarray,
    pd_boxes: np.ndarray, pd_ids: np.ndarray,
    use_iou: bool, thr: float
) -> Tuple[List[Tuple[int,int]], List[int], List[int]]:
    """
    Return:
      matches: list of (gt_id, pd_id)
      gt_unmatched: list of gt_id
      pd_unmatched: list of pd_id
    """
    G, P = len(gt_boxes), len(pd_boxes)
    if G == 0 and P == 0: return [], [], []
    if G == 0: return [], [], list(pd_ids)
    if P == 0: return [], list(gt_ids), []

    if use_iou:
        # IoU matrix -> cost = 1 - IoU (maximize IoU)
        M = np.zeros((G,P), dtype=np.float32)
        for i in range(G):
            for j in range(P):
                M[i,j] = iou_xywh(gt_boxes[i], pd_boxes[j])
        mask = (M >= thr)
        if not mask.any():
            return [], list(gt_ids), list(pd_ids)
        cost = 1.0 - M
    else:
        # center-distance: match when dist <= thr (pixels)
        M = np.zeros((G,P), dtype=np.float32)
        for i in range(G):
            for j in range(P):
                d = center_distance(gt_boxes[i], pd_boxes[j])
                M[i,j] = d
        mask = (M <= thr)
        if not mask.any():
            return [], list(gt_ids), list(pd_ids)
        cost = M

    # match via Hungarian if available; else greedy
    if _HAS_SCIPY:
        gi, pj = linear_sum_assignment(cost)
        pairs = []
        used_g, used_p = set(), set()
        for i, j in zip(gi, pj):
            if not mask[i,j]: continue
            pairs.append((int(gt_ids[i]), int(pd_ids[j])))
            used_g.add(i); used_p.add(j)
        gt_un = [int(gt_ids[i]) for i in range(G) if i not in used_g]
        pd_un = [int(pd_ids[j]) for j in range(P) if j not in used_p]
        return pairs, gt_un, pd_un
    else:
        # greedy: pick best valid pairs descending by score (IoU) or ascending by distance
        if use_iou:
            # sort by IoU desc
            idxs = np.dstack(np.unravel_index(np.argsort(-M, axis=None), M.shape))[0]
        else:
            # sort by distance asc
            idxs = np.dstack(np.unravel_index(np.argsort(M, axis=None), M.shape))[0]
        used_g = set(); used_p = set(); pairs=[]
        for i,j in idxs:
            if i in used_g or j in used_p: continue
            if not mask[i,j]: continue
            used_g.add(i); used_p.add(j)
            pairs.append((int(gt_ids[i]), int(pd_ids[j])))
        gt_un = [int(gt_ids[i]) for i in range(G) if i not in used_g]
        pd_un = [int(pd_ids[j]) for j in range(P) if j not in used_p]
        return pairs, gt_un, pd_un

# ---------- HOTA components ----------
def det_accuracy(tp: int, fp: int, fn: int) -> float:
    # 与 HOTA 论文一致的 Detection Accuracy:
    # DetA = TP / (TP + 0.5*(FP + FN))
    denom = tp + 0.5*(fp + fn)
    return (tp / denom) if denom > 0 else 0.0

def ass_accuracy(
    frames: List[int],
    matches_by_frame: Dict[int, List[Tuple[int,int]]],
    gt_presence: Dict[int, set],   # frame -> set(gt_id present)
    pd_presence: Dict[int, set]    # frame -> set(pd_id present)
) -> float:
    """
    Association Accuracy（简化高保真版）：
      对每个被匹配过的 (g,p) 对，计算：
        TPA = #frames 两者同时出现且被匹配
        FPA = #frames p 出现但未与 g 匹配（且 g 同帧出现）
        FNA = #frames g 出现但未与 p 匹配（且 p 同帧出现）
      AssA_pair = TPA / (TPA + 0.5*(FPA+FNA))
      AssA = 所有 (g,p) 对的平均
    """
    # 收集所有出现过的 (g,p) 对
    pair_frames = {}  # (g,p) -> set(frames matched)
    for f in frames:
        for g,p in matches_by_frame.get(f, []):
            pair_frames.setdefault((g,p), set()).add(f)

    if not pair_frames: return 0.0
    vals = []
    for (g,p), matched_fs in pair_frames.items():
        # 所有同时出现的帧（g 和 p 都出现）
        joint_fs = [f for f in frames if (g in gt_presence.get(f,set())) and (p in pd_presence.get(f,set()))]
        if not joint_fs:
            continue
        TPA = len(matched_fs)
        # p 有但没和 g 匹配（g 同帧也存在）
        FPA = len([f for f in joint_fs if f not in matched_fs and p in pd_presence.get(f,set()) and g in gt_presence.get(f,set())])
        # g 有但没和 p 匹配（p 同帧也存在）
        FNA = FPA  # 对称近似（在 joint_fs 下，此两者数量相等）
        denom = TPA + 0.5*(FPA + FNA)
        vals.append( (TPA/denom) if denom>0 else 0.0 )
    return float(np.mean(vals)) if vals else 0.0

# ---------- overall evaluation ----------
def evaluate_so_hota(
    gt: Dict[str, pd.DataFrame],
    pred: Dict[str, pd.DataFrame],
    use_iou: bool,
    thr_list: List[float],
    conf_thr: float = 0.0
) -> Dict[str, float]:
    """
    Return dict with:
      - HOTA, DetA, AssA (averaged over thresholds)
      - Per-tau details
    """
    det_vals = []
    ass_vals = []
    per_tau = []

    for thr in thr_list:
        all_tp = all_fp = all_fn = 0
        # association bookkeeping
        matches_by_frame = {}  # frame -> list[(g,p)]
        gt_presence = {}
        pd_presence = {}
        frames_all = []

        for vid, gt_df in gt.items():
            if vid not in pred:
                continue
            pd_df = pred[vid]
            # 可选：按 conf 过滤预测
            if conf_thr > 0:
                pd_df = pd_df[pd_df["conf"] >= conf_thr]

            # 按帧分组
            ggb = dict(list(gt_df.groupby("frame")))
            pgb = dict(list(pd_df.groupby("frame")))
            frames = sorted(set(ggb.keys()) | set(pgb.keys()))
            frames_all.extend(frames)
            for f in frames:
                g = ggb.get(f, pd.DataFrame(columns=COLS))
                p = pgb.get(f, pd.DataFrame(columns=COLS))
                g_boxes = g[["x","y","w","h"]].to_numpy(float)
                p_boxes = p[["x","y","w","h"]].to_numpy(float)
                g_ids   = g["id"].to_numpy(int)
                p_ids   = p["id"].to_numpy(int)

                # 记录出现
                gt_presence.setdefault(f, set()).update(g_ids.tolist())
                pd_presence.setdefault(f, set()).update(p_ids.tolist())

                pairs, g_un, p_un = match_frame(g_boxes, g_ids, p_boxes, p_ids, use_iou, thr)
                tp = len(pairs); fp = len(p_un); fn = len(g_un)
                all_tp += tp; all_fp += fp; all_fn += fn

                if tp > 0:
                    matches_by_frame[f] = pairs

        detA = det_accuracy(all_tp, all_fp, all_fn)
        assA = ass_accuracy(frames_all, matches_by_frame, gt_presence, pd_presence)
        HOTA = math.sqrt(detA * assA)
        det_vals.append(detA); ass_vals.append(assA)
        per_tau.append({"thr": thr, "DetA": detA, "AssA": assA, "HOTA": HOTA})

    out = {
        "HOTA": float(np.mean([x["HOTA"] for x in per_tau]) if per_tau else 0.0),
        "DetA": float(np.mean(det_vals) if det_vals else 0.0),
        "AssA": float(np.mean(ass_vals) if ass_vals else 0.0),
        "details": per_tau
    }
    return out

# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser(description="SO-HOTA evaluation for MOT")
    ap.add_argument("--gt_dir", required=True, help="Directory with GT MOT txts")
    ap.add_argument("--pred_dir", required=True, help="Directory with predicted MOT txts")
    ap.add_argument("--out_json", default="output/reports/so_hota.json")
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--iou", action="store_true", help="Use IoU-based thresholds")
    group.add_argument("--center", action="store_true", help="Use center-distance thresholds (pixels)")
    ap.add_argument("--thr", type=float, nargs="+", default=[0.3, 0.5, 0.7],
                    help="IoU thr list (if --iou) or pixel thresholds (if --center)")
    ap.add_argument("--conf_thr", type=float, default=0.0, help="Filter predictions with conf < conf_thr")
    args = ap.parse_args()

    gt_dir = Path(args.gt_dir); pred_dir = Path(args.pred_dir)
    gt = load_mot_dir(gt_dir)
    pred = load_mot_dir(pred_dir)

    res = evaluate_so_hota(gt, pred, use_iou=args.iou, thr_list=args.thr, conf_thr=args.conf_thr)
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json,"w",encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)

    print(json.dumps(res, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
