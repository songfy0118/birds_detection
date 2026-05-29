# tracking/mot_to_sequences.py
# ------------------------------------------------------------
# 将（稳像后或原始）MOT 轨迹切成 past/future 序列并保存为 jsonl
# - 自动生成 train/val/test 三份
# - 过滤短轨迹，保证帧连续
# - 每条样本包含：past/future（像素坐标）、W/H、video、track_id、agent_type
# - 若存在 <video>_track_meta.json，读取track的主类别映射成 {bird,drone,unknown}
# ------------------------------------------------------------

from pathlib import Path
import argparse, json
import numpy as np
import pandas as pd

COLS = ["frame","id","x","y","w","h","conf","x3","y3","z3"]

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def read_mot(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, header=None, names=COLS)
    return df.astype({"frame":int, "id":int, "x":float, "y":float, "w":float, "h":float, "conf":float})

def read_sizes(path: Path) -> dict[int, tuple[int,int]]:
    m = {}
    with open(path, "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            fr, W, H = line.strip().split(",")
            m[int(fr)] = (int(W), int(H))
    return m

def load_agent_type(meta_json: Path, tid: int) -> str:
    if not meta_json.exists():
        return "unknown"
    meta = json.loads(meta_json.read_text(encoding="utf-8"))
    major = meta.get("track_majority_class", {})
    if str(tid) in major:
        name = major[str(tid)].get("cls_name","unknown").lower()
    elif tid in major:
        name = major[tid].get("cls_name","unknown").lower()
    else:
        return "unknown"
    if "drone" in name or "uav" in name:
        return "drone"
    if "bird" in name:
        return "bird"
    return "unknown"

def slice_track(frames: list[int],
                cxy: np.ndarray,      # [L,2]
                wh_list: list[tuple[int,int]],
                past_len: int, future_len: int, stride: int,
                video: str, tid: int, agent_type: str):
    """从单轨迹切样本，要求帧连续。"""
    seqs = []
    # 检查连续
    fr_arr = np.array(frames, dtype=int)
    gaps = np.where(np.diff(fr_arr) != 1)[0]
    if len(gaps) > 0:
        # 分段递归
        start = 0
        for g in gaps:
            part = slice_track(list(fr_arr[start:g+1]),
                               cxy[start:g+1], wh_list[start:g+1],
                               past_len, future_len, stride, video, tid, agent_type)
            seqs.extend(part)
            start = g+1
        part = slice_track(list(fr_arr[start:]),
                           cxy[start:], wh_list[start:],
                           past_len, future_len, stride, video, tid, agent_type)
        seqs.extend(part)
        return seqs

    L = len(frames)
    for i in range(past_len-1, L - future_len, stride):
        past   = cxy[i-past_len+1 : i+1]         # [P,2]
        future = cxy[i+1 : i+future_len+1]       # [F,2]
        W,H    = wh_list[i]

        seqs.append({
            "video": video,
            "track_id": int(tid),
            "agent_type": agent_type,
            "past": past.tolist(),
            "future": future.tolist(),
            "W": int(W), "H": int(H)
        })
    return seqs

def split_three(seqs: list[dict], ratios=(0.8,0.1,0.1)):
    n = len(seqs)
    n_train = int(n * ratios[0])
    n_val   = int(n * ratios[1])
    return seqs[:n_train], seqs[n_train:n_train+n_val], seqs[n_train+n_val:]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mot_root", required=True, help="MOT目录（可用稳像后的 *_stab.txt）")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--past_len", type=int, default=8)
    ap.add_argument("--future_len", type=int, default=60)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--min_track_len", type=int, default=80, help="最小轨迹长度（帧）")
    ap.add_argument("--split", type=float, nargs=3, default=[0.8,0.1,0.1], help="train/val/test 比例")
    args = ap.parse_args()

    mot_root = Path(args.mot_root)
    out_dir  = Path(args.out_dir)
    ensure_dir(out_dir)

    all_samples = []
    for mot in sorted(mot_root.glob("*.txt")):
        stem = mot.stem
        sizes = mot.with_name(f"{stem}_sizes.csv")
        if not sizes.exists():
            print(f"[WARN] Sizes not found for {stem}, skip.")
            continue
        meta_json = mot.with_name(f"{stem.replace('_stab','')}_track_meta.json")

        df = read_mot(mot)
        # 按轨迹聚合
        for tid, g in df.groupby("id"):
            # 过滤短轨迹
            if len(g) < args.min_track_len:
                continue
            g = g.sort_values("frame")
            frames = g["frame"].astype(int).tolist()
            # center
            cx = g["x"].to_numpy(float) + g["w"].to_numpy(float)/2.0
            cy = g["y"].to_numpy(float) + g["h"].to_numpy(float)/2.0
            cxy = np.stack([cx,cy], axis=1)   # [L,2]

            size_map = read_sizes(sizes)
            wh_list = [size_map.get(int(f), (int(g["x"].iloc[0]+g["w"].iloc[0]), int(g["y"].iloc[0]+g["h"].iloc[0]))) for f in frames]

            agent_type = load_agent_type(meta_json, int(tid))
            seqs = slice_track(frames, cxy, wh_list,
                               args.past_len, args.future_len, args.stride,
                               video=stem, tid=int(tid), agent_type=agent_type)
            all_samples.extend(seqs)

    # 简单打乱
    rng = np.random.default_rng(2025)
    rng.shuffle(all_samples)

    tr, va, te = split_three(all_samples, ratios=tuple(args.split))
    for name, data in [("train.jsonl", tr), ("val.jsonl", va), ("test.jsonl", te)]:
        p = out_dir / name
        with open(p, "w", encoding="utf-8") as f:
            for s in data: f.write(json.dumps(s)+"\n")
        print(f"[OK] {name}: {len(data)}")

    print(f"Total samples: {len(all_samples)} -> {out_dir}")


if __name__ == "__main__":
    main()

"""
# 运行示例（Windows）
# 使用稳像后的 MOT 目录
python tracking/mot_to_sequences.py ^
  --mot_root output\mot_stab_fbd ^
  --out_dir  output\jsonl_fbd ^
  --past_len 8 --future_len 60 --stride 1 --min_track_len 80 ^
  --split 0.8 0.1 0.1
"""
