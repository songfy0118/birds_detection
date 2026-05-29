import argparse
import json
from pathlib import Path
import numpy as np


def load_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def to_xy(arr):
    """arr: list of [x,y] or list of dict with x/y"""
    if len(arr) == 0:
        return None
    if isinstance(arr[0], dict):
        return np.array([[p["x"], p["y"]] for p in arr], dtype=np.float32)
    return np.array(arr, dtype=np.float32)


def dump_split(jsonl_path: Path, out_npz: Path, past_len=8, future_len=60):
    obs_list = []
    pred_list = []

    for rec in load_jsonl(jsonl_path):
        # 你的数据里字段叫 "past"（截图可见），future 常见叫 "future"
        # 兼容几种命名，避免你不同版本字段名不一致
        past_key = "past" if "past" in rec else ("past_xy" if "past_xy" in rec else None)
        fut_key = "future" if "future" in rec else ("future_xy" if "future_xy" in rec else None)

        if past_key is None or fut_key is None:
            raise KeyError(
                f"{jsonl_path} missing keys. Found keys: {list(rec.keys())[:20]}. "
                f"Need past and future (or past_xy/future_xy)."
            )

        past = to_xy(rec[past_key])       # [T_obs, 2]
        future = to_xy(rec[fut_key])      # [T_pred, 2]

        if past.shape[0] < past_len or future.shape[0] < future_len:
            # 跳过长度不够的样本（一般不会发生）
            continue

        past = past[:past_len]
        future = future[:future_len]

        # HiVT 输入我们先用单目标轨迹：Nmax=1
        obs_list.append(past[None, :, None, :])     # [1, 8, 1, 2]
        pred_list.append(future[None, :, None, :])  # [1, 60, 1, 2]

    if len(obs_list) == 0:
        raise RuntimeError(f"No valid samples dumped from {jsonl_path}")

    obs = np.concatenate(obs_list, axis=0).astype(np.float32)   # [N, 8, 1, 2]
    pred = np.concatenate(pred_list, axis=0).astype(np.float32) # [N, 60, 1, 2]
    mask = np.ones((obs.shape[0], obs.shape[2]), dtype=np.float32)  # [N, 1]

    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_npz, obs=obs, pred=pred, mask=mask)
    print(f"Saved: {out_npz}")
    print(f"  obs : {obs.shape}")
    print(f"  pred: {pred.shape}")
    print(f"  mask: {mask.shape}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_jsonl", required=True)
    ap.add_argument("--val_jsonl", required=True)
    ap.add_argument("--test_jsonl", required=True)
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--past_len", type=int, default=8)
    ap.add_argument("--future_len", type=int, default=60)
    args = ap.parse_args()

    out_root = Path(args.out_root)
    dump_split(Path(args.train_jsonl), out_root / "train.npz", args.past_len, args.future_len)
    dump_split(Path(args.val_jsonl), out_root / "val.npz", args.past_len, args.future_len)
    dump_split(Path(args.test_jsonl), out_root / "test.npz", args.past_len, args.future_len)


if __name__ == "__main__":
    main()
