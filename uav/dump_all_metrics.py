import json
from pathlib import Path

def flatten_dict(d, parent_key="", sep="."):
    """把嵌套 dict 展平，方便打印"""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def main():
    root = Path(__file__).resolve().parents[1]
    out_root = root / "output"

    print(f"[ROOT] {root}")
    print(f"[SCAN] searching for report_eval_model.json under {out_root}\n")

    # 递归搜所有 report_eval_model.json
    json_paths = sorted(out_root.rglob("report_eval_model.json"))

    if not json_paths:
        print("[WARN] No report_eval_model.json found.")
        return

    for jp in json_paths:
        rel = jp.relative_to(out_root)
        exp_name = rel.parent.as_posix()
        print("=" * 80)
        print(f"[EXP] {exp_name}")
        print(f"[FILE] {jp}")
        try:
            with jp.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"[ERR ] failed to load json: {e}")
            continue

        flat = flatten_dict(data)

        # 优先打印和 ADE / FDE / NLL 相关的 key
        interesting_keys = [k for k in flat.keys()
                            if any(s in k.lower() for s in ["ade", "fde", "nll", "rmse", "mae"])]

        if not interesting_keys:
            print("[INFO] no ADE/FDE/NLL-like keys found, print all scalars:")
            for k, v in flat.items():
                if isinstance(v, (int, float)):
                    print(f"  {k:30s} = {v}")
        else:
            print("[METRICS] ADE / FDE / NLL related keys:")
            for k in sorted(interesting_keys):
                v = flat[k]
                if isinstance(v, (int, float)):
                    print(f"  {k:30s} = {v}")
                else:
                    print(f"  {k:30s} = {v} (non-scalar)")
        print()

if __name__ == "__main__":
    main()
