# uav/dump_main_metrics.py
import argparse
import json
from pathlib import Path


def extract_metrics(d):
    """
    从 eval json 里安全地取出 ADE@30 / ADE@60 / FDE@60
    """
    def get_any(keys, default=None):
        for k in keys:
            if k in d:
                return float(d[k])
        return default

    ade30 = get_any(["ADE@30", "ADE30", "ADE_30"])
    ade60 = get_any(["ADE@60", "ADE60", "ADE_60", "ADE"])
    fde60 = get_any(["FDE@60", "FDE60", "FDE_60", "FDE"])

    return ade30, ade60, fde60


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True, help="方法名字，比如 Ours / LSTM / CV 等")
    parser.add_argument("--json", required=True, help="report_eval_model.json 的路径")
    args = parser.parse_args()

    path = Path(args.json)
    if not path.is_file():
        raise FileNotFoundError(f"JSON not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    ade30, ade60, fde60 = extract_metrics(data)

    print(f"[{args.name}]")
    print(f"  ADE@30 = {ade30:.6f}")
    print(f"  ADE@60 = {ade60:.6f}")
    print(f"  FDE@60 = {fde60:.6f}")
    print()


if __name__ == "__main__":
    main()
