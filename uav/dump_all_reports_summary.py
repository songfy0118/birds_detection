# uav/dump_all_reports_summary.py
"""
扫描 output 下所有 report_eval_model.json，
把每个实验的路径 + ADE / FDE / NLL 系列指标都打印出来。

你只要运行一次，把控制台输出完整复制给我，我来帮你做主结果表。
"""

import json
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output"

# 可能存放实验结果的目录
REPORT_DIRS = [
    OUTPUT / "reports",
    OUTPUT / "reports_base",
    OUTPUT / "reports_base1",
    OUTPUT / "reports_stab",
]

# 希望抓出来的指标 key
TARGET_KEYS = {
    "ADE",
    "ADE@30", "ADE@60",
    "FDE",
    "FDE@30", "FDE@60",
    "NLL",
    "JADE", "JFDE",
}

def collect_keys(obj: Any) -> Dict[str, float]:
    """
    在一个嵌套 dict/list 里递归搜索所有 key in TARGET_KEYS 的 value（如果是数值）。
    返回一个 key -> value 的字典。
    """
    found: Dict[str, float] = {}

    if isinstance(obj, dict):
        for k, v in obj.items():
            key = str(k).strip()
            if key in TARGET_KEYS and isinstance(v, (int, float)):
                found[key] = float(v)
            # 递归继续往里找
            sub = collect_keys(v)
            found.update(sub)
    elif isinstance(obj, list):
        for v in obj:
            sub = collect_keys(v)
            found.update(sub)

    return found


def dump_one_report(path: Path):
    try:
        data = json.loads(path.read_text())
    except Exception as e:
        print(f"  [ERR] 读取失败: {e}")
        return

    metrics = collect_keys(data)
    rel = path.relative_to(ROOT)
    print("=" * 80)
    print(f"[REPORT] {rel}")
    if not metrics:
        print("  [WARN] 未找到 ADE/FDE/NLL 相关 key")
        return

    ordered_keys = [
        "ADE", "ADE@30", "ADE@60",
        "FDE", "FDE@30", "FDE@60",
        "NLL",
        "JADE", "JFDE",
    ]
    for k in ordered_keys:
        if k in metrics:
            print(f"  {k:8s} = {metrics[k]:.6f}")


def main():
    print(f"[ROOT] {ROOT}")
    print(f"[SCAN] Output dir: {OUTPUT}")
    print()

    # 扫描指定目录
    for d in REPORT_DIRS:
        if not d.exists():
            continue
        print(f"----- 扫描目录: {d.relative_to(ROOT)} -----")
        for path in sorted(d.rglob("report_eval_model.json")):
            dump_one_report(path)
        print()

    # 再扫根目录下是否有独立的 report_eval_model.json
    root_report = OUTPUT / "report_eval_model.json"
    if root_report.exists():
        print("----- 根目录 report_eval_model.json -----")
        dump_one_report(root_report)

    print("\n[OK] 扫描结束。请把上面所有输出复制给我。")


if __name__ == "__main__":
    main()
