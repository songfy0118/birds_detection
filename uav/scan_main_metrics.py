# uav/scan_main_metrics.py
#
# 扫描 output/aaafinal_output（如果存在）或者整个 output 目录，
# 自动找出所有 report_eval_model.json，
# 提取 ADE / FDE / NLL / JADE / JFDE，
# 打印汇总，并导出一个 main_metrics.tsv 方便我帮你做主结果表。

import json
from pathlib import Path


def find_metrics_dict(d):
    """在 JSON 里找到真正存 ADE/FDE 的那一层 dict."""
    if not isinstance(d, dict):
        return None

    # 情况 1：顶层就有 ADE/FDE
    if any(k.upper().startswith("ADE") or k.upper().startswith("FDE") for k in d.keys()):
        return d

    # 情况 2：常见结构：{"metrics": {...}}
    m = d.get("metrics")
    if isinstance(m, dict) and any(
        k.upper().startswith("ADE") or k.upper().startswith("FDE") for k in m.keys()
    ):
        return m

    return None


def extract_metrics(path: Path):
    """从单个 report_eval_model.json 中抽取关键指标."""
    try:
        data = json.loads(path.read_text())
    except Exception as e:
        print(f"[WARN] 读取 {path} 失败: {e}")
        return None

    m = find_metrics_dict(data)
    if m is None:
        return None

    # 小工具：找到第一个以某前缀开头的 key（比如 ADE, ADE@30 都算）
    def get_first(prefix):
        prefix = prefix.upper()
        for k, v in m.items():
            if k.upper().startswith(prefix):
                try:
                    return float(v)
                except Exception:
                    return None
        return None

    ade = get_first("ADE")
    fde = get_first("FDE")

    def get_exact(keys):
        for k in keys:
            if k in m:
                try:
                    return float(m[k])
                except Exception:
                    return None
            if k.lower() in m:
                try:
                    return float(m[k.lower()])
                except Exception:
                    return None
        return None

    nll = get_exact(["NLL"])
    jade = get_exact(["JADE"])
    jfde = get_exact(["JFDE"])

    return {
        "ADE": ade,
        "FDE": fde,
        "NLL": nll,
        "JADE": jade,
        "JFDE": jfde,
    }


def main():
    # 优先扫你整理过的 aaafinal_output，如果没有就扫整个 output
    base = Path("output") / "aaafinal_output"
    if not base.exists():
        base = Path("output")
    print(f"[INFO] 扫描目录: {base.resolve()}")

    rows = []

    for jp in base.rglob("report_eval_model.json"):
        rel = jp.relative_to(base)
        metrics = extract_metrics(jp)
        if metrics is None:
            print(f"[WARN] {rel} 里没找到 ADE/FDE 相关字段，跳过。")
            continue
        rows.append((rel, metrics))

    if not rows:
        print("[WARN] 没有发现带 ADE/FDE 的 report_eval_model.json，请确认 output 目录结构。")
        return

    # 控制台里先打印一版摘要，方便你快速看
    header = ["Tag", "ADE", "FDE", "NLL", "JADE", "JFDE"]
    print("\n[SUMMARY] 主结果汇总")
    print("\t".join(header))

    for rel, m in sorted(rows, key=lambda x: str(x[0])):
        vals = [str(rel)]
        for k in header[1:]:
            v = m.get(k)
            vals.append("-" if v is None else f"{v:.4f}")
        print("\t".join(vals))

    # 同时写入 TSV 文件，方便你上传 / 我做 LaTeX 表格
    out_tsv = base / "main_metrics.tsv"
    with out_tsv.open("w", encoding="utf-8") as f:
        f.write("\t".join(header) + "\n")
        for rel, m in sorted(rows, key=lambda x: str(x[0])):
            vals = [str(rel)]
            for k in header[1:]:
                v = m.get(k)
                vals.append("" if v is None else f"{v:.6f}")
            f.write("\t".join(vals) + "\n")

    print(f"\n[OK] 已写出汇总表: {out_tsv}")


if __name__ == "__main__":
    main()
