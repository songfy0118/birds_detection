import shutil
from pathlib import Path

# day1 根目录
ROOT = Path(__file__).resolve().parent

# 原始输出目录
OUTPUT = ROOT / "output"

# 备份目录（论文保险库）
BACKUP = ROOT / "aaafinal_output"

# ===== 这里列出“重要输出”名称 =====
# 说明：都是相对于 output/ 的路径名
IMPORTANT_ITEMS = [
    # -------- 1) 轨迹预测主实验：预测 & 报告 --------
    "preds_csv",
    "preds_csv_base1",
    "preds_offline",

    "reports",
    "reports_base",
    "reports_base1",
    "reports_stab",

    "proxy_tracking_quality.json",
    "report_eval_model.json",
    "report_eval_preds.json",

    # -------- 2) 模型权重（避免重新训练） --------
    "weights",
    "weights_base1",

    # -------- 3) 数据中间件（mot/jsonl，可重现实验） --------
    "mot",
    "mot_base1",
    "mot_fbd",
    "mot_gt",
    "mot_stab",
    "mot_stab_base1",
    "mot_stab_fbd",

    "jsonl_base_8_60",
    "jsonl_fbd",
    "jsonl_stab_8_60",

    # -------- 4) UAV / 合成视频相关 --------
    "synth",
    "synth2",
    "synth_100",
    "synth2_100",
    "synth3",
    "synth3_100",
    "synth3_paper",
    "synth3_recall",

    "uav_presence",
    "uav_presence_gdino_synth2",
    "uav_presence_yolo",
    "uav_presence_owlvit.json",

    "yolo_compare",

    "synth2_yolo.json",
    "synth_yolo.json",
    "yolo_strict_orig.json",
    "yolo_strict_synth.json",
    "yolo_synth2_result.json",
    "yolo_synth_result.json",
]


def copy_item(name: str):
    """
    从 output/ 复制一个目录或文件到 aaafinal_output/
    """
    src = OUTPUT / name
    dst = BACKUP / name

    if not src.exists():
        print(f"[SKIP] {name} (not found in output/)")
        return

    if src.is_dir():
        print(f"[COPY] dir  {name}")
        # Python 3.8+ 支持 dirs_exist_ok
        shutil.copytree(src, dst, dirs_exist_ok=True)
    else:
        print(f"[COPY] file {name}")
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def main():
    if not OUTPUT.exists():
        print(f"[ERR] OUTPUT dir not found: {OUTPUT}")
        return

    BACKUP.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Backup from: {OUTPUT}")
    print(f"[INFO] Backup to  : {BACKUP}\n")

    for name in IMPORTANT_ITEMS:
        copy_item(name)

    print("\n[OK] 所有重要输出已复制/更新到 aaafinal_output/。")
    print("    以后有新结果，改一下 IMPORTANT_ITEMS 或直接重跑此脚本即可同步。")


if __name__ == "__main__":
    main()
