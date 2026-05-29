# main.py
# ------------------------------------------------------------
# Unified CLI for the Bird Trajectory & Tracking Project
# Subcommands:
#   train-tf         -> train/train_transformer.py
#   train-lstm       -> train/train_lstm.py
#   predict          -> predict/predict.py
#   predict-online   -> predict/predict_online.py
#   eval-preds       -> metrics/eval_predictions.py
#   eval-model       -> metrics/eval_model.py
#   eval_sohota      -> metrics/so_hota.py   (SO-HOTA / 小目标友好)
# Notes:
#   - 所有子进程在项目根目录运行，并设置 PYTHONPATH=项目根，确保 imports 稳定
#   - 这里不做参数二次解析，所有 flags 透明透传给对应脚本
# ------------------------------------------------------------
from __future__ import annotations
import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def run_script(script_rel: str, passthrough_argv: list[str]) -> None:
    """
    以工程根为工作目录运行脚本，并注入 PYTHONPATH=ROOT。
    passthrough_argv 即用户在 main.py 后面的全部参数（去掉子命令名）。
    """
    script_path = ROOT / script_rel
    if not script_path.exists():
        print(f"[ERR] Script not found: {script_path}")
        sys.exit(2)

    # 继承环境，并强制 PYTHONPATH 包含 ROOT
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)

    cmd = [sys.executable, str(script_path), *passthrough_argv]
    print("[RUN]", " ".join(cmd))
    try:
        # 在工程根目录下执行，确保相对路径/导入稳定
        r = subprocess.run(cmd, cwd=str(ROOT), env=env)
    except KeyboardInterrupt:
        print("\n[MAIN] Interrupted by user.")
        sys.exit(130)
    if r.returncode != 0:
        sys.exit(r.returncode)

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Bird Trajectory Forecasting & Tracking — Unified CLI"
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    # ---------------- Train: Transformer ----------------
    sub.add_parser(
        "train-tf",
        help="Train Transformer forecaster (student_t / mdn_student_t recommended)"
    )

    # ---------------- Train: LSTM baseline ----------------
    sub.add_parser(
        "train-lstm",
        help="Train LSTM forecaster baseline"
    )

    # ---------------- Predict (offline) ----------------
    sub.add_parser(
        "predict",
        help="Offline batch prediction on jsonl (export CSV)"
    )

    # ---------------- Predict (online, MOT-based) ----------------
    sub.add_parser(
        "predict-online",
        help="Online prediction given MOT txts (+sizes.csv), optional video rendering"
    )

    # ---------------- Evaluate predictions (CSV) ----------------
    sub.add_parser(
        "eval-preds",
        help="Evaluate CSV predictions produced by predict.py against test.jsonl"
    )

    # ---------------- Evaluate model directly on jsonl ----------------
    sub.add_parser(
        "eval-model",
        help="Load weights & evaluate model directly on jsonl (ADE/FDE/JADE/JFDE)"
    )

    # ---------------- SO-HOTA (tracking) ----------------
    sub.add_parser(
        "eval_sohota",
        help="SO-HOTA (Simplified Soft-HOTA) on MOT txt folders/files"
    )

    return p

def main():
    # 我们不在 main 再解析各子命令的所有 flags，而是透明透传给子脚本
    parser = build_parser()
    # 只解析出子命令名，其余参数保持在 sys.argv 中
    # eg: python main.py train-tf --train_jsonl ... -> cmd_idx=2 后的全部原样传给脚本
    args, _ = parser.parse_known_args()

    # 找到子命令在 sys.argv 的位置，并取其后的所有参数透明传递
    # sys.argv[0] = main.py, sys.argv[1] = <subcommand>, sys.argv[2:] = passthrough
    if len(sys.argv) < 2:
        parser.print_help()
        sys.exit(1)
    subcmd = sys.argv[1]
    passthrough = sys.argv[2:]

    # 分发
    if subcmd == "train-tf":
        run_script("train/train_transformer.py", passthrough)

    elif subcmd == "train-lstm":
        run_script("train/train_lstm.py", passthrough)

    elif subcmd == "predict":
        run_script("predict/predict.py", passthrough)

    elif subcmd == "predict-online":
        run_script("predict/predict_online.py", passthrough)

    elif subcmd == "eval-preds":
        run_script("metrics/eval_predictions.py", passthrough)

    elif subcmd == "eval-model":
        run_script("metrics/eval_model.py", passthrough)

    elif subcmd == "eval_sohota":
        run_script("metrics/so_hota.py", passthrough)

    else:
        print(f"[ERR] Unknown subcommand: {subcmd}")
        parser.print_help()
        sys.exit(2)

if __name__ == "__main__":
    main()
