# eval_noise.py (Debug修复版)
import torch
import numpy as np
import sys
from pathlib import Path
import os
import json

# 路径设置
ROOT = Path(__file__).resolve().parents[0]
sys.path.insert(0, str(ROOT))

try:
    from utils.data import build_loaders
    from models.transformer import TransformerForecaster
except ImportError:
    print("Error: Could not import utils.data or models.transformer.")
    sys.exit(1)


def add_noise(x, sigma):
    if sigma == 0: return x
    noise = torch.randn_like(x) * sigma
    return x + noise


def evaluate_noise(jsonl_path, weights_path, head_type, args, noise_levels, label):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # --- Debug: 检查文件是否存在 ---
    if not os.path.exists(jsonl_path):
        print(f"ERROR: 找不到数据文件: {os.path.abspath(jsonl_path)}")
        return {}
    if not os.path.exists(weights_path):
        print(f"ERROR: 找不到权重文件: {os.path.abspath(weights_path)}")
        return {}

    # 动态调整参数
    is_mini = "mini" in weights_path
    d_model = 96 if is_mini else 256
    layers = 2 if is_mini else 4
    nhead = 4 if is_mini else 8

    print(f"\n--- Evaluating {label} ---")
    print(f"Config: d_model={d_model}, layers={layers}, head={head_type}")
    print(f"Weights: {weights_path}")

    # 加载模型
    model = TransformerForecaster(
        past_len=args.past_len, future_len=args.future_len,
        d_model=d_model, nhead=nhead,
        enc_layers=layers, dec_layers=layers,
        dropout=args.dropout, head=head_type
    ).to(device)

    checkpoint = torch.load(weights_path, map_location=device)
    if "model" in checkpoint:
        state_dict = {k.replace('module.', ''): v for k, v in checkpoint["model"].items()}
    else:
        state_dict = {k.replace('module.', ''): v for k, v in checkpoint.items()}

    model.load_state_dict(state_dict, strict=False)
    model.eval()

    # 加载数据
    # 注意：build_loaders 返回 (ds_tr, ds_va, dl_tr, dl_va)
    # 我们把 jsonl_path 传给第一个参数，所以用 dl_tr (第三个返回值)
    ds_test, _, dl_test, _ = build_loaders(
        jsonl_path, jsonl_path,
        batch_size=args.batch_size, num_workers=0,  # Windows 设为 0
        normalize=True, return_pixels=False
    )

    print(f"Dataset Loaded: {len(ds_test)} samples found.")
    if len(ds_test) == 0:
        print("FATAL ERROR: 数据集为空！请检查 jsonl 文件内容。")
        return {}

    all_results = {}

    for sigma in noise_levels:
        ade_sum, count = 0.0, 0
        for batch in dl_test:
            past_xy = batch["past_xy"].to(device)
            future = batch["future_xy"].to(device)
            past_wh = batch["past_wh"].to(device)

            with torch.no_grad():
                noisy_inputs = add_noise(past_xy, sigma)
                out = model(noisy_inputs, past_wh=past_wh, return_hidden=False)

                if 'logits' in out:
                    pred = out["mu"].mean(dim=2)
                else:
                    pred = out["mu"]

                error = torch.linalg.vector_norm(pred - future, dim=-1).mean()
                ade_sum += error.item() * len(past_xy)
                count += len(past_xy)

        if count == 0:
            print("Warning: Loop finished but count is 0 (DataLoader issue?)")
            avg_ade = 0.0
        else:
            avg_ade = ade_sum / count

        all_results[f"{sigma}"] = avg_ade
        print(f"Noise Sigma {sigma}: ADE = {avg_ade:.4f}")

    return all_results


if __name__ == "__main__":
    class Args:
        past_len, future_len = 8, 60
        dropout = 0.1
        batch_size = 256


    args = Args()
    noise_levels = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06]
    TEST_JSONL = "output/jsonl_fbd/test.jsonl"

    # 请确认文件名无误
    MINI_WEIGHTS = "output/weights/trans_mdn_student_t_wta_mini.pt"
    ORIGINAL_WEIGHTS = "output/weights/trans_mdn_student_t_wta_best.pt"

    print(f"Checking files...")
    if not os.path.exists(TEST_JSONL):
        print(f"❌ 找不到测试集: {TEST_JSONL}")
    else:
        print(f"✅ 测试集存在: {TEST_JSONL}")

    # 1. 跑 Mini 模型
    mini_res = evaluate_noise(TEST_JSONL, MINI_WEIGHTS, "mdn_student_t", args, noise_levels, "Mini-Transformer")

    # 2. 跑 Original Best 模型
    orig_res = evaluate_noise(TEST_JSONL, ORIGINAL_WEIGHTS, "mdn_student_t", args, noise_levels, "Original-Transformer")

    # 打印结果
    final_res = {"Mini": mini_res, "Original": orig_res}
    print("\n--- Final Noise Robustness Results (JSON) ---")
    print(json.dumps(final_res, indent=4))