# measure_efficiency.py (已修复 LSTM forward 兼容性问题)
import torch
import time
import sys
import os
from pathlib import Path

# 确保能导入 models
ROOT = Path(__file__).resolve().parents[0]
sys.path.append(str(ROOT))
try:
    from models.transformer import TransformerForecaster
    from models.lstm import LSTMForecaster
except ImportError:
    print("Error: Could not import models. Please ensure models/transformer.py and models/lstm.py exist.")
    sys.exit(1)
import numpy as np


def count_parameters(model):
    """计算模型参数量 (M)"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6


def measure_fps(model, device='cuda'):
    """测量 Latency (ms) 和 FPS"""
    model.to(device)
    model.eval()
    dummy_input = torch.randn(1, 8, 2).to(device)
    dummy_wh = torch.zeros_like(dummy_input).to(device)

    # --- 核心修复：兼容 LSTM 的 forward 方法 ---
    # TransformerForecaster 需要 past_wh，LSTMForecaster 不需要
    if isinstance(model, LSTMForecaster):
        forward_kwargs = {}
        print("  -> Using LSTM forward signature (no past_wh)")
    else:
        forward_kwargs = {'past_wh': dummy_wh}
        print("  -> Using Transformer forward signature (with past_wh)")
    # -----------------------------------------------

    # 预热 GPU
    for _ in range(20):
        with torch.no_grad(): _ = model(dummy_input, **forward_kwargs)

    # 测速
    start = time.perf_counter()
    iters = 500
    for _ in range(iters):
        with torch.no_grad(): _ = model(dummy_input, **forward_kwargs)
    torch.cuda.synchronize()
    end = time.perf_counter()

    avg_time = (end - start) / iters
    fps = 1.0 / avg_time
    return avg_time * 1000, fps


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("Warning: CUDA is not available. Running on CPU, which will affect performance results.")

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"Running efficiency benchmark on {device}...\n")

    results = []

    # 1. LSTM Baseline
    # 构造函数已在上次修复，无需 hidden_dim
    lstm = LSTMForecaster(past_len=8, future_len=60, d_model=256, num_layers=1, head="gauss")
    lstm_ms, lstm_fps = measure_fps(lstm, device)
    lstm_params = count_parameters(lstm)
    results.append(("LSTM Baseline", lstm_params, lstm_ms, lstm_fps))

    # 2. BirdFormer (Full)
    tf_full = TransformerForecaster(past_len=8, future_len=60, d_model=256, enc_layers=4, dec_layers=4,
                                    head="mdn_student_t")
    tf_full_ms, tf_full_fps = measure_fps(tf_full, device)
    tf_full_params = count_parameters(tf_full)
    results.append(("BirdFormer (Full)", tf_full_params, tf_full_ms, tf_full_fps))

    # 3. BirdFormer (Mini)
    tf_mini = TransformerForecaster(past_len=8, future_len=60, d_model=96, enc_layers=2, dec_layers=2,
                                    head="mdn_student_t")
    tf_mini_ms, tf_mini_fps = measure_fps(tf_mini, device)
    tf_mini_params = count_parameters(tf_mini)
    results.append(("BirdFormer (Mini)", tf_mini_params, tf_mini_ms, tf_mini_fps))

    print("\n--- Efficiency Results (Q1 Table 4) ---")
    print(f"{'Model':<20} | {'Params (M)':<10} | {'Latency (ms)':<15} | {'FPS':<10}")
    print("-" * 60)
    for name, p, ms, fps in results:
        print(f"{name:<20} | {p:<10.2f} | {ms:<15.2f} | {fps:<10.1f}")