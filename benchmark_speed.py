import torch
import time
import numpy as np
from models.lstm import LSTMForecaster
from models.transformer import TransformerForecaster


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def benchmark(model, name, batch_size=1, device='cuda'):
    model.to(device)
    model.eval()

    # 模拟输入: [Batch, Past_Len=8, 2]
    dummy_input = torch.randn(batch_size, 8, 2).to(device)

    # Warmup (预热 GPU)
    for _ in range(10):
        with torch.no_grad():
            _ = model(dummy_input)

    # Test
    start = time.time()
    iters = 100
    for _ in range(iters):
        with torch.no_grad():
            _ = model(dummy_input)
    end = time.time()

    avg_time = (end - start) / iters
    fps = batch_size / avg_time
    params = count_parameters(model) / 1e6  # Million

    print(f"--- {name} ---")
    print(f"Params: {params:.2f} M")
    print(f"Latency (bs={batch_size}): {avg_time * 1000:.2f} ms")
    print(f"FPS: {fps:.1f}")
    print("-" * 20)


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Benchmarking on {device}...")

    # 1. Baseline: LSTM + Gaussian (最轻量)
    lstm = LSTMForecaster(past_len=8, future_len=60, d_model=128, head="gauss")
    benchmark(lstm, "Baseline: LSTM (Gaussian)", batch_size=1, device=device)

    # 2. Ours: Transformer + Student-t (你的主模型)
    # 注意：保持和你训练时的 d_model 一致，假设是 256
    tf = TransformerForecaster(past_len=8, future_len=60, d_model=256, head="mdn_student_t")
    benchmark(tf, "Ours: BirdFormer (Student-t MDN)", batch_size=1, device=device)