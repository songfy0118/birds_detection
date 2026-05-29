import time
import torch
import numpy as np
# 根据你的目录结构导入模型
from models.transformer import TransformerModel
from models.studentT import StudentTMDN


def benchmark():
    # 1. 配置参数 (与你训练时一致)
    D_MODEL = 96
    N_HEAD = 4
    N_LAYER = 2
    PAST_LEN = 8
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"正在测试设备: {DEVICE}")

    # 2. 初始化模型 (随机权重即可，测速不需要真实权重)
    encoder = TransformerModel(input_dim=2, model_dim=D_MODEL, num_heads=N_HEAD, num_layers=N_LAYER,
                               output_dim=D_MODEL).to(DEVICE)
    head = StudentTMDN(hidden_dim=D_MODEL, action_dim=2, num_components=5).to(DEVICE)
    encoder.eval()
    head.eval()

    # 3. 构造假数据 (Batch Size = 1, 模拟实时推理)
    dummy_input = torch.randn(1, PAST_LEN, 2).to(DEVICE)

    # 4. 预热 (Warmup)
    print("正在预热 GPU...")
    with torch.no_grad():
        for _ in range(50):
            _ = head(encoder(dummy_input)[:, -1, :])

    # 5. 正式测速
    iterations = 1000
    print(f"开始测试 {iterations} 次推理...")

    t0 = time.time()
    with torch.no_grad():
        for _ in range(iterations):
            # 模拟完整流程：编码 -> 解码
            features = encoder(dummy_input)
            last_feat = features[:, -1, :]
            _ = head(last_feat)

            # 如果是 GPU，需要同步时间
            if torch.cuda.is_available():
                torch.cuda.synchronize()

    t1 = time.time()

    total_time = t1 - t0
    fps = iterations / total_time
    latency_ms = (total_time / iterations) * 1000

    print("=" * 30)
    print(f"模型配置: Mini-BirdFormer (d={D_MODEL}, L={N_LAYER})")
    print(f"总耗时: {total_time:.4f} 秒")
    print(f"单帧延迟: {latency_ms:.2f} ms")
    print(f"FPS (每秒帧数): {fps:.1f}")
    print("=" * 30)


if __name__ == "__main__":
    benchmark()