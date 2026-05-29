# eval_noise_direct.py (不依赖 DataLoader 的终极版)
import torch
import json
import sys
import os
from pathlib import Path

# 路径设置
ROOT = Path(__file__).resolve().parents[0]
sys.path.insert(0, str(ROOT))

try:
    from models.transformer import TransformerForecaster
except ImportError:
    print("Error: Could not import models.transformer")
    sys.exit(1)


def add_noise(x, sigma):
    if sigma == 0: return x
    noise = torch.randn_like(x) * sigma
    return x + noise


def load_raw_jsonl(path, limit=100):
    """直接读取 jsonl 文件，不经过 dataset/dataloader"""
    data = []
    with open(path, 'r') as f:
        for i, line in enumerate(f):
            if i >= limit: break
            item = json.loads(line)
            # 假设 jsonl 里存的是 list of list
            past = torch.tensor(item['past'], dtype=torch.float32)
            future = torch.tensor(item['future'], dtype=torch.float32)
            # 如果没有 wh，造一个全 0 的
            wh = torch.zeros_like(past)
            data.append((past, future, wh))
    return data


def evaluate_noise_direct():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Running on {device}")

    # 1. 配置
    models_config = {
        "Original-Large": {
            "path": "output/weights/trans_mdn_student_t_wta_best.pt",
            "d_model": 256, "layers": 4, "nhead": 8
        },
        "Mini-Ours": {
            # 注意：用 best_ade
            "path": "output/weights/trans_mdn_student_t_wta_best_ade.pt",
            "d_model": 96, "layers": 2, "nhead": 4
        }
    }

    noise_levels = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06]
    TEST_JSONL = "output/jsonl_fbd/test.jsonl"

    # 2. 直接读取数据 (Load all into memory)
    print(f"Reading {TEST_JSONL} directly...")
    raw_data = load_raw_jsonl(TEST_JSONL, limit=200)
    if len(raw_data) == 0:
        print("Error: No data loaded from jsonl!")
        return

    # 把数据堆叠成一个大 Batch
    # past_xy: [N, 8, 2], future_xy: [N, 60, 2]
    all_past = torch.stack([d[0] for d in raw_data]).to(device)
    all_future = torch.stack([d[1] for d in raw_data]).to(device)
    all_wh = torch.stack([d[2] for d in raw_data]).to(device)

    print(f"Loaded {len(raw_data)} samples. Batch shape: {all_past.shape}")

    results = {}

    for name, cfg in models_config.items():
        print(f"\nEvaluating {name}...")

        if not os.path.exists(cfg["path"]):
            print(f"Skip {name}: File not found {cfg['path']}")
            continue

        # 加载模型
        model = TransformerForecaster(
            past_len=8, future_len=60,
            d_model=cfg["d_model"], nhead=cfg["nhead"],
            enc_layers=cfg["layers"], dec_layers=cfg["layers"],
            head="mdn_student_t"
        ).to(device)

        ckpt = torch.load(cfg["path"], map_location=device)
        state_dict = ckpt["model"] if "model" in ckpt else ckpt
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict, strict=False)
        model.eval()

        res_curve = {}
        for sigma in noise_levels:
            with torch.no_grad():
                # 添加噪声
                noisy_input = add_noise(all_past, sigma)

                # 一次性推理所有数据 (Batch Inference)
                out = model(noisy_input, past_wh=all_wh)

                if 'logits' in out:
                    pred = out["mu"].mean(dim=2)
                else:
                    pred = out["mu"]

                # 计算 ADE
                # pred: [N, 60, 2], all_future: [N, 60, 2]
                err = torch.linalg.vector_norm(pred - all_future, dim=-1).mean()
                ade = err.item()

            res_curve[str(sigma)] = ade
            print(f"  Sigma {sigma}: ADE={ade:.4f}")

        results[name] = res_curve

    print("\n--- Final Robustness Data ---")
    print(json.dumps(results, indent=4))


if __name__ == "__main__":
    evaluate_noise_direct()