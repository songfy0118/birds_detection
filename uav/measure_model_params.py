import sys
from pathlib import Path

# 让 Python 找到 day1/models
sys.path.append(str(Path(__file__).resolve().parents[1]))

import torch
from models.transformer import TrajTransformer
from models.lstm import LSTMModel


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    past = 8
    fut = 60

    print("===== PARAM COUNT =====")

    # ---- LSTM ----
    lstm = LSTMModel(
        past_len=past,
        future_len=fut,
        input_dim=2,
        hidden_dim=128,
        num_layers=2
    )
    lstm_params = count_params(lstm)
    print(f"LSTM params: {lstm_params/1e6:.3f} M")

    # ---- Transformer ----
    trans = TrajTransformer(
        past_len=past,
        future_len=fut,
        d_model=256,
        nhead=8,
        num_layers=4,
        head_type="gauss",
        mdn_k=5
    )
    trans_params = count_params(trans)
    print(f"Transformer params: {trans_params/1e6:.3f} M")


if __name__ == "__main__":
    main()
