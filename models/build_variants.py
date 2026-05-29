# models/build_variants.py
from pathlib import Path
import torch

from models.transformer import TransformerForecaster
from models.lstm import LSTMForecaster


def build_transf_gauss(device="cuda:0"):
    model = TransformerForecaster(
        past_len=8,
        future_len=60,
        d_model=256,
        nhead=8,
        enc_layers=4,
        dec_layers=4,
        head="gauss",
        mdn_K=5,
    ).to(device)
    ckpt = torch.load("output/weights/trans_gauss.pt", map_location=device)
    sd = ckpt.get("model", ckpt)
    model.load_state_dict(sd, strict=False)
    model.eval()
    return model


def build_transf_mdn_gauss(device="cuda:0"):
    model = TransformerForecaster(
        past_len=8,
        future_len=60,
        d_model=256,
        nhead=8,
        enc_layers=4,
        dec_layers=4,
        head="mdn_gauss",
        mdn_K=5,
    ).to(device)
    ckpt = torch.load("output/weights/trans_mdn_gauss.pt", map_location=device)
    sd = ckpt.get("model", ckpt)
    model.load_state_dict(sd, strict=False)
    model.eval()
    return model


def build_transf_mdn_student_t(device="cuda:0"):
    model = TransformerForecaster(
        past_len=8,
        future_len=60,
        d_model=256,
        nhead=8,
        enc_layers=4,
        dec_layers=4,
        head="mdn_student_t",
        mdn_K=5,
    ).to(device)
    ckpt = torch.load("output/weights/trans_mdn_student_t_wta_best.pt", map_location=device)
    sd = ckpt.get("model", ckpt)
    model.load_state_dict(sd, strict=False)
    model.eval()
    return model


def build_lstm_student_t(device="cuda:0"):
    model = LSTMForecaster(
        past_len=8,
        future_len=60,
        d_input=4,
        d_model=256,
        num_layers=1,
        head="student_t",
        mdn_K=5,
    ).to(device)
    ckpt = torch.load("output/weights/lstm_student_t.pt", map_location=device)
    sd = ckpt.get("model", ckpt)
    model.load_state_dict(sd, strict=False)
    model.eval()
    return model
