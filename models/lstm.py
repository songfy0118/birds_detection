# models/lstm.py
# ------------------------------------------------------------
# LSTM baseline forecaster
# Supports same heads as Transformer:
#   - GaussianHead2D
#   - MDNGaussianHead2D
#   - StudentTHead2D
#   - MDNStudentTHead2D
# ------------------------------------------------------------

from __future__ import annotations
import torch
import torch.nn as nn
from typing import Dict

from models.heads.gaussian import GaussianHead2D, MDNGaussianHead2D
from models.heads.student_t import StudentTHead2D, MDNStudentTHead2D


class LSTMForecaster(nn.Module):
    def __init__(
        self,
        past_len: int,
        future_len: int,
        d_input: int = 4,         # (x,y,vx,vy)
        d_model: int = 256,
        num_layers: int = 1,
        dropout: float = 0.1,
        head: str = "student_t",  # gauss, student_t, mdn_gauss, mdn_student_t
        mdn_K: int = 5,
        student_t_dof_init: float = 3.0,
        learnable_dof: bool = True
    ):
        super().__init__()
        self.P = past_len
        self.M = future_len
        self.d_model = d_model
        self.head_type = head
        self.mdn_K = mdn_K

        # ------------- Encoder -------------
        self.in_proj = nn.Linear(d_input, d_model)
        self.encoder = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )

        # ------------- Decoder query -------------
        self.query = nn.Parameter(torch.zeros(1, future_len, d_model))
        nn.init.normal_(self.query, std=0.02)

        # ------------- Decoder (simple GRU) -------------
        self.decoder = nn.GRU(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=1,
            batch_first=True
        )

        # ------------- Heads -------------
        if head == "gauss":
            self.head = GaussianHead2D(d_model, hidden=2*d_model)
        elif head == "mdn_gauss":
            self.head = MDNGaussianHead2D(d_model, K=mdn_K, hidden=2*d_model)
        elif head == "student_t":
            self.head = StudentTHead2D(
                d_model, hidden=2*d_model,
                dof_init=student_t_dof_init,
                learnable_dof=learnable_dof
            )
        elif head == "mdn_student_t":
            self.head = MDNStudentTHead2D(
                d_model, K=mdn_K, hidden=2*d_model,
                dof_init=student_t_dof_init,
                learnable_dof=learnable_dof
            )
        else:
            raise ValueError(f"Unknown head: {head}")

    @staticmethod
    def to_vel(xy: torch.Tensor) -> torch.Tensor:
        # Compute velocity
        v = torch.zeros_like(xy)
        v[:, 1:] = xy[:, 1:] - xy[:, :-1]
        return v

    def forward(self, past_xy: torch.Tensor) -> Dict:
        """
        past_xy: [B,P,2] (normalized or pixel)
        """

        # ------------- Input prep -------------
        v = self.to_vel(past_xy)
        x = torch.cat([past_xy, v], dim=-1)   # [B,P,4]

        x = self.in_proj(x)
        enc_out, (h, c) = self.encoder(x)

        # ------------- Decoder -------------
        ctx = enc_out.mean(dim=1, keepdim=True)       # [B,1,D]
        q = self.query.expand(past_xy.size(0), -1, -1) + ctx
        dec_out, _ = self.decoder(q)

        # ------------- Head -------------
        out = self.head(dec_out)

        # unify interface
        if self.head_type == "gauss":
            return {"type": "gauss", **out}
        elif self.head_type == "mdn_gauss":
            return {"type": "mdn_gauss", **out}
        elif self.head_type == "student_t":
            return {"type": "student_t", **out}
        elif self.head_type == "mdn_student_t":
            return {"type": "mdn_student_t", **out}
        else:
            raise ValueError(self.head_type)
