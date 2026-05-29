# models/transformer.py
# ------------------------------------------------------------
# Transformer-based trajectory forecaster for small objects
# ------------------------------------------------------------
from __future__ import annotations
from typing import Optional, Literal, Dict
import math
import torch
import torch.nn as nn

# 直接从子模块导入（不依赖 __init__.py 重导出）
from models.heads.gaussian import GaussianHead2D, MDNGaussianHead2D
from models.heads.student_t import StudentTHead2D, MDNStudentTHead2D

def _subsequent_mask(sz: int, device=None) -> torch.Tensor:
    return torch.triu(torch.ones(sz, sz, device=device, dtype=torch.bool), diagonal=1)

class SinusoidalPE(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0)/d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        L = x.size(1)
        return x + self.pe[:, offset:offset+L, :]

class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0): super().__init__(); self.drop_prob = drop_prob
    def forward(self, x: torch.Tensor):
        if self.drop_prob == 0.0 or not self.training: return x
        keep = 1.0 - self.drop_prob
        shape = (x.size(0),) + (1,)*(x.ndim-1)
        rand = keep + torch.rand(shape, dtype=x.dtype, device=x.device); rand.floor_()
        return x/keep * rand

class TransformerForecaster(nn.Module):
    def __init__(self,
                 past_len: int,
                 future_len: int,
                 d_model: int = 256,
                 nhead: int = 8,
                 enc_layers: int = 4,
                 dec_layers: int = 4,
                 dim_ff: int = 1024,
                 dropout: float = 0.1,
                 droppath: float = 0.0,
                 use_velocity: bool = True,
                 normalize_xy: bool = True,
                 head: Literal["gauss","student_t","mdn_gauss","mdn_student_t"] = "student_t",
                 mdn_K: int = 5,
                 student_t_dof_init: float = 3.0,
                 learnable_dof: bool = True,
                 add_cam_token: bool = True):
        super().__init__()
        self.P, self.M = past_len, future_len
        self.use_velocity, self.normalize_xy = use_velocity, normalize_xy
        self.add_cam_token, self.head_type, self.mdn_K = add_cam_token, head, mdn_K

        in_dim = 2 + (2 if use_velocity else 0)
        self.in_proj = nn.Linear(in_dim, d_model)
        self.pos_enc = SinusoidalPE(d_model)

        enc_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_ff, dropout=dropout,
                                               batch_first=True, activation="gelu", norm_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=enc_layers)

        dec_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_ff, dropout=dropout,
                                               batch_first=True, activation="gelu", norm_first=True)
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=dec_layers)

        self.future_queries = nn.Parameter(torch.zeros(1, future_len, d_model))
        nn.init.normal_(self.future_queries, std=0.02)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model)) if add_cam_token else None
        if self.cls_token is not None: nn.init.normal_(self.cls_token, std=0.02)

        self.drop_path = DropPath(droppath)

        if head == "gauss":
            self.head = GaussianHead2D(d_model, hidden=2*d_model)
        elif head == "mdn_gauss":
            self.head = MDNGaussianHead2D(d_model, K=mdn_K, hidden=2*d_model)
        elif head == "student_t":
            self.head = StudentTHead2D(d_model, hidden=2*d_model, dof_init=student_t_dof_init, learnable_dof=learnable_dof)
        elif head == "mdn_student_t":
            self.head = MDNStudentTHead2D(d_model, K=mdn_K, hidden=2*d_model, dof_init=student_t_dof_init, learnable_dof=learnable_dof)
        else:
            raise ValueError(f"Unknown head: {head}")

    @staticmethod
    def _to_vel(x: torch.Tensor) -> torch.Tensor:
        v = torch.zeros_like(x); v[:,1:] = x[:,1:] - x[:,:-1]; return v

    @staticmethod
    def _norm_xy(past_xy: torch.Tensor, wh: Optional[torch.Tensor]) -> torch.Tensor:
        if wh is None: return past_xy
        s = (wh.clamp(min=1e-6)) * 0.5
        return (past_xy - s) / s

    def forward(self,
                past_xy: torch.Tensor,            # [B,P,2]
                past_wh: Optional[torch.Tensor] = None,  # [B,P,2]
                cam_feat: Optional[torch.Tensor] = None,
                return_hidden: bool = False) -> Dict:
        B, P, _ = past_xy.shape
        assert P == self.P, f"expect past_len={self.P}, got {P}"

        x = self._norm_xy(past_xy, past_wh) if self.normalize_xy else past_xy
        feats = [x, self._to_vel(x)] if self.use_velocity else [x]
        inp = torch.cat(feats, dim=-1)                   # [B,P,2/4]

        enc = self.in_proj(inp)
        enc = self.pos_enc(enc, offset=0)
        if self.cls_token is not None:
            enc = torch.cat([self.cls_token.expand(B, -1, -1), enc], dim=1)  # [B,1+P,D]
        mem = self.encoder(enc)
        mem = self.drop_path(mem)

        qry = self.future_queries.expand(B, -1, -1)
        qry = self.pos_enc(qry, offset=0)
        tgt_mask = _subsequent_mask(self.M, device=qry.device)
        dec_out = self.decoder(qry, mem, tgt_mask=tgt_mask)                  # [B,M,D]
        h_fut = self.drop_path(dec_out)

        if   self.head_type == "gauss":         params = self.head(h_fut); out = {"type":"gauss", **params}
        elif self.head_type == "mdn_gauss":     params = self.head(h_fut); out = {"type":"mdn_gauss", **params}
        elif self.head_type == "student_t":     params = self.head(h_fut); out = {"type":"student_t", **params}
        else:                                   params = self.head(h_fut); out = {"type":"mdn_student_t", **params}

        if return_hidden: out["hidden_future"]=h_fut; out["hidden_memory"]=mem
        return out
