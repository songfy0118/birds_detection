# models/heads/intent_head.py
# ------------------------------------------------------------
# Temporal intent/event head:
#   - Aggregate per-step features [B,M,D] → pooled [B,D]
#   - Predict logits for multi-class (CE) or multi-label (BCE)
# Use cases:
#   - Video-level "Has drone?" (binary)
#   - "Bird count ≥ K?" / "species group present?" (multi-label)
#   - Optional: agent-level intent if pooling='none'
# ------------------------------------------------------------
from __future__ import annotations
from typing import Literal, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F

class IntentHead(nn.Module):
    def __init__(self,
                 d_in: int,
                 n_classes: int,
                 pooling: Literal["mean","max","temporal_fc","none"] = "mean",
                 multilabel: bool = False,
                 hidden: int = 0):
        """
        Args:
            d_in: feature dim per step
            n_classes: number of labels
            pooling:
               - "mean"/"max": global temporal pooling
               - "temporal_fc": small temporal conv+pool
               - "none": return per-step logits [B,M,C]
            multilabel: True→BCEWithLogits, False→CE
        """
        super().__init__()
        self.multilabel = multilabel
        self.pooling = pooling
        if pooling == "temporal_fc":
            self.temporal = nn.Sequential(
                nn.Conv1d(d_in, max(64, d_in//2), kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv1d(max(64, d_in//2), d_in, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            )
        if hidden > 0:
            self.fc = nn.Sequential(nn.Linear(d_in, hidden), nn.ReLU(inplace=True),
                                    nn.Linear(hidden, n_classes))
        else:
            self.fc = nn.Linear(d_in, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B,M,D]
        return:
            logits:
              - pooling!="none": [B,C]
              - pooling=="none": [B,M,C]
        """
        if self.pooling == "none":
            B,M,D = x.shape
            x = x.reshape(B*M, D)
            logits = self.fc(x).reshape(B,M,-1)
            return logits

        if self.pooling == "temporal_fc":
            # [B,M,D] -> [B,D,M] -> conv1d -> [B,D,M] -> mean pool
            x = x.transpose(1,2)
            x = self.temporal(x)
            x = x.mean(dim=-1)  # [B,D]
        elif self.pooling == "max":
            x = x.max(dim=1).values
        else:
            x = x.mean(dim=1)
        logits = self.fc(x)
        return logits

    def loss(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        If multilabel=True: target is float in {0,1}, shape [B,C]
        Else: target is long class id, shape [B]
        """
        if self.multilabel:
            return F.binary_cross_entropy_with_logits(logits, target.float())
        return F.cross_entropy(logits, target.long())
