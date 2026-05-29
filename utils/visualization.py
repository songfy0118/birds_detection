# utils/visualization.py
# ------------------------------------------------------------
# Minimal drawing helpers for qualitative figs
# ------------------------------------------------------------
from __future__ import annotations
import numpy as np
import cv2

def draw_traj(img: np.ndarray, pts: np.ndarray, color=(0,255,0), put_text:str|None=None):
    # pts: [L,2] in pixel coordinates
    for i in range(len(pts)-1):
        p1 = tuple(np.round(pts[i]).astype(int))
        p2 = tuple(np.round(pts[i+1]).astype(int))
        cv2.line(img, p1, p2, color, 2)
    if put_text:
        cv2.putText(img, put_text, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
    return img
