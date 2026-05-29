# -*- coding: utf-8 -*-
"""
扫描 synth2（原始）与 synth（合成）两个目录内的视频，使用本地 GroundingDINO 检测“无人机”，
并把命中的帧（画红框）导出到 output/synth3/。
"""

import os, sys, json, argparse
from pathlib import Path
import cv2
import numpy as np
import torch

# ---- 1) 加载本地 GroundingDINO 源码 ----
PROJ = Path(__file__).resolve().parents[1]  # day1/
GDINO_ROOT = PROJ / "groundingdino_bak"
if not GDINO_ROOT.exists():
    raise FileNotFoundError(f"[FATAL] 找不到 {GDINO_ROOT}")

sys.path.insert(0, str(GDINO_ROOT))
from groundingdino.util.inference import load_model, predict  # 用源码，不走 pip

# ---- 2) 预处理/工具 ----
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
PROMPT = "drone. UAV. quadcopter."

def build_model(device: str):
    cfg = GDINO_ROOT / "groundingdino" / "config" / "GroundingDINO_SwinB_cfg.py"
    ckpt = GDINO_ROOT / "weights" / "groundingdino_swinb_cogcoor.pth"
    if not cfg.exists():  raise FileNotFoundError(f"缺少配置：{cfg}")
    if not ckpt.exists(): raise FileNotFoundError(f"缺少权重：{ckpt}")
    print(f"[LOAD] GroundingDINO\nCONFIG: {cfg}\nWEIGHTS: {ckpt}")
    model = load_model(str(cfg), str(ckpt)).to(device).eval()
    return model

def frame_to_tensor(frame_bgr: np.ndarray, device: str) -> torch.Tensor:
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(rgb).float().permute(2, 0, 1) / 255.0  # [3,H,W]
    t = (t - MEAN) / STD
    return t.to(device)

def safe_predict(model, image_chw: torch.Tensor, caption: str,
                 box_thr: float, text_thr: float, device: str):
    try:
        return predict(model=model, image=image_chw,
                       caption=caption, box_threshold=box_thr,
                       text_threshold=text_thr, device=device)
    except TypeError:
        return predict(model, image_chw, caption, box_thr, text_thr, device=device)

def draw_boxes(frame_bgr: np.ndarray, boxes, logits, phrases):
    vis = frame_bgr.copy()
    if isinstance(boxes, torch.Tensor):  boxes = boxes.cpu().numpy()
    if isinstance(logits, torch.Tensor): logits = logits.cpu().numpy()
    for (x1, y1, x2, y2), sc, ph in zip(boxes, logits, phrases):
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(vis, f"{ph}:{float(sc):.2f}",
                    (x1, max(0, y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
    return vis

def scan_video(video_path: Path, model, device: str,
               every_n: int, box_thr: float, text_thr: float,
               vis_dir: Path = None):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[WARN] 打不开视频：{video_path}")
        return False, []
    hits, idx = [], 0
    while True:
        ok = cap.grab()
        if not ok: break
        if idx % every_n != 0:
            idx += 1; continue
        ok, frame = cap.retrieve()
        if not ok: break
        img_t = frame_to_tensor(frame, device)
        boxes, logits, phrases = safe_predict(model, img_t, PROMPT, box_thr, text_thr, device)
        if boxes is not None and len(boxes) > 0:
            hits.append(idx)
            if vis_dir is not None:
                vis_dir.mkdir(parents=True, exist_ok=True)
                vis = draw_boxes(frame, boxes, logits, phrases)
                out_name = f"{video_path.stem}_f{idx:05d}.jpg"
                cv2.imwrite(str(vis_dir / out_name), vis)
        idx += 1
    cap.release()
    return len(hits) > 0, hits

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synth2", default="output/synth2")
    ap.add_argument("--synth",  default="output/synth")
    ap.add_argument("--visdir", default="output/synth3")   # 命中帧保存到这里（只对 synth）
    ap.add_argument("--every_n", type=int, default=6)
    ap.add_argument("--box_thr", type=float, default=0.40) # 你要求的 0.4
    ap.add_argument("--text_thr", type=float, default=0.25)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = "cuda" if (args.device.startswith("cuda") and torch.cuda.is_available()) else "cpu"
    print(f"[PATH] Using GroundingDINO from: {GDINO_ROOT}")
    print(f"[DEVICE] {device}")
    model = build_model(device)

    synth2_dir = Path(args.synth2); synth_dir = Path(args.synth); vis_dir = Path(args.visdir)
    results = {"synth2": {}, "synth": {}}

    print("\n===== 检测 synth2（应为 False）=====")
    for vp in sorted(synth2_dir.glob("*.mp4")):
        print(f"[SCAN] {vp.name}")
        has, hit_frames = scan_video(vp, model, device, args.every_n, args.box_thr, args.text_thr, vis_dir=None)
        results["synth2"][vp.name] = {"has_drone": has, "hits": hit_frames}
        print(f"[DONE] {vp.name} → has_drone={has}, hits={len(hit_frames)}")

    print("\n===== 检测 synth（应为 True）=====")
    for vp in sorted(synth_dir.glob("*.mp4")):
        print(f"[SCAN] {vp.name}")
        has, hit_frames = scan_video(vp, model, device, args.every_n, args.box_thr, args.text_thr, vis_dir=vis_dir)
        results["synth"][vp.name] = {"has_drone": has, "hits": hit_frames}
        print(f"[DONE] {vp.name} → has_drone={has}, hits={len(hit_frames)}")

    out_json = PROJ / "output" / "uav_presence_gdino.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n>>> 全部检测完成！结果 JSON：{out_json}")
    print(f"    有无人机的帧（红框）已保存到：{vis_dir}")

if __name__ == "__main__":
    main()
