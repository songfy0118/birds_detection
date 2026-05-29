# -*- coding: utf-8 -*-
"""
presence_owlvit_dual.py
用 OWL-ViT 零样本目标检测 同时扫描 output/synth2 与 output/synth，
并把合成视频中命中的帧（红框）导出到 output/synth3/。
依赖：transformers, torch, pillow, opencv-python
"""
import argparse, json
from pathlib import Path
import cv2
import torch
import numpy as np
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

PROMPTS = ["drone", "uav", "quadcopter", "dji phantom", "drone camera"]

def load_model(device="cpu"):
    model_id = "google/owlvit-base-patch32"
    print(f"[LOAD] {model_id} on {device}")
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device).eval()
    return processor, model

def detect_frame(frame_bgr, processor, model, device, box_thr=0.30):
    # BGR -> PIL RGB
    image_pil = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    inputs = processor(text=PROMPTS, images=image_pil, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    target_sizes = torch.tensor([image_pil.size[::-1]]).to(device)   # (H,W)
    results = processor.post_process_object_detection(
        outputs=outputs, threshold=box_thr, target_sizes=target_sizes
    )[0]  # dict: boxes[N,4], scores[N], labels[N]
    return results

def draw_vis(frame_bgr, boxes, scores, labels):
    vis = frame_bgr.copy()
    if isinstance(boxes, torch.Tensor):  boxes = boxes.cpu().numpy()
    if isinstance(scores, torch.Tensor): scores = scores.cpu().numpy()
    if isinstance(labels, torch.Tensor): labels = labels.cpu().numpy()
    for (x1,y1,x2,y2), sc, li in zip(boxes, scores, labels):
        x1,y1,x2,y2 = map(int, [x1,y1,x2,y2])
        cv2.rectangle(vis, (x1,y1), (x2,y2), (0,0,255), 2)
        txt = f"{PROMPTS[int(li)]}:{float(sc):.2f}"
        cv2.putText(vis, txt, (x1, max(0,y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
    return vis

def scan_video(vp: Path, processor, model, device, every_n, box_thr, vis_dir: Path=None):
    cap = cv2.VideoCapture(str(vp))
    if not cap.isOpened():
        print(f"[WARN] 打不开视频：{vp}")
        return False, []
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    hits, fid = [], 0
    while True:
        ok = cap.grab()
        if not ok: break
        if fid % every_n != 0:
            fid += 1; continue
        ok, frame = cap.retrieve()
        if not ok: break

        res = detect_frame(frame, processor, model, device, box_thr=box_thr)
        if res["boxes"].shape[0] > 0:
            hits.append({"frame": fid, "time_s": float(fid/fps), "num": int(res["boxes"].shape[0])})
            if vis_dir is not None:
                vis_dir.mkdir(parents=True, exist_ok=True)
                vis = draw_vis(frame, res["boxes"], res["scores"], res["labels"])
                out_name = f"{vp.stem}_f{fid:05d}.jpg"
                cv2.imwrite(str(vis_dir/out_name), vis)
        fid += 1
    cap.release()
    return len(hits) > 0, hits

def run_folder(folder: Path, processor, model, device, every_n, box_thr, vis_dir: Path=None):
    results = {}
    for vp in sorted(folder.glob("*.mp4")):
        print(f"[SCAN] {vp.name}")
        has, hits = scan_video(vp, processor, model, device, every_n, box_thr, vis_dir=vis_dir)
        results[vp.name] = {"has_drone": bool(has), "hits": hits}
        print(f"[DONE] {vp.name} -> has_drone={has}, hits={len(hits)}")
    return results

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synth2", default="output/synth2")
    ap.add_argument("--synth",  default="output/synth")
    ap.add_argument("--visdir", default="output/synth3")   # 命中帧导出到这里（只对 synth）
    ap.add_argument("--every_n", type=int, default=2)      # 密一点，召回更好
    ap.add_argument("--box_thr", type=float, default=0.25) # 召回优先，后续再抬高
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out_json", default="output/uav_presence_owlvit.json")
    args = ap.parse_args()

    device = "cuda" if (args.device.startswith("cuda") and torch.cuda.is_available()) else "cpu"
    processor, model = load_model(device)

    print("\n===== [1] 原始 synth2（期望 False）=====")
    r2 = run_folder(Path(args.synth2), processor, model, device, args.every_n, args.box_thr, vis_dir=None)

    print("\n===== [2] 合成 synth（期望 True）=====")
    r1 = run_folder(Path(args.synth), processor, model, device, args.every_n, args.box_thr, vis_dir=Path(args.visdir))

    out = {"synth2": r2, "synth": r1}
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n>>> 全部完成！JSON：{args.out_json}")
    print(f"    命中帧（红框）输出：{args.visdir}")

if __name__ == "__main__":
    main()
