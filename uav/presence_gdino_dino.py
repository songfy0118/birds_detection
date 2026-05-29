import os
import cv2
import json
import argparse
import torch
from pathlib import Path

# ====== 强制使用你项目里的 GroundingDINO ======
import sys
sys.path.append("GroundingDINO/groundingdino")   # ★★ 正确路径
# ===============================================

from groundingdino.util.inference import Model
from groundingdino.util.inference import predict

TEXT_PROMPT = "drone . UAV . quadcopter . unmanned aerial vehicle"

def load_model(device):
    config_path = "GroundingDINO/groundingdino/config/GroundingDINO_SwinB.cfg.py"
    weight_path = "GroundingDINO/weights/groundingdino_swinb_cogcoor.pth"

    print("[CONFIG]", config_path)
    print("[WEIGHT]", weight_path)

    model = Model(
        model_config_path=config_path,
        model_checkpoint_path=weight_path,
        device=device
    )
    return model

def detect_dino(model, frame):
    boxes, scores, phrases = predict(
        model=model,
        image=frame,
        caption=TEXT_PROMPT,
        box_threshold=0.25,
        text_threshold=0.25
    )

    hits = []
    for box, score, phrase in zip(boxes, scores, phrases):
        p = phrase.lower()
        if any(k in p for k in ["drone", "uav", "quadcopter"]):
            hits.append({
                "bbox": [float(b) for b in box],
                "score": float(score),
                "phrase": phrase
            })
    return hits

def scan_video(vp, model):
    cap = cv2.VideoCapture(str(vp))
    hits = []
    f = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if f % 5 == 0:
            dets = detect_dino(model, frame)
            if dets:
                hits.append({"frame": f, "detections": dets})

        f += 1

    cap.release()
    return hits

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video_glob", required=True)
    ap.add_argument("--out_dir", default="output/uav_presence_dino")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = "cuda" if (torch.cuda.is_available() and args.device=="cuda") else "cpu"
    print("[DEVICE]", device)

    model = load_model(device)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    from glob import glob
    for vp in glob(args.video_glob):
        vp = Path(vp)
        print(f"[SCAN] {vp.name} ...")

        hits = scan_video(vp, model)

        result = {
            "video": vp.name,
            "has_drone": len(hits) > 0,
            "hits": hits
        }

        outfile = out_dir / f"{vp.stem}_presence.json"
        outfile.write_text(
            json.dumps(result, indent=2, ensure_ascii=False),
            encoding="utf-8"
        )

        print(f"[DONE] {vp.name} → has_drone={result['has_drone']}, hits={len(hits)}")

if __name__ == "__main__":
    main()
