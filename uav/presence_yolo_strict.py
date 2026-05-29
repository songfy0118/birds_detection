import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import json
from glob import glob

# 你的干净无人机PNG模板路径（任选一个PNG即可）
TEMPLATE = "archive/dataset/drone/final_uav_clean/1_clean.png"


def template_match_score(crop, template):
    crop = cv2.resize(crop, (template.shape[1], template.shape[0]))
    crop_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    res = cv2.matchTemplate(crop_gray, template, cv2.TM_CCOEFF_NORMED)
    return res.max()


def detect_true_drone(model, video_path, template, score_thr=0.2, sim_thr=0.5):
    cap = cv2.VideoCapture(str(video_path))
    frame_id = 0
    hits = []
    has_drone = False
    saved = False

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_id % 5 == 0:
            result = model(frame, verbose=False)[0]

            for b in result.boxes:
                conf = float(b.conf)
                if conf < score_thr:
                    continue

                x1, y1, x2, y2 = [int(v) for v in b.xyxy[0].tolist()]
                crop = frame[y1:y2, x1:x2]

                if crop.size == 0:
                    continue

                sim = template_match_score(crop, template)

                if sim >= sim_thr:
                    has_drone = True
                    hit = {
                        "frame": frame_id,
                        "score": conf,
                        "similarity": float(sim),
                        "bbox": [float(x1), float(y1), float(x2), float(y2)]
                    }
                    hits.append(hit)

                    if not saved:
                        save_path = Path(video_path).parent / "synth3" / f"{Path(video_path).stem}_hit.jpg"
                        save_path.parent.mkdir(exist_ok=True)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                        cv2.imwrite(str(save_path), frame)
                        saved = True

        frame_id += 1

    cap.release()
    return has_drone, hits


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--synth2", required=True)
    ap.add_argument("--synth", required=True)
    args = ap.parse_args()

    template = cv2.imread(TEMPLATE, cv2.IMREAD_GRAYSCALE)
    model = YOLO("yolov8n.pt")

    out_orig = []
    out_synth = []

    print("\n===== 原视频（应为 False） =====")
    for vp in sorted(glob(args.synth2 + "/*.mp4")):
        has, hits = detect_true_drone(model, vp, template)
        print(f"{Path(vp).name}: {has} ({len(hits)} hits)")
        out_orig.append({"video": Path(vp).name, "has_drone": has, "hits": hits})

    print("\n===== 合成视频（应为 True） =====")
    for vp in sorted(glob(args.synth + "/*.mp4")):
        has, hits = detect_true_drone(model, vp, template)
        print(f"{Path(vp).name}: {has} ({len(hits)} hits)")
        out_synth.append({"video": Path(vp).name, "has_drone": has, "hits": hits})

    Path("output/yolo_strict_orig.json").write_text(json.dumps(out_orig, indent=2))
    Path("output/yolo_strict_synth.json").write_text(json.dumps(out_synth, indent=2))


if __name__ == "__main__":
    main()
