import cv2
import json
from pathlib import Path
from ultralytics import YOLO
from glob import glob


def detect_uav(model, video_path, score_thr=0.25, save_vis=True):
    cap = cv2.VideoCapture(str(video_path))
    frame_id = 0
    has_drone = False
    hits = []
    vis_saved = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_id % 5 == 0:
            result = model(frame, verbose=False)[0]
            for b in result.boxes:
                conf = float(b.conf)
                if conf < score_thr:
                    continue

                x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
                cls_id = int(b.cls)

                # YOLO会把无人机错误识别为 bird/airplane/kite，因此只要有高置信物体就判定为无人机
                has_drone = True
                hits.append({
                    "frame": frame_id,
                    "conf": conf,
                    "cls": cls_id,
                    "bbox": [x1, y1, x2, y2]
                })

                # 保存可视化帧
                if save_vis and not vis_saved:
                    out_dir = Path(video_path).parent / "synth3"
                    out_dir.mkdir(exist_ok=True)
                    out_path = out_dir / f"{Path(video_path).stem}_hit.jpg"
                    vis = frame.copy()
                    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 3)
                    cv2.imwrite(str(out_path), vis)
                    vis_saved = True
                break

        frame_id += 1

    cap.release()
    return has_drone, hits


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--synth2", required=True)
    ap.add_argument("--synth", required=True)
    args = ap.parse_args()

    print("[LOAD] YOLOv8n...")
    model = YOLO("yolov8n.pt")

    orig_res = []
    synth_res = []

    print("\n===== 检测原始 synth2（应为 False） =====")
    for vp in sorted(glob(args.synth2 + "/*.mp4")):
        has, hits = detect_uav(model, vp)
        print(f"[ORIG] {Path(vp).name} → has_drone={has}, hits={len(hits)}")
        orig_res.append({"video": Path(vp).name, "has_drone": has, "hits": hits})

    print("\n===== 检测合成 synth（应为 True） =====")
    for vp in sorted(glob(args.synth + "/*.mp4")):
        has, hits = detect_uav(model, vp)
        print(f"[SYNTH] {Path(vp).name} → has_drone={has}, hits={len(hits)}")
        synth_res.append({"video": Path(vp).name, "has_drone": has, "hits": hits})

    Path("output/synth2_yolo.json").write_text(json.dumps(orig_res, indent=2), encoding="utf-8")
    Path("output/synth_yolo.json").write_text(json.dumps(synth_res, indent=2), encoding="utf-8")

    print("\n>>> 全部检测完成！结果见 output/，可视化帧见 synth/synth3/")


if __name__ == "__main__":
    main()
