import cv2
import json
from pathlib import Path
from ultralytics import YOLO
from glob import glob


def detect_uav_in_video(model, video_path, score_thr=0.25, save_vis=True):
    cap = cv2.VideoCapture(str(video_path))
    frame_id = 0
    has_drone = False
    hits = []
    save_done = False

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_id % 5 == 0:
            result = model(frame, verbose=False)[0]

            for b in result.boxes:
                conf = float(b.conf)
                cls_id = int(b.cls)

                # YOLO COCO 类别中，UAV 常被误当 airplane, bird, kite, sports ball, etc.
                # 所以我们只依赖 conf 阈值来判定存在物体即可
                if conf >= score_thr:
                    has_drone = True

                    x1, y1, x2, y2 = [int(v) for v in b.xyxy[0].tolist()]
                    hits.append({
                        "frame": frame_id,
                        "conf": conf,
                        "bbox": [x1, y1, x2, y2],
                        "cls": cls_id
                    })

                    # 保存第一帧可视化
                    if save_vis and not save_done:
                        save_dir = Path(video_path).parent / "synth3"
                        save_dir.mkdir(exist_ok=True)
                        vis = frame.copy()
                        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        out_path = save_dir / f"{Path(video_path).stem}_hit.jpg"
                        cv2.imwrite(str(out_path), vis)
                        save_done = True

                    break

        frame_id += 1

    cap.release()
    return has_drone, hits


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--synth2", required=True)  # 原视频
    ap.add_argument("--synth", required=True)   # 合成视频
    args = ap.parse_args()

    print("[LOAD] YOLOv8n...")
    model = YOLO("yolov8n.pt")

    synth2_results = []
    synth_results = []

    print("\n===== [1] 检测原始 synth2（应为 False） =====")
    for vp in sorted(glob(args.synth2 + "/*.mp4")):
        has, hits = detect_uav_in_video(model, vp)
        print(f"[ORIG] {Path(vp).name} → has_drone={has}, hits={len(hits)}")
        synth2_results.append({
            "video": Path(vp).name,
            "has_drone": has,
            "hits": hits
        })

    print("\n===== [2] 检测合成 synth（应为 True） =====")
    for vp in sorted(glob(args.synth + "/*.mp4")):
        has, hits = detect_uav_in_video(model, vp)
        print(f"[SYNTH] {Path(vp).name} → has_drone={has}, hits={len(hits)}")
        synth_results.append({
            "video": Path(vp).name,
            "has_drone": has,
            "hits": hits
        })

    # 保存 JSON
    Path("output/synth2_yolo.json").write_text(json.dumps(synth2_results, indent=2))
    Path("output/synth_yolo.json").write_text(json.dumps(synth_results, indent=2))

    print("\n>>> 所有检测完成！")
    print(">>> 结果 JSON 已保存到 output/")
    print(">>> 可视化帧在 output/synth/synth3/ 中。")


if __name__ == "__main__":
    main()
