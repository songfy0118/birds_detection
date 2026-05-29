from ultralytics import YOLO
import cv2
import json
from pathlib import Path
from glob import glob
import argparse


def save_bbox_image(frame, box, save_path):
    """画框 + 保存一张图（无人机可视化）"""
    x1, y1, x2, y2 = [int(v) for v in box["bbox"]]
    frame = frame.copy()

    cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
    cv2.putText(frame, f"{box['score']:.2f}", (x1, y1-6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    cv2.imwrite(str(save_path), frame)


def detect_video(model, video_path, score_thr=0.20):
    """YOLO 检测视频 → 是否有无人机 + hits + 可视化一帧"""

    cap = cv2.VideoCapture(str(video_path))
    hits = []
    has_uav = False
    vis_saved = False
    frame_id = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_id % 5 == 0:
            result = model(frame, verbose=False)[0]

            for b in result.boxes:
                conf = float(b.conf)
                if conf >= score_thr:
                    xyxy = [float(x) for x in b.xyxy[0].tolist()]
                    hit = {
                        "frame": frame_id,
                        "score": conf,
                        "bbox": xyxy
                    }
                    hits.append(hit)
                    has_uav = True

                    # 保存可视化帧
                    if not vis_saved:
                        save_path = video_path.parent / "synth3" / f"{video_path.stem}_hit.jpg"
                        save_path.parent.mkdir(parents=True, exist_ok=True)
                        save_bbox_image(frame, hit, save_path)
                        vis_saved = True

                    break

        frame_id += 1

    cap.release()
    return has_uav, hits



def process_folder(model, folder, out_json_path):
    """对一个目录跑全部视频"""

    folder = Path(folder)
    videos = sorted(folder.glob("*.mp4"))

    out_json = []

    for vp in videos:
        print(f"[SCAN] {vp.name}")
        has_uav, hits = detect_video(model, vp)

        out_json.append({
            "video": vp.name,
            "has_drone": has_uav,
            "hits": hits
        })

        print(f"[DONE] {vp.name} → has_drone={has_uav}, hits={len(hits)}")

    # 保存 JSON
    Path(out_json_path).write_text(json.dumps(out_json, indent=2), encoding="utf-8")



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synth2", required=True, help="原始无无人机的视频目录")
    ap.add_argument("--synth", required=True, help="合成有无人机的视频目录")
    ap.add_argument("--score_thr", type=float, default=0.20)
    args = ap.parse_args()

    print("[LOAD] Loading YOLOv8n...")
    model = YOLO("yolov8n.pt")

    # 原视频（应该全部 False）
    print("\n===== [1] 检测原始 synth2（应为 False） =====")
    process_folder(model, args.synth2, "output/yolo_synth2_result.json")

    # 合成（应该全部 True）
    print("\n===== [2] 检测合成 synth（应为 True） =====")
    process_folder(model, args.synth, "output/yolo_synth_result.json")

    print("\n>>> 所有检测完成！可视化帧已保存到 synth3 文件夹。")



if __name__ == "__main__":
    main()
