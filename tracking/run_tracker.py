'''''# tracking/run_tracker.py
# ------------------------------------------------------------
# YOLO + ByteTrack（Ultralytics原生）小目标友好版本
# - 支持 classes 过滤（如 bird/drone）
# - 自动生成自定义 tracker.yaml（按fps自适应阈值）
# - 输出：<video>.txt (MOT)、<video>_sizes.csv、<video>_track_meta.json
# - 可选保存可视化视频
# ------------------------------------------------------------
'''
from pathlib import Path
import argparse, json, os, sys, tempfile
import cv2
import numpy as np

try:
    import pandas as pd  # 仅用于健壮性检查，不是硬依赖
except Exception:
    pd = None

from ultralytics import YOLO


# -----------------------------
# 工具
# -----------------------------
def ensure_dir(p: str | Path):
    Path(p).mkdir(parents=True, exist_ok=True)

def write_custom_tracker_yaml(out_dir: Path, fps: float,
                              track_thresh=0.12, high_thresh=0.3,
                              match_thresh=0.8, track_buffer=None) -> Path:
    """
    生成自定义 ByteTrack 配置并返回文件路径。
    - 小目标：降低 track_thresh；提高 match_thresh；
    - track_buffer：若未指定，按 fps 自适应（fps*1.0）
    """
    if track_buffer is None:
        track_buffer = max(15, int(round(fps * 1.0)))

    yaml_text = f"""# Auto-generated for small-object bird/drone tracking
type: bytetrack
track_thresh: {track_thresh}     # detection score for initial track
track_buffer: {track_buffer}     # frames to keep lost tracks
match_thresh: {match_thresh}     # IoU matching threshold
high_thresh: {high_thresh}       # score for matching existing tracks
new_track_thresh: {track_thresh}
frame_rate: {max(1, int(round(fps)))}
# below are default-like; keep stable
gamma: 1.0
min_box_area: 5
fuse_score: True
"""
    p = out_dir / "tracker_bird.yaml"
    with open(p, "w", encoding="utf-8") as f:
        f.write(yaml_text)
    return p


# -----------------------------
# 主流程
# -----------------------------
def track_one_video(model: YOLO,
                    video_path: Path,
                    out_dir: Path,
                    classes: list[int] | None,
                    imgsz: int,
                    conf: float,
                    iou: float,
                    device: str,
                    save_vis: bool,
                    half: bool,
                    track_thresh: float,
                    high_thresh: float,
                    match_thresh: float):
    """对单个视频做检测+跟踪，并导出 MOT / sizes / meta。"""
    ensure_dir(out_dir)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[WARN] Cannot open video: {video_path}")
        return
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    # 自定义 ByteTrack 配置（小目标友好）
    tracker_yaml = write_custom_tracker_yaml(
        out_dir=out_dir, fps=fps,
        track_thresh=track_thresh, high_thresh=high_thresh,
        match_thresh=match_thresh
    )

    stem = video_path.stem
    mot_txt   = out_dir / f"{stem}.txt"
    sizes_csv = out_dir / f"{stem}_sizes.csv"
    meta_json = out_dir / f"{stem}_track_meta.json"
    vis_mp4   = out_dir / f"{stem}_vis.mp4"

    # 结果写句柄
    f_mot = open(mot_txt, "w", encoding="utf-8")
    f_sz  = open(sizes_csv, "w", encoding="utf-8")
    f_sz.write("frame,W,H\n")

    # 统计 track 的主类别（多数表决）
    cls_counts: dict[int, dict[int,int]] = {}   # tid -> {cls_id: count}

    # 可视化
    vw = None
    if save_vis:
        vw = cv2.VideoWriter(str(vis_mp4), cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (w, h))

    # 逐帧遍历（Ultralytics 内部完成检测+跟踪）
    frame_id = 0
    stream = model.track(
        source=str(video_path),
        stream=True,
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        classes=classes,
        device=device,
        half=half,
        tracker="tracking/bytetrack.yaml",  # ← ★ 关键修正
        persist=True,
        verbose=False
    )

    for res in stream:
        # 原始帧尺寸
        H, W = res.orig_img.shape[:2]
        f_sz.write(f"{frame_id},{W},{H}\n")

        if res.boxes is not None and res.boxes.id is not None:
            ids  = res.boxes.id.cpu().numpy().astype(int)
            xyxy = res.boxes.xyxy.cpu().numpy()
            confs= res.boxes.conf.cpu().numpy()
            clss = res.boxes.cls.cpu().numpy().astype(int)

            for i, tid in enumerate(ids):
                x1, y1, x2, y2 = xyxy[i]
                w_box, h_box   = max(0., x2-x1), max(0., y2-y1)
                score          = float(confs[i])
                c              = int(clss[i])

                # 写 MOT（frame,id,x,y,w,h,score,-1,-1,-1）
                f_mot.write(f"{frame_id},{tid},{x1:.2f},{y1:.2f},{w_box:.2f},{h_box:.2f},{score:.4f},-1,-1,-1\n")

                # 类别统计
                d = cls_counts.get(tid, {})
                d[c] = d.get(c, 0) + 1
                cls_counts[tid] = d

                # 可视化
                if save_vis:
                    cv2.rectangle(res.orig_img,
                                  (int(x1), int(y1)), (int(x2), int(y2)),
                                  (0, 255, 0), 2)
                    name = model.model.names.get(c, str(c)) if hasattr(model.model, "names") else str(c)
                    cv2.putText(res.orig_img,
                                f"ID{tid}:{name} {score:.2f}",
                                (int(x1), max(0, int(y1)-7)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,0), 2)

        if save_vis:
            vw.write(res.orig_img)

        frame_id += 1

    f_mot.close()
    f_sz.close()
    if vw: vw.release()

    # 汇总 track 主类别（全部键转换为字符串，避免 numpy.int64 导致 JSON 错误）
    names = model.model.names if hasattr(model.model, "names") else {}

    id2major = {}
    id2counts = {}

    for tid, cc in cls_counts.items():
        # tid 可能是 numpy.int64, 转成 Python int
        tid = int(tid)
        # 多数类
        maj_cls = max(cc.items(), key=lambda kv: kv[1])[0]
        # 将所有键转为 str，确保 JSON 可写
        id2major[str(tid)] = {
            "cls_id": int(maj_cls),
            "cls_name": names.get(int(maj_cls), str(int(maj_cls)))
        }
        id2counts[str(tid)] = {str(k): int(v) for k, v in cc.items()}

    meta = {
        "video": str(video_path),
        "fps": fps,
        "width": w,
        "height": h,
        "frames": nframes,
        "track_majority_class": id2major,
        "track_counts": id2counts
    }

    # 写 JSON（这次不会报错了）
    with open(meta_json, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"[OK] {video_path.name} -> {out_dir}")
    print(f"     - MOT:   {mot_txt.name}")
    print(f"     - SIZES: {sizes_csv.name}")
    print(f"     - META:  {meta_json.name}")
    if save_vis:
        print(f"     - VIS:   {vis_mp4.name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video_glob", required=True,
                    help=r'如："C:\path\to\videos\*.mp4" 或 "dataset/videos/**/*.mp4"')
    ap.add_argument("--yolo", default="yolov8n.pt")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--imgsz", type=int, default=1280,
                    help="小目标建议 >=1280")
    ap.add_argument("--conf", type=float, default=0.15)
    ap.add_argument("--iou", type=float, default=0.6)
    ap.add_argument("--classes", type=int, nargs="+", default=None,
                    help="可选：类别ID过滤，如 --classes 14 0（bird/drone）")
    ap.add_argument("--save_vis", action="store_true")
    ap.add_argument("--half", action="store_true", help="半精度（NVIDIA）")
    ap.add_argument("--out_dir", default="output/mot")
    # 小目标友好 ByteTrack 调参
    ap.add_argument("--track_thresh", type=float, default=0.12)
    ap.add_argument("--high_thresh", type=float, default=0.30)
    ap.add_argument("--match_thresh", type=float, default=0.80)

    args = ap.parse_args()

    ensure_dir(args.out_dir)
    model = YOLO(args.yolo)

    videos = sorted([p for p in Path().glob(args.video_glob) if p.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"}])
    if not videos:
        print(f"[ERR] No videos matched: {args.video_glob}")
        sys.exit(1)

    for vp in videos:
        track_one_video(model, vp, Path(args.out_dir),
                        classes=args.classes, imgsz=args.imgsz, conf=args.conf, iou=args.iou,
                        device=args.device, save_vis=args.save_vis, half=args.half,
                        track_thresh=args.track_thresh, high_thresh=args.high_thresh, match_thresh=args.match_thresh)


if __name__ == "__main__":
    main()

