# -*- coding: utf-8 -*-
"""
presence_owlvit_offline.py (High Recall Mode for Paper)

Modifications for High Recall:
1. Lower score threshold (0.05)
2. Scan every frame (every_n=1)
3. More synonyms in prompts
"""
import argparse, json, os
from pathlib import Path
import math
import cv2
import numpy as np
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

# --- 增强的 Prompt 列表 ---
DEFAULT_PROMPTS = [
    "drone", "uav", "quadcopter", "quadrotor",
    "drone aircraft", "small aircraft", "flying toy",
    "remote control helicopter", "dji", "mavic"
]


# --------------------- 基础工具 ---------------------
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def bgr2pil(img_bgr: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))


def draw_box_with_label(img, box, text, color=(0, 0, 255)):
    x1, y1, x2, y2 = [int(v) for v in box]
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    # 文本底色条
    ((tw, th), _) = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    cv2.rectangle(img, (x1, max(0, y1 - 22)), (x1 + tw + 6, y1), color, -1)
    cv2.putText(img, text, (x1 + 3, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)


def map_label(lbl, prompts):
    if isinstance(lbl, str): return lbl
    if isinstance(lbl, (int, np.integer)):
        idx = int(lbl)
        return prompts[idx] if 0 <= idx < len(prompts) else str(idx)
    if torch.is_tensor(lbl):
        idx = int(lbl.item())
        return prompts[idx] if 0 <= idx < len(prompts) else str(idx)
    return str(lbl)


def build_model(local_dir, device="cpu"):
    print(f"[LOAD] OWL-ViT (local) -> {local_dir}")
    try:
        processor = AutoProcessor.from_pretrained(local_dir, local_files_only=True)
        model = AutoModelForZeroShotObjectDetection.from_pretrained(local_dir, local_files_only=True)
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Try checking your local_model_dir path.")
        exit(1)
    model.to(device)
    model.eval()
    return processor, model


def make_tiles(img, tiles: int):
    h, w = img.shape[:2]
    th, tw = h // tiles, w // tiles
    out = []
    for ty in range(tiles):
        for tx in range(tiles):
            x0 = tx * tw
            y0 = ty * th
            x1 = w if tx == tiles - 1 else (x0 + tw)
            y1 = h if ty == tiles - 1 else (y0 + th)
            out.append((img[y0:y1, x0:x1], x0, y0, x1 - x0, y1 - y0))
    return out


def resize_short_side(img, short_side=None):
    if not short_side or short_side <= 0:
        return img, 1.0
    h, w = img.shape[:2]
    s = min(h, w)
    if s == short_side:
        return img, 1.0
    scale = short_side / float(s)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    return resized, scale


# --------------------- 单帧检测 ---------------------
def detect_frame(img_bgr, processor, model, prompts, score_thr=0.05, tiles=2, device="cpu", short_side=768):
    img_big, scale = resize_short_side(img_bgr, short_side)
    hits = []

    for tile, x0, y0, w, h in make_tiles(img_big, tiles):
        pil = bgr2pil(tile)
        inputs = processor(text=[prompts], images=pil, return_tensors="pt")
        inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        target_sizes = torch.tensor([pil.size[::-1]]).to(device)
        results = processor.post_process_object_detection(
            outputs=outputs, target_sizes=target_sizes, threshold=0.0
        )[0]

        boxes = results["boxes"].cpu().numpy()
        scores = results["scores"].cpu().numpy()
        labels = results["labels"]

        for b, sc, lb in zip(boxes, scores, labels):
            if sc < score_thr:
                continue

            # 只要是列表里的词都算命中
            # (因为我们列表里只放了无人机相关的)
            phrase = map_label(lb, prompts)

            bx = np.array([b[0] + x0, b[1] + y0, b[2] + x0, b[3] + y0], dtype=np.float32)
            bx /= scale
            hits.append((bx, float(sc), phrase))

    return hits


# --------------------- 视频扫描 ---------------------
def scan_video(video_path: Path, vis_dir: Path, processor, model, prompts, every_n=1, score_thr=0.05,
               tiles=2, device="cpu", short_side=768, viz=True, min_hits=1):
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    ensure_dir(vis_dir)
    frame_id, hit_frames, all_boxes, best = 0, 0, 0, 0.0
    hits_records = []

    while True:
        ok, frame = cap.read()
        if not ok: break

        # 跳帧逻辑
        if frame_id % every_n != 0:
            frame_id += 1
            continue

        hits = detect_frame(frame, processor, model, prompts, score_thr, tiles, device, short_side)

        if hits:
            hit_frames += 1
            all_boxes += len(hits)
            best = max(best, max(sc for _, sc, _ in hits))

            # 保存可视化图 (Paper Figure)
            # 只保存置信度比较高的帧，或者每隔几帧保存一次，防止硬盘爆炸
            if viz and (hit_frames % 5 == 1 or max(sc for _, sc, _ in hits) > 0.2):
                vis = frame.copy()
                for (bx, sc, phr) in hits:
                    # 画红框
                    draw_box_with_label(vis, bx, f"UAV {sc:.2f}", color=(0, 0, 255))

                cv2.putText(vis, f"{video_path.stem} f={frame_id}", (10, 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                # 保存文件名带上分数，方便挑图
                save_name = f"{video_path.stem}_f{frame_id:04d}_score{int(best * 100)}.jpg"
                cv2.imwrite(str(vis_dir / save_name), vis)

        hits_records.append({
            "frame": int(frame_id),
            "time": frame_id / fps,
            "n_boxes": len(hits),
            "max_score": float(max([sc for _, sc, _ in hits], default=0.0))
        })
        frame_id += 1

    cap.release()
    # 判定该视频是否含有无人机
    has_drone = (hit_frames >= min_hits)
    return has_drone, hit_frames, all_boxes, best, hits_records


# --------------------- 目录遍历 ---------------------
def run_folder(folder, output_json, processor, model, prompts, args):
    folder = Path(folder)
    vids = sorted([p for p in folder.glob("*.mp4")])

    # 可视化图片保存在 json同级目录下的 images 文件夹
    vis_dir = Path(output_json).parent / "uav_viz_images"
    ensure_dir(vis_dir)

    out_list = []
    print(f"Scanning {len(vids)} videos in {folder}...")

    count_true = 0
    for i, v in enumerate(vids):
        has, hits, boxes, best, records = scan_video(
            v, vis_dir, processor, model, prompts,
            every_n=args.every_n,
            score_thr=args.score_thr,
            tiles=args.tiles,
            device=args.device,
            short_side=args.short_side,
            viz=True,
            min_hits=args.min_hits
        )

        status = "DETECTED" if has else "MISSED"
        if has: count_true += 1

        print(f"[{i + 1}/{len(vids)}] {v.stem} -> {status} (hits={hits}, max={best:.3f})")

        out_list.append({
            "video": v.stem,
            "has_drone": has,
            "hits": int(hits),
            "max_score": float(best)
        })

    # 保存 JSON
    Path(output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(out_list, f, indent=2)

    print(f"\nTotal Detected: {count_true} / {len(vids)}")
    print(f"Results saved to {output_json}")
    print(f"Visualizations saved to {vis_dir}")


def main():
    ap = argparse.ArgumentParser()
    # 改为单文件夹模式，方便你灵活跑
    ap.add_argument("--input_dir", required=True, help="视频文件夹路径")
    ap.add_argument("--output_json", required=True, help="结果JSON保存路径")

    ap.add_argument("--local_model_dir", default="models/owlvit-base-patch32")
    ap.add_argument("--every_n", type=int, default=1, help="每隔几帧检测(设为1最准)")
    ap.add_argument("--score_thr", type=float, default=0.05, help="置信度阈值(设低点)")
    ap.add_argument("--tiles", type=int, default=2, help="切图数量(2x2)")
    ap.add_argument("--short_side", type=int, default=960, help="放大短边以检测小目标")
    ap.add_argument("--min_hits", type=int, default=1, help="只要有1帧检测到就算有")
    ap.add_argument("--device", default="cuda")  # 默认尝试cuda

    args = ap.parse_args()

    device = "cuda" if (args.device.startswith("cuda") and torch.cuda.is_available()) else "cpu"
    print(f"Using device: {device}")

    processor, model = build_model(args.local_model_dir, device)

    run_folder(args.input_dir, args.output_json, processor, model, DEFAULT_PROMPTS, args)


if __name__ == "__main__":
    main()