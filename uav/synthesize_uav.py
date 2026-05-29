import cv2
import json
import argparse
import numpy as np
from pathlib import Path
from PIL import Image

def overlay_png(background, overlay, x, y):
    """Overlay PNG with alpha channel onto background frame."""
    bh, bw = background.shape[:2]
    oh, ow = overlay.shape[:2]

    if x + ow > bw or y + oh > bh:
        return background

    # If JPG (no alpha), use simple paste
    if overlay.shape[2] == 3:
        bg = background.copy()
        bg[y:y+oh, x:x+ow] = overlay
        return bg

    alpha = overlay[:, :, 3] / 255.0
    bg = background.copy()

    for c in range(3):
        bg[y:y+oh, x:x+ow, c] = (
            alpha * overlay[:, :, c] +
            (1 - alpha) * bg[y:y+oh, x:x+ow, c]
        )
    return bg

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_in", required=True)
    parser.add_argument("--drone_png", required=True)
    parser.add_argument("--out_video", required=True)
    parser.add_argument("--appear_s", type=float, default=2.0)
    parser.add_argument("--scale", type=float, default=0.15)
    args = parser.parse_args()

    # load video
    cap = cv2.VideoCapture(args.video_in)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {args.video_in}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # load drone
    drone = cv2.imread(args.drone_png, cv2.IMREAD_UNCHANGED)
    if drone is None:
        raise RuntimeError(f"Cannot load drone image: {args.drone_png}")

    # resize drone
    max_dim = int(min(W, H) * args.scale)
    scale = max_dim / max(drone.shape[0], drone.shape[1])
    drone = cv2.resize(drone, (int(drone.shape[1] * scale), int(drone.shape[0] * scale)))

    oh, ow = drone.shape[:2]
    x = np.random.randint(10, W - ow - 10)
    y = np.random.randint(10, H - oh - 10)

    appear_frames = int(args.appear_s * fps)
    start = np.random.randint(int(total * 0.2), int(total * 0.6))
    end = min(start + appear_frames, total - 1)

    print(f"[INFO] Video: {args.video_in}")
    print(f"[INFO] Drone inserts from frame {start} to {end} at position ({x}, {y})")

    # output writer
    out_dir = Path(args.out_video).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(args.out_video, fourcc, fps, (W, H))

    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if start <= idx <= end:
            frame = overlay_png(frame, drone, x, y)

        out.write(frame)
        idx += 1

    cap.release()
    out.release()

    print(f"[DONE] Saved synthesized video → {args.out_video}")

if __name__ == "__main__":
    main()
