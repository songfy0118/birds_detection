import random
import shutil
import subprocess
from pathlib import Path
import glob
import cv2


# ----------- 配置 -------------
ROOT = Path(__file__).resolve().parents[1]

VIDEO_PATTERN = str(ROOT / "training_data_base2_FBD-SV-2024" / "videos" / "train" / "*.mp4")
CLEAN_DIR = ROOT / "output" / "synth2_100"
SYNTH_DIR = ROOT / "output" / "synth_100"
DRONE_DIR = ROOT / "archive" / "dataset" / "drone" / "final_uav_clean"

NUM_VIDEOS = 100
RANDOM_SEED = 42
# --------------------------------


def list_videos_abs(pattern):
    return [Path(p) for p in glob.glob(pattern)]


def list_drones():
    return sorted(DRONE_DIR.glob("*_clean.png"))


def get_video_duration(video_path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 5.0
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    return frames / fps if frames and fps else 5.0


def choose_appear_time(duration):
    if duration <= 3.0:
        return 1.0
    start = duration * 0.3
    end = max(start + 1.0, duration * 0.8)
    return round(random.uniform(start, end), 1)


def choose_scale():
    return 0.60   # ⭐ 固定无人机大小


def main():
    random.seed(RANDOM_SEED)

    CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    SYNTH_DIR.mkdir(parents=True, exist_ok=True)

    all_videos = list_videos_abs(VIDEO_PATTERN)
    drones = list_drones()

    if len(all_videos) == 0:
        print(f"[ERR] No videos matched {VIDEO_PATTERN}")
        return
    if len(drones) == 0:
        print(f"[ERR] No drone PNG found in {DRONE_DIR}")
        return

    print(f"[INFO] Found {len(all_videos)} source videos")
    print(f"[INFO] Found {len(drones)} drone PNGs")
    print(f"[INFO] Generating {NUM_VIDEOS} synthetic pairs...\n")

    random.shuffle(all_videos)
    selected = all_videos[:NUM_VIDEOS]

    for idx, vid in enumerate(selected, 1):
        stem = vid.stem
        clean_out = CLEAN_DIR / f"{stem}.mp4"
        synth_out = SYNTH_DIR / f"{stem}_uav.mp4"
        drone_png = random.choice(drones)

        print(f"[{idx}/{NUM_VIDEOS}] {vid.name}")
        print(f"   clean → {clean_out.name}")
        print(f"   synth → {synth_out.name}")
        print(f"   drone → {drone_png.name}")

        if not clean_out.exists():
            shutil.copy2(vid, clean_out)

        if synth_out.exists():
            print("   [SKIP] synth exists")
            continue

        duration = get_video_duration(vid)
        appear_s = choose_appear_time(duration)
        scale = choose_scale()  # 正确 return 0.60

        cmd = [
            "python",
            str(ROOT / "uav" / "synthesize_uav.py"),
            "--video_in", str(vid),
            "--drone_png", str(drone_png),
            "--out_video", str(synth_out),
            "--appear_s", str(appear_s),
            "--scale", str(scale),
        ]

        print(f"   [RUN] appear_s={appear_s}, scale={scale}")
        subprocess.run(cmd, check=True)

    print("\n[OK] Finished generating 100 synthetic pairs.")
    print(f"[INFO] Clean videos → {CLEAN_DIR}")
    print(f"[INFO] Synth videos → {SYNTH_DIR}")


if __name__ == "__main__":
    main()
