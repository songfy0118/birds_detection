import os

videos = [
    "bird_120.mp4",
    "bird_233.mp4",
    "bird_366.mp4",
    "bird_367.mp4",
    "bird_445.mp4",
    "bird_94.mp4",
    "bird_142.mp4",
    "bird_203.mp4",
    "bird_128.mp4",
    "bird_312.mp4"
]

uavs = [
    "1_clean.png",
    "3_clean.png",
    "1_clean.png",
    "4_clean.png",
    "5_clean.png",
    "1_clean.png",
    "3_clean.png",
    "5_clean.png",
    "2_clean.png",
    "1_clean.png"
]

appear_s = [2,3,2,4,2,3,4,2,3,2]

scales = [0.4] * 10   # 全部 0.12

for i in range(10):
    video = videos[i]
    uav = uavs[i]
    out_name = video.replace(".mp4", "_uav.mp4")

    cmd = (
        f"python uav/synthesize_uav.py "
        f"--video_in training_data_base2_FBD-SV-2024/videos/train/{video} "
        f"--drone_png archive/dataset/drone/final_uav_clean/{uav} "
        f"--out_video output/synth/{out_name} "
        f"--appear_s {appear_s[i]} "
        f"--scale {scales[i]}"
    )

    print("[RUN]", cmd)
    os.system(cmd)
