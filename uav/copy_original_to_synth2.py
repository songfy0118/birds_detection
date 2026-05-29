from pathlib import Path
import shutil

# 你的 10 个合成视频所在目录
synth_dir = Path("output/synth")

# 原始大视频库目录（你的鸟视频来源）
orig_dir = Path("training_data_base2_FBD-SV-2024/videos/train")

# 输出目标目录（原始视频复制到这里）
dst_dir = Path("output/synth2")
dst_dir.mkdir(parents=True, exist_ok=True)

# 遍历 synth 中的 *_uav.mp4
for f in synth_dir.glob("*_uav.mp4"):
    name = f.stem                     # bird_94_uav
    base = name.replace("_uav", "")   # bird_94

    orig_video = orig_dir / f"{base}.mp4"
    dst_video = dst_dir / f"{base}.mp4"

    if orig_video.exists():
        print(f"[COPY] {orig_video.name} → synth2/")
        shutil.copy(orig_video, dst_video)
    else:
        print(f"[MISS] 原视频不存在: {orig_video}")
