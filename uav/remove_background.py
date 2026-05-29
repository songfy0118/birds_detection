from rembg import remove
from PIL import Image
from pathlib import Path

# 输入和输出文件夹
src = Path("archive/dataset/drone/final_uav")  # 你的原图
dst = Path("archive/dataset/drone/final_uav_clean")  # 输出透明 PNG
dst.mkdir(exist_ok=True)

# 支持的格式
valid_ext = [".jpg", ".jpeg", ".png"]

print("=== UAV Auto Background Removal (rembg) ===\n")

for img_path in src.iterdir():
    if img_path.suffix.lower() not in valid_ext:
        continue

    print(f"[PROCESS] {img_path.name}")

    try:
        img = Image.open(img_path)
        out = remove(img)  # 自动扣图（AI segmentation）

        # 导出为透明 PNG
        out_name = img_path.stem + "_clean.png"
        out_path = dst / out_name

        out.save(out_path)
        print(f"[SAVED] {out_name}\n")

    except Exception as e:
        print(f"[ERROR] {img_path.name}: {e}")

print("\n=== ALL DONE! Clean UAV saved to final_uav_clean ===")
