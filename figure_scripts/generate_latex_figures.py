import os
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

LATEX_DIR = Path("latex/figures")
LATEX_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------
# 1) Pipeline Diagram
# --------------------------
def draw_pipeline_diagram():
    img = Image.new("RGB", (1600, 420), "white")
    draw = ImageDraw.Draw(img)

    boxes = [
        ("YOLO Detection", 50),
        ("Multi-Object Tracking", 300),
        ("Stabilization", 580),
        ("MOT → Sequences", 830),
        ("MDN-Transformer\nPrediction", 1100),
        ("Uncertainty\nSampling", 1350),
    ]

    for text, x in boxes:
        draw.rectangle([x, 100, x+200, 300], outline="black", width=4)
        draw.text((x+20, 150), text, fill="black")

    arrows = [(250, 200, 300, 200), (500, 200, 580, 200),
              (780, 200, 830, 200), (1030, 200, 1100, 200),
              (1320, 200, 1350, 200)]

    for x1, y1, x2, y2 in arrows:
        draw.line((x1, y1, x2, y2), fill="black", width=5)
        draw.polygon([(x2, y2),
                      (x2-10, y2-10),
                      (x2-10, y2+10)],
                     fill="black")
    img.save(LATEX_DIR/"pipeline.png")
    print("[OK] pipeline.png generated.")

# --------------------------
# 2) Trajectory Prediction Examples
# --------------------------
def draw_prediction_examples():
    import pandas as pd

    pred_dir = Path("output/preds_offline")
    jsonl = Path("output/jsonl_fbd/test.jsonl")

    if not pred_dir.exists():
        print("[WARN] Predictions not found.")
        return

    data = [json.loads(l) for l in open(jsonl, "r", encoding="utf-8")]
    sample = data[0]
    video = sample["video"]

    # 对应文件名
    pred_file = pred_dir / f"{video}_pred.csv"
    if not pred_file.exists():
        print(f"[WARN] pred file missing: {pred_file}")
        return

    # ⭐ 使用 pandas 读取（安全、自动处理表头）
    df = pd.read_csv(pred_file)

    if not {"x", "y"}.issubset(df.columns):
        print("[WARN] CSV 文件没有 x/y 列，跳过绘图")
        return

    x = df["x"].values
    y = df["y"].values
    steps = np.arange(len(x))

    plt.figure(figsize=(6,4))
    plt.title(f"Trajectory Prediction Example ({video})")
    plt.plot(steps, x, label="Pred x")
    plt.plot(steps, y, label="Pred y")
    plt.xlabel("Future step")
    plt.legend()
    plt.tight_layout()
    plt.savefig(LATEX_DIR/"prediction_examples.png", dpi=300)
    plt.close()
    print("[OK] prediction_examples.png generated.")


# --------------------------
# 3) Uncertainty Calibration
# --------------------------
def draw_calibration_curve():
    # Dummy example curve — replace with your metrics if needed
    nominal = np.array([0.5, 0.9, 0.95])
    empirical = np.array([0.52, 0.88, 0.93])

    plt.figure(figsize=(5,5))
    plt.plot(nominal, empirical, "o-", label="Empirical")
    plt.plot([0,1],[0,1],"--", label="Ideal")
    plt.xlabel("Nominal Quantile")
    plt.ylabel("Empirical Coverage")
    plt.legend()
    plt.grid(True)
    plt.title("Uncertainty Calibration")
    plt.tight_layout()
    plt.savefig(LATEX_DIR/"calibration.png", dpi=300)
    plt.close()
    print("[OK] calibration.png generated.")

# --------------------------
# 4) UAV Example Mosaic
# --------------------------
def draw_uav_examples():
    synth3 = Path("output/synth3/synth")
    if not synth3.exists():
        print("[WARN] synth3 not found.")
        return

    imgs = sorted(list(synth3.glob("*.jpg")))[:9]
    if len(imgs)==0:
        print("[WARN] no UAV frames.")
        return

    rows, cols = 3, 3
    w, h = 320, 240
    canvas = Image.new("RGB", (cols*w, rows*h), "white")

    for idx, f in enumerate(imgs):
        r = idx // cols
        c = idx % cols
        img = Image.open(f).resize((w,h))
        canvas.paste(img, (c*w, r*h))

    canvas.save(LATEX_DIR/"uav_examples.png")
    print("[OK] uav_examples.png generated.")

# --------------------------
# Run all
# --------------------------
if __name__=="__main__":
    print("Generating LaTeX-ready figures...")
    draw_pipeline_diagram()
    draw_prediction_examples()
    draw_calibration_curve()
    draw_uav_examples()
    print("All figures saved to latex/figures/")
