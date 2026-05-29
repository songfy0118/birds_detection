import cv2
import json
import numpy as np
from pathlib import Path
from PIL import Image
import torch
from transformers import OwlViTProcessor, OwlViTForObjectDetection

# ------ CONFIG ------
PROMPTS = ["a drone", "an unmanned aerial vehicle", "a quadcopter"]
AVOID = ["person", "man", "woman"]
MIN_AREA = 0.05     # drone bbox must occupy at least 5% of image
MAX_AREA = 0.60     # not too large
BLUR_THRESHOLD = 30 # Laplacian variance threshold
# ---------------------

def detect_objects(image_pil, processor, model, device="cpu"):
    inputs = processor(text=PROMPTS + AVOID, images=image_pil, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    target_sizes = torch.tensor([image_pil.size[::-1]], device=device)
    results = processor.post_process_object_detection(outputs, threshold=0.20, target_sizes=target_sizes)[0]
    return results

def variance_of_laplacian(image):
    return cv2.Laplacian(image, cv2.CV_64F).var()

def evaluate_image(path, processor, model, device="cpu"):
    img_cv = cv2.imread(str(path))
    if img_cv is None:
        return {"ok": False, "reason": "cannot_load"}

    img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(img_rgb)
    H, W = img_cv.shape[:2]
    img_area = H * W

    result = detect_objects(image_pil, processor, model, device)

    # flags
    has_drone = False
    has_person = False
    drone_score = 0
    drone_area_ratio = 0

    for label, box, score in zip(result["labels"], result["boxes"], result["scores"]):
        label_str = PROMPTS + AVOID
        label_str = label_str[label]

        x1,y1,x2,y2 = box
        area = (x2-x1)*(y2-y1)
        ratio = area / img_area

        if label_str in PROMPTS:
            has_drone = True
            drone_score = float(score)
            drone_area_ratio = ratio

        if label_str in AVOID:
            has_person = True

    # blur score
    blur = variance_of_laplacian(img_cv)

    # ----------- DECISION -------------
    if not has_drone:
        return {"ok": False, "reason": "no_drone_detected"}

    if has_person:
        return {"ok": False, "reason": "contains_human"}

    if drone_area_ratio < MIN_AREA:
        return {"ok": False, "reason": f"drone_too_small({drone_area_ratio:.3f})"}

    if drone_area_ratio > MAX_AREA:
        return {"ok": False, "reason": f"drone_too_large({drone_area_ratio:.3f})"}

    if blur < BLUR_THRESHOLD:
        return {"ok": False, "reason": f"too_blurry({blur:.1f})"}

    # score
    score = 50
    score += drone_score * 20
    score += (1 - drone_area_ratio) * 20
    score += min(blur/200, 1) * 10

    return {
        "ok": True,
        "reason": "clean",
        "score": float(score),
        "drone_score": float(drone_score),
        "area_ratio": float(drone_area_ratio),
        "blur": float(blur)
    }

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_id = "google/owlvit-base-patch32"
    processor = OwlViTProcessor.from_pretrained(model_id)
    model = OwlViTForObjectDetection.from_pretrained(model_id).to(device)

    src = Path("archive/dataset/drone")
    good_dir = src / "clean_uav"
    bad_dir = src / "bad_uav"
    good_dir.mkdir(exist_ok=True)
    bad_dir.mkdir(exist_ok=True)

    report = {}

    for img in src.glob("*"):
        if img.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
            continue

        print(f"[CHECK] {img.name}")
        info = evaluate_image(img, processor, model, device)
        report[img.name] = info

        if info["ok"]:
            target = good_dir / img.name
        else:
            target = bad_dir / img.name

        img.replace(target)

    # save report
    (src / "drone_quality_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print("[DONE] Drone folder cleaned. Report saved.")

if __name__ == "__main__":
    main()
