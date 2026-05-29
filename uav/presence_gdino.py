# presence_gdino.py — Final Stable OWL-ViT Version
import json, argparse
from pathlib import Path
import cv2
from PIL import Image
import torch

from transformers import OwlViTProcessor, OwlViTForObjectDetection

PROMPTS = ["a drone", "an unmanned aerial vehicle", "a quadcopter"]

def load_model(device="cpu"):
    model_id = "google/owlvit-base-patch32"
    print(f"[LOAD] Loading model: {model_id}")
    processor = OwlViTProcessor.from_pretrained(model_id)
    model = OwlViTForObjectDetection.from_pretrained(model_id).to(device)
    print("[LOAD] Model ready.\n")
    return processor, model

def scan_video(video_path, processor, model, every_n=8, min_hits=2, device="cpu"):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {"video": video_path.stem, "drone_present": False, "hits": []}

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    hits = []
    f = 0

    while True:
        ok = cap.grab()
        if not ok:
            break
        if f % every_n != 0:
            f += 1
            continue

        ok, frame = cap.retrieve()
        if not ok:
            break

        image_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        inputs = processor(
            images=image_pil,
            text=PROMPTS,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        # ★★★ FIXED: must be tensor, not list ★★★
        target_sizes = torch.tensor([image_pil.size[::-1]], device=device)

        results = processor.post_process_object_detection(
            outputs=outputs,
            threshold=0.25,
            target_sizes=target_sizes
        )[0]

        labels = results["labels"]
        scores = results["scores"]

        for label, score in zip(labels, scores):
            label_str = PROMPTS[label]
            if any(k in label_str.lower() for k in ["drone", "uav", "quadcopter"]):
                hits.append({
                    "frame": f,
                    "time": float(f/fps),
                    "score": float(score),
                    "phrase": label_str
                })
                break

        f += 1

    cap.release()
    return {
        "video": video_path.stem,
        "drone_present": len(hits) >= min_hits,
        "hits": hits
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video_glob", required=True)
    ap.add_argument("--meta_dir", default="output/mot_fbd")
    ap.add_argument("--out_dir", default="output/uav_presence")
    ap.add_argument("--every_n", type=int, default=8)
    ap.add_argument("--min_hits", type=int, default=2)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    device = "cuda" if (torch.cuda.is_available() and args.device.startswith("cuda")) else "cpu"

    processor, model = load_model(device)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    meta_dir = Path(args.meta_dir)

    from glob import glob
    for vp in glob(args.video_glob):
        vpath = Path(vp)
        print(f"[SCAN] {vpath.name}")
        res = scan_video(vpath, processor, model, every_n=args.every_n, min_hits=args.min_hits, device=device)

        # Save detection result
        (out_dir / f"{vpath.stem}_uav_presence.json").write_text(
            json.dumps(res, indent=2, ensure_ascii=False),
            encoding="utf-8"
        )

        # Update tracking meta if exists
        meta_path = meta_dir / f"{vpath.stem}_track_meta.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            meta["has_drone"] = res["drone_present"]
            meta["uav_hits"] = res["hits"]
            meta_path.write_text(
                json.dumps(meta, indent=2, ensure_ascii=False),
                encoding="utf-8"
            )

        print(f"[DONE] → has_drone={res['drone_present']}  hits={len(res['hits'])}\n")


if __name__ == "__main__":
    main()
