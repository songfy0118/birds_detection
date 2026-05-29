# tracking/stabilize.py
# ------------------------------------------------------------
# Final stable version for bird trajectory stabilization
# - No unicodeescape issues
# - No index errors on smoothing
# - Handles short videos
# - ORB + RANSAC + Optional ECC
# - Fully compatible with your run_tracker output
# ------------------------------------------------------------

import os
import argparse
import numpy as np
import cv2
from pathlib import Path
import pandas as pd

COLS = ["frame","id","x","y","w","h","conf","x3","y3","z3"]


# ============ Basic IO utils ==================
def read_mot(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, header=None, names=COLS)
    df["frame"] = df["frame"].astype(int)
    df["id"] = df["id"].astype(int)
    return df

def read_sizes(path: Path) -> dict:
    m = {}
    with open(path, "r", encoding="utf-8") as f:
        next(f)
        for ln in f:
            fr,W,H = ln.strip().split(",")
            m[int(fr)] = (int(W),int(H))
    return m

def ensure_dir(p):
    Path(p).mkdir(exist_ok=True, parents=True)


# ============ Geometry utils ================
def mat2params(A: np.ndarray):
    dx = A[0,2]
    dy = A[1,2]
    da = np.arctan2(A[1,0], A[0,0])
    return dx, dy, da

def params2mat(dx, dy, da):
    return np.array([
        [np.cos(da), -np.sin(da), dx],
        [np.sin(da),  np.cos(da), dy]
    ], dtype=np.float32)


# ============ Stabilization core =============
def compute_stabilization(video_path: Path, use_ecc=True, smooth_radius=10):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(video_path)

    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ok, prev = cap.read()
    if not ok:
        cap.release()
        raise RuntimeError("Cannot read first frame")

    prev_g = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(nfeatures=2000)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    C_list = [np.hstack([np.eye(2), np.zeros((2,1))]).astype(np.float32)]

    # --- pairwise transforms ---
    for i in range(1, n):
        ok, curr = cap.read()
        if not ok:
            break

        curr_g = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)

        kp1, des1 = orb.detectAndCompute(prev_g, None)
        kp2, des2 = orb.detectAndCompute(curr_g, None)

        if (
            kp1 is None or kp2 is None or
            des1 is None or des2 is None or
            len(kp1) < 10 or len(kp2) < 10
        ):
            T = np.hstack([np.eye(2), np.zeros((2,1))]).astype(np.float32)
        else:
            matches = bf.match(des1, des2)
            matches = sorted(matches, key=lambda m: m.distance)[:200]
            if len(matches) < 8:
                T = np.hstack([np.eye(2), np.zeros((2,1))]).astype(np.float32)
            else:
                pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
                pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
                T, _ = cv2.estimateAffinePartial2D(
                    pts1, pts2,
                    method=cv2.RANSAC,
                    ransacReprojThreshold=3.0
                )
                if T is None:
                    T = np.hstack([np.eye(2), np.zeros((2,1))]).astype(np.float32)

        # Optional ECC refinement
        if use_ecc:
            try:
                warp = T.copy().astype(np.float32)
                cc, warp = cv2.findTransformECC(
                    prev_g, curr_g,
                    warp, cv2.MOTION_AFFINE,
                    (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 1e-4),
                    None, 3
                )
                if warp is not None:
                    T = warp
            except:
                pass

        prev_g = curr_g

        # accumulate
        C_prev = np.vstack([C_list[-1], [0,0,1]])
        C_now  = np.vstack([T, [0,0,1]])
        C_list.append((C_prev @ C_now)[:2,:])

    cap.release()

    # Convert to parameters
    params = np.array([mat2params(C) for C in C_list])  # [n,3]

    # Smoothing with padding (never shrinks length)
    def smooth_pad(x, r):
        if r<=0:
            return x
        k = 2*r + 1
        pad_left  = np.repeat(x[0:1], r, axis=0)
        pad_right = np.repeat(x[-1:], r, axis=0)
        xx = np.concatenate([pad_left, x, pad_right], axis=0)
        out = []
        for d in range(x.shape[1]):
            sm = np.convolve(xx[:,d], np.ones(k)/k, mode="valid")
            out.append(sm[:,None])
        return np.concatenate(out, axis=1)

    dx_s = smooth_pad(params[:,0:1], smooth_radius).reshape(-1)
    dy_s = smooth_pad(params[:,1:2], smooth_radius).reshape(-1)
    da_s = smooth_pad(params[:,2:3], smooth_radius).reshape(-1)

    # Match length
    L = len(C_list)
    dx_s = dx_s[:L]
    dy_s = dy_s[:L]
    da_s = da_s[:L]

    S_list = [params2mat(dx_s[i], dy_s[i], da_s[i]) for i in range(L)]
    return C_list, S_list


# ============ Apply transform to tracks ====================
def apply_to_tracks(video_path, mot_path, sizes_path, out_dir,
                    use_ecc=True, smooth_radius=10):

    ensure_dir(out_dir)
    stem = Path(mot_path).stem

    out_txt = Path(out_dir)/f"{stem}_stab.txt"
    out_sz  = Path(out_dir)/f"{stem}_stab_sizes.csv"

    C_list, S_list = compute_stabilization(video_path, use_ecc=use_ecc, smooth_radius=smooth_radius)

    # the effective transform = S_i @ inv(C_i)
    new_T = []
    for i in range(len(C_list)):
        Ci = np.vstack([C_list[i], [0,0,1]])
        Si = np.vstack([S_list[i], [0,0,1]])
        Ti = (Si @ np.linalg.inv(Ci))[:2,:]
        new_T.append(Ti)

    df = read_mot(mot_path)
    sizes = read_sizes(sizes_path)

    with open(out_txt,"w",encoding="utf-8") as fm, open(out_sz,"w",encoding="utf-8") as fs:
        fs.write("frame,W,H\n")

        for _, r in df.iterrows():
            f  = int(r["frame"])
            tid= int(r["id"])
            x,y,w,h = float(r["x"]), float(r["y"]), float(r["w"]), float(r["h"])
            cx, cy = x + w/2.0, y + h/2.0

            T = new_T[min(f, len(new_T)-1)]
            cx2 = T[0,0]*cx + T[0,1]*cy + T[0,2]
            cy2 = T[1,0]*cx + T[1,1]*cy + T[1,2]

            x2, y2 = cx2 - w/2.0, cy2 - h/2.0
            fm.write(
                f"{f},{tid},{x2:.2f},{y2:.2f},{w:.2f},{h:.2f},1,-1,-1,-1\n"
            )

            if f in sizes:
                W,H = sizes[f]
                fs.write(f"{f},{W},{H}\n")


# ============ CLI ===================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mot_dir", required=True)
    ap.add_argument("--video_dir", required=True)
    ap.add_argument("--out_dir", default="output/mot_stab_fbd")
    ap.add_argument("--no_ecc", action="store_true")
    ap.add_argument("--smooth_radius", type=int, default=10)

    args = ap.parse_args()

    mot_dir = Path(args.mot_dir)
    vid_dir = Path(args.video_dir)
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    for mot in sorted(mot_dir.glob("*.txt")):
        stem = mot.stem
        sizes = mot.with_name(f"{stem}_sizes.csv")
        if not sizes.exists():
            print(f"[WARN] Missing sizes: {sizes}")
            continue

        # find matching video
        vid = (vid_dir / f"{stem}.mp4")
        if not vid.exists():
            # try other extensions
            cand = list(vid_dir.glob(f"{stem}.*"))
            if not cand:
                print(f"[WARN] Video not found for {stem}")
                continue
            vid = cand[0]

        print(f"[STAB] {stem}")
        apply_to_tracks(
            vid, mot, sizes, out_dir,
            use_ecc=(not args.no_ecc),
            smooth_radius=args.smooth_radius
        )
        print(f"[OK] {stem} stabilized.")


if __name__ == "__main__":
    main()
