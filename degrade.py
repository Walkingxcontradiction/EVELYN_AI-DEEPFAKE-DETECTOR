"""Make a degraded copy of a dataset for robustness experiments.

Every video is re-encoded with the same codec, so also make a "--none" copy
as the control (re-encoding alone changes things).

Usage:
  python degrade.py --data dataset/test --out dataset_jpeg50 --jpeg 50
  python degrade.py --data dataset/test --out dataset_half --scale 0.5
  python degrade.py --data dataset/test --out dataset_ctrl --none
Then: extract_landmarks.py on the new folder -> evaluate.py with the trained model.
"""
import argparse
import os

import cv2
import numpy as np

from evelyn.landmarks import VIDEO_EXT


def degrade(frame, a, rng):
    if a.scale and a.scale != 1.0:
        h, w = frame.shape[:2]
        small = cv2.resize(frame, (max(8, int(w * a.scale)), max(8, int(h * a.scale))), interpolation=cv2.INTER_AREA)
        frame = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    if a.blur:
        k = a.blur | 1
        frame = cv2.GaussianBlur(frame, (k, k), 0)
    if a.noise:
        frame = np.clip(frame + rng.normal(0, a.noise, frame.shape), 0, 255).astype(np.uint8)
    if a.jpeg:
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, a.jpeg])
        frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return frame


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--jpeg", type=int, default=0, help="JPEG quality per frame (e.g. 30, 50, 70)")
    ap.add_argument("--scale", type=float, default=1.0, help="down-then-up scale factor (e.g. 0.5)")
    ap.add_argument("--blur", type=int, default=0, help="Gaussian kernel size")
    ap.add_argument("--noise", type=float, default=0.0, help="Gaussian pixel noise sigma")
    ap.add_argument("--fps", type=float, default=0.0, help="temporal subsample to this fps")
    ap.add_argument("--none", action="store_true", help="re-encode only (control)")
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    n = 0
    for dp, _, files in os.walk(a.data):
        for fn in files:
            if not fn.lower().endswith(VIDEO_EXT):
                continue
            src = os.path.join(dp, fn)
            dst = os.path.join(a.out, os.path.relpath(src, a.data))
            dst = os.path.splitext(dst)[0] + ".mp4"
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            cap = cv2.VideoCapture(src)
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            out_fps = a.fps if a.fps else fps
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            wr = cv2.VideoWriter(dst, cv2.VideoWriter_fourcc(*"mp4v"), out_fps, (w, h))
            i, nxt = 0, 0.0
            while True:
                ok, fr = cap.read()
                if not ok:
                    break
                t = i / fps
                i += 1
                if a.fps and t + 1e-9 < nxt:
                    continue
                nxt += 1.0 / out_fps
                wr.write(fr if a.none else degrade(fr, a, rng))
            cap.release()
            wr.release()
            n += 1
            print(f"  {n}: {dst}", end="\r")
    print(f"\nWrote {n} videos to {a.out}")


if __name__ == "__main__":
    main()
