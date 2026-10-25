"""E.V.E.L.Y.N {Enhanced Visual Evaluation of Lip Yielding Network}

Inspired by Dame Evelyn Glennie, the percussionist who "hears" music through
vibration without relying on her ears. EVELYN ignores audio and reads the
physics of lip motion instead.

Author : Gokul Prasath
Powered by : Python, torch, mediapipe, numpy, scipy, scikit-learn and caffeine

Usage:
  python app.py                                  (interactive)
  python app.py path/to/video.mp4 [more.mp4 ...]
  python app.py video.mp4 --model results/main/evelyn_final.joblib
"""
import argparse
import os
import sys

import cv2
import numpy as np

from evelyn import cues as CU
from evelyn import kinematics as K
from evelyn import landmarks as L
from evelyn.infer import Detector
from evelyn.ui import Colors as C, banner, done, step

DEFAULT_MODEL = os.path.join("models", "evelyn_final.joblib")
SHOW = ["omega_va_raw", "omega_aj_raw", "v_persist", "a_persist", "hf_ratio", "snap_rate",
        "aperture_coherence"]
NICE = {"omega_va_raw": "Articulation bandwidth ω(v→a)", "omega_aj_raw": "Articulation bandwidth ω(a→j)",
        "v_persist": "Velocity persistence (momentum)", "a_persist": "Acceleration persistence",
        "hf_ratio": "High-frequency energy ratio", "snap_rate": "Zero-inertia snap rate",
        "aperture_coherence": "Inner/outer lip coherence"}


def draw_debug(frame_pts, path):
    frame, pts = frame_pts
    img = frame.copy()
    h, w = img.shape[:2]
    for k in L.ANCHORS:
        cv2.circle(img, (int(pts[k, 0] * w), int(pts[k, 1] * h)), 2, (255, 200, 0), -1)
    for k in L.LIPS:
        cv2.circle(img, (int(pts[k, 0] * w), int(pts[k, 1] * h)), 2, (0, 0, 255), -1)
    cv2.imwrite(path, img)


def report_png(path, segs, cfg, win_probs, spans, verdict, p):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    si = int(np.argmax([len(s["X"]) for s in segs]))
    X = segs[si]["X"]
    fps = cfg["fps"]
    t = np.arange(len(X)) / fps
    af = np.diff(X, 2, axis=0) * fps ** 2
    am = np.linalg.norm(af, axis=-1).mean(1)
    med = np.median(am)
    mad = 1.4826 * np.median(np.abs(am - med)) + 1e-12
    snaps = np.where(am > med + 6 * mad)[0]
    fig, ax = plt.subplots(3, 1, figsize=(10, 7.5), sharex=True)
    ax[0].plot(t, K.aperture(X), color="#264653")
    ax[0].set_ylabel("lip aperture\n(IOD)")
    ax[0].set_title(f"EVELYN: {verdict}  (P(fake) = {p:.2f})")
    ax[1].plot(t[1:-1], am, color="#e76f51", lw=1)
    ax[1].plot(t[1:-1][snaps], am[snaps], "kx", label="zero-inertia snap")
    ax[1].set_ylabel("|acceleration|\n(IOD/s²)")
    if len(snaps):
        ax[1].legend(loc="upper right")
    sel = spans[:, 0] == si
    for (s0, st, ln), pr in zip(spans[sel], win_probs[sel]):
        ax[2].hlines(pr, st / fps, (st + ln) / fps, color="#e76f51" if pr > 0.5 else "#2a9d8f", lw=4)
    ax[2].axhline(0.5, color="k", ls=":", lw=0.8)
    ax[2].set_ylim(0, 1)
    ax[2].set_ylabel("P(fake)\nper window")
    ax[2].set_xlabel("time (s)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def analyse(path, det, args):
    cfg = det.cfg
    print(f"\n{C.BLUE}>> ANALYZING TARGET: {os.path.basename(path)} <<{C.ENDC}")
    step("Tracking 478-point face mesh")
    d = L.extract_video(path, args.landmarker_model, args.max_seconds, return_first_frame=True)
    done(f"({len(d['t'])} frames, face found in {100 * d['valid'].mean():.0f}%)")
    if d.get("first_frame") is None:
        print(f"  {C.FAIL}>> FAILED: no face detected.{C.ENDC}")
        return
    step("Removing head motion (Procrustes alignment)")
    segs, meta = K.prepare(d, fps=cfg["fps"], dims=cfg["dims"], max_gap=cfg.get("max_gap", 0.25))
    done()
    step("Computing lip velocity / acceleration / jerk")
    w = K.windows(segs, cfg["fps"], cfg["win"], cfg["stride"], cfg.get("min_len", 32), None)
    if w is None:
        print(f"\n  {C.FAIL}>> FAILED: need at least {cfg.get('min_len', 32) / cfg['fps']:.1f} s of continuous face track.{C.ENDC}")
        return
    done(f"({len(w['C'])} windows)")

    A, _ = CU.video_cue_vector(d, w["H"])
    vid = dict(C=w["C"], M=w["M"], H=w["H"], label=0, meta=[0] * 6, A=A, group="q")
    scores = {k: float(v[0]) for k, v in det.score_videos([vid]).items()}
    wprobs = det.window_probs(w["C"], w["M"], w["H"])
    # the appearance cue is per-video, so the timeline always shows the motion cue
    wp = wprobs.get(det.primary, wprobs.get("fusion", next(iter(wprobs.values()))))
    p = scores[det.primary]

    base = os.path.splitext(os.path.basename(path))[0]
    os.makedirs(args.report_dir, exist_ok=True)
    dbg = os.path.join(args.report_dir, f"{base}_landmarks.jpg")
    draw_debug(d["first_frame"], dbg)

    print("\n" + C.BOLD + "  LIP BIOMETRIC INERTIA READOUT:" + C.ENDC)
    Hm = np.nanmean(w["H"], 0)
    ref = det.b["reference"]["real"]
    n_out = 0
    for f in SHOW:
        j = K.FEATURES.index(f)
        lo, hi = ref[0, j], ref[4, j]
        out = not (lo <= Hm[j] <= hi)
        n_out += out
        colr = C.WARNING if out else C.GREEN
        print(f"    {NICE[f]:<34s} {colr}{Hm[j]:>9.3f}{C.ENDC}   real 5–95%: [{lo:.3f}, {hi:.3f}]"
              + (f"  {C.WARNING}<< outside real range{C.ENDC}" if out else ""))

    p_mot = scores.get("fusion", p)

    cue_rows = det.cue_report([vid])
    if cue_rows:
        print("\n" + C.BOLD + "  SIX-CUE PANEL:" + C.ENDC)
        print(f"    {'cue':<8s}{'measures':<44s}{'P(fake)':>9s}{'weight':>8s}{'AUC':>7s}")
        for cue, pc, wgt, auc in cue_rows:
            colr = C.FAIL if pc >= 0.75 else (C.WARNING if pc >= 0.25 else C.GREEN)
            bar = "#" * int(round(wgt * 20))
            print(f"    {cue:<8s}{CU.CUE_DESC[cue][:43]:<44s}{colr}{pc:9.3f}{C.ENDC}"
                  f"{wgt:8.3f}{auc:7.2f}  {bar}")
        flag = [c for c, pc, wgt, _ in cue_rows if pc >= 0.65 and wgt > 0.01]
        if flag:
            print(f"    {C.WARNING}cues firing: {', '.join(flag)}{C.ENDC}")

    lo_t, hi_t = args.band
    print("\n" + C.BOLD + "  FINAL DIAGNOSTIC REPORT:" + C.ENDC)
    print("  " + "=" * 60)
    if p >= hi_t:
        verdict = ("MANIPULATION DETECTED" if cue_rows
                   else "SYNTHETIC LIP MOTION DETECTED")
        print(f"  VERDICT:     {C.FAIL}{C.BOLD}🔴 {verdict} 🔴{C.ENDC}")
    elif p <= lo_t:
        verdict = ("NO ANOMALY FOUND" if cue_rows else "NO KINEMATIC ANOMALY")
        print(f"  VERDICT:     {C.GREEN}{C.BOLD}🟢 {verdict} 🟢{C.ENDC}")
    else:
        verdict = "INCONCLUSIVE"
        print(f"  VERDICT:     {C.WARNING}{C.BOLD}⚠️  {verdict} ⚠️{C.ENDC}")
    print(f"  P(fake):     {p:.3f}   (model: {det.primary}, calibrated)")
    print(f"    lip motion physics             P={p_mot:.3f}"
          + f"   ({n_out} of {len(SHOW)} readout features outside the real range)")
    if not cue_rows:
        print(f"    {C.WARNING}six-cue panel unavailable in this model (re-extract and retrain){C.ENDC}")
    print("  Models:      " + "  ".join(f"{k}={v:.2f}" for k, v in scores.items()))
    if meta["detect_rate"] < 0.8 or meta["duration"] < 3:
        print(f"  {C.WARNING}NOTE: short clip or unstable face track; treat with caution.{C.ENDC}")
    if p <= lo_t:
        if not cue_rows:
            print(f"\n  {C.CYAN}SCOPE: only the lip MOTION was tested. If a real person performed{C.ENDC}")
            print(f"  {C.CYAN}and only the face was replaced, the motion is human and this test{C.ENDC}")
            print(f"  {C.CYAN}cannot flag it. Enable cue 2 to cover that case.{C.ENDC}")
        else:
            print(f"\n  {C.CYAN}All weighted cues are clean. None of them can rule out a full-frame{C.ENDC}")
            print(f"  {C.CYAN}regeneration, a heavily re-compressed composite, or a generator that{C.ENDC}")
            print(f"  {C.CYAN}models time explicitly. A clean result is not proof of authenticity.{C.ENDC}")
    try:
        rp = os.path.join(args.report_dir, f"{base}_report.png")
        report_png(rp, segs, cfg, wp, w["span"], verdict, p)
        print(f"  Report:      {C.UNDERLINE}{rp}{C.ENDC}")
    except Exception as e:  # plotting is optional
        print(f"  (report plot skipped: {e})")
    print(f"  Landmarks:   {C.UNDERLINE}{dbg}{C.ENDC}")
    print("  " + "=" * 60)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("videos", nargs="*")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--landmarker-model", default=None)
    ap.add_argument("--max-seconds", type=float, default=20.0)
    ap.add_argument("--band", type=float, nargs=2, default=[0.25, 0.75],
                    help="P(fake) band reported as inconclusive")
    ap.add_argument("--report-dir", default="evelyn_reports")
    args = ap.parse_args()

    banner()
    print(C.BOLD + "SYSTEM DIAGNOSTICS:" + C.ENDC)
    if not os.path.exists(args.model):
        print(f"  [-] {C.FAIL}Model not found: {args.model}{C.ENDC}\n"
              f"      Train one with train.py, then copy results/<run>/evelyn_final.joblib to {DEFAULT_MODEL}")
        return
    det = Detector(args.model)
    cv = det.b["cv_summary"].get(det.primary, {})
    print(f"  [+] Model loaded: {args.model}")
    print(f"  [+] Primary: {det.primary} | validation AUC {cv.get('auc', float('nan')):.3f} ({det.b['protocol']})")
    print(C.CYAN + "-" * 60 + C.ENDC)

    if args.videos:
        for v in args.videos:
            analyse(v, det, args)
        return
    while True:
        path = input(f"\n{C.BOLD}INPUT STREAM > {C.ENDC}").strip().strip('"')
        if path.lower() in ("q", "quit", "exit"):
            print(C.CYAN + "\nShutting down EVELYN system... Goodbye." + C.ENDC)
            break
        if not os.path.exists(path):
            print(f"{C.FAIL}ERROR: File path invalid.{C.ENDC}")
            continue
        try:
            analyse(path, det, args)
        except Exception as e:
            print(f"{C.FAIL}ERROR: {e}{C.ENDC}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nForce Quit.")
        sys.exit(0)
