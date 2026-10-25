"""Step 3: dataset audit, physics-feature statistics, and paper figures.

Usage:
  python analyze.py --cache cache/main --out results/main
  (run after train.py to also get ROC curves from oof_predictions.csv)

Writes:
  audit.md             class-wise metadata (fps, resolution, face size ...) -> shortcut check
  feature_stats.md/csv per-feature real vs fake: medians, AUC, Cliff's delta,
                       Mann-Whitney p (Holm-corrected), video- and identity-level
  fig_features.png     distributions of the physics features
  fig_trajectory.png   example real vs fake lip trajectories + acceleration
  fig_roc.png          ROC curves of all models (needs oof_predictions.csv)
"""
import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import mannwhitneyu
from sklearn.metrics import roc_auc_score, roc_curve

from evelyn import data as D
from evelyn import kinematics as K

UNITS = {"v_mean": "IOD/s", "v_p95": "IOD/s", "a_mean": "IOD/s²", "a_p95": "IOD/s²",
         "j_mean": "IOD/s³", "j_p95": "IOD/s³", "omega_va": "rad/s", "omega_aj": "rad/s",
         "omega_va_raw": "rad/s", "omega_aj_raw": "rad/s", "aperture_std": "IOD",
         "rigid_residual": "IOD"}


def holm(p):
    p = np.asarray(p)
    order = np.argsort(p)
    adj = np.empty_like(p)
    run = 0.0
    for r, i in enumerate(order):
        run = max(run, (len(p) - r) * p[i])
        adj[i] = min(1.0, run)
    return adj


def audit(rows, out):
    lines = ["# Dataset audit (shortcut check)\n",
             "If real and fake differ strongly here, a classifier can cheat. "
             "Kinematics are resampled to a common rate and normalised by inter-ocular "
             "distance, but face size (iod_px) still changes landmark noise.\n",
             "| quantity | real median [IQR] | fake median [IQR] | AUC (fake>real) |", "|---|---|---|---|"]
    y = np.array([r["label"] for r in rows])
    warn = []
    for k in ["fps", "width", "height", "duration", "n_frames", "detect_rate", "iod_px"]:
        x = np.array([r[k] for r in rows], float)
        ok = np.isfinite(x)
        q = lambda a: f"{np.median(a):.3g} [{np.percentile(a, 25):.3g}, {np.percentile(a, 75):.3g}]" if len(a) else "n/a"
        auc = roc_auc_score(y[ok], x[ok]) if len(set(y[ok])) == 2 and np.ptp(x[ok]) > 0 else 0.5
        lines.append(f"| {k} | {q(x[ok & (y == 0)])} | {q(x[ok & (y == 1)])} | {auc:.3f} |")
        if abs(auc - 0.5) > 0.2:
            warn.append(k)
    fr = {}
    for r in rows:
        fr.setdefault(round(r["fps"], 1), [0, 0])[r["label"]] += 1
    lines += ["\n**Frame-rate distribution** (fps: real / fake): " +
              ", ".join(f"{k}: {v[0]}/{v[1]}" for k, v in sorted(fr.items()))]
    if warn:
        lines.append(f"\n> **WARNING:** strong class difference in {', '.join(warn)} "
                     f"(|AUC−0.5| > 0.2). Balance these (e.g. re-encode/resize all videos the same way, "
                     f"match durations) or report results stratified by them.")
    else:
        lines.append("\nNo strong metadata differences between classes.")
    with open(os.path.join(out, "audit.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    print("\n".join(lines))


def feature_stats(vids, out):
    y = np.array([v["label"] for v in vids])
    g = np.array([v["group"] for v in vids])
    Hv = np.array([np.nanmean(v["H"], 0) for v in vids])
    res = []
    for j, f in enumerate(K.FEATURES):
        x = Hv[:, j]
        ok = np.isfinite(x)
        a, b = x[ok & (y == 0)], x[ok & (y == 1)]
        u, p = mannwhitneyu(b, a, alternative="two-sided")
        auc = u / (len(a) * len(b))
        # identity level: mean per (group, class)
        ga, gb = [], []
        for gg in np.unique(g):
            s = (g == gg) & ok
            if (s & (y == 0)).any():
                ga.append(x[s & (y == 0)].mean())
            if (s & (y == 1)).any():
                gb.append(x[s & (y == 1)].mean())
        pg = mannwhitneyu(gb, ga).pvalue if len(ga) > 2 and len(gb) > 2 else np.nan
        grp = next(k for k, fs in K.FEATURE_GROUPS.items() if f in fs)
        res.append(dict(feature=f, group=grp, unit=UNITS.get(f, ""), real_median=np.median(a),
                        fake_median=np.median(b), auc=auc, cliffs_delta=2 * auc - 1, p=p, p_identity=pg))
    ph = holm([r["p"] for r in res])
    pgh = holm([r["p_identity"] if np.isfinite(r["p_identity"]) else 1.0 for r in res])
    for r, a, b in zip(res, ph, pgh):
        r["p_holm"], r["p_identity_holm"] = a, b
    with open(os.path.join(out, "feature_stats.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(res[0]))
        w.writeheader()
        w.writerows(res)
    lines = ["# Physics feature statistics (video level)\n",
             f"{(y == 0).sum()} real vs {(y == 1).sum()} fake videos. AUC = P(fake > real). "
             "Cliff's δ = 2·AUC−1 (|δ| ≥ 0.147 small, ≥ 0.33 medium, ≥ 0.474 large). "
             "p: two-sided Mann–Whitney U, Holm-corrected over all features. "
             "p (identity): same test on per-identity means (conservative, handles non-independence).\n",
             "| feature | group | unit | real median | fake median | AUC | Cliff's δ | p (Holm) | p identity (Holm) |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(res, key=lambda r: -abs(r["cliffs_delta"])):
        lines.append(f"| {r['feature']} | {r['group']} | {r['unit']} | {r['real_median']:.4g} | "
                     f"{r['fake_median']:.4g} | {r['auc']:.3f} | {r['cliffs_delta']:+.3f} | "
                     f"{r['p_holm']:.2e} | {r['p_identity_holm']:.2e} |")
    with open(os.path.join(out, "feature_stats.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    print("\n".join(lines[:4] + lines[4:]))
    return res, Hv, y


def fig_features(res, Hv, y, out):
    top = sorted(res, key=lambda r: -abs(r["cliffs_delta"]))[:8]
    fig, axs = plt.subplots(2, 4, figsize=(14, 6.5))
    for ax, r in zip(axs.flat, top):
        j = K.FEATURES.index(r["feature"])
        data = [Hv[(y == c) & np.isfinite(Hv[:, j]), j] for c in (0, 1)]
        vp = ax.violinplot(data, showmedians=True)
        for body, c in zip(vp["bodies"], ["#2a9d8f", "#e76f51"]):
            body.set_facecolor(c)
            body.set_alpha(0.6)
        ax.set_xticks([1, 2], ["real", "fake"])
        ax.set_title(f"{r['feature']}\nδ={r['cliffs_delta']:+.2f}", fontsize=10)
        ax.set_ylabel(r["unit"])
        if r["feature"].startswith(("omega", "j_", "a_")) and min(d.min() for d in data if len(d)) > 0:
            ax.set_yscale("log")
    fig.suptitle("Lip kinematic features: real vs fake (strongest effects)")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_features.png"), dpi=200)
    plt.close(fig)


def fig_trajectory(cache, rows, cfg, out):
    """Real video and a fake from the same identity group, side by side."""
    by_g = {}
    for r in rows:
        by_g.setdefault(r["group"], {}).setdefault(r["label"], []).append(r)
    pair = next(((v[0][0], v[1][0]) for v in by_g.values() if 0 in v and 1 in v), None)
    if pair is None:
        reals = [r for r in rows if r["label"] == 0]
        fakes = [r for r in rows if r["label"] == 1]
        if not reals or not fakes:
            return
        pair = (reals[0], fakes[0])
    fig, axs = plt.subplots(2, 2, figsize=(13, 6), sharex=True)
    for c, r in enumerate(pair):
        _, _, segs = D.video_features(os.path.join(cache, r["cache"]), cfg)
        if not segs:
            continue
        X = max(segs, key=lambda s: len(s["X"]))["X"]
        _, acc, _ = K.sg_derivs(X, cfg["fps"])
        e = K.EDGE
        X, acc = X[e:-e], acc[e:-e]          # drop the one-sided edge estimates
        t = np.arange(len(X)) / cfg["fps"]
        ap = K.aperture(X)
        am = np.linalg.norm(acc, axis=-1).mean(1)
        col = "#2a9d8f" if r["label"] == 0 else "#e76f51"
        axs[0, c].plot(t, ap, color=col)
        axs[0, c].set_title(f"{'REAL' if r['label'] == 0 else 'FAKE'}: {os.path.basename(r['rel'])}")
        axs[0, c].set_ylabel("inner lip aperture (IOD)")
        axs[1, c].plot(t, am, color=col)
        axs[1, c].set_ylabel("mean |acceleration| (IOD/s²)")
        axs[1, c].set_xlabel("time (s)")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_trajectory.png"), dpi=200)
    plt.close(fig)


def fig_roc(out):
    p = os.path.join(out, "oof_predictions.csv")
    if not os.path.exists(p):
        return
    with open(p, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    y = np.array([int(r["label"]) for r in rows])
    names = [k for k in rows[0] if k not in ("video", "label", "method", "group") and "[" not in k]
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for k in names:
        s = np.array([float(r[k]) for r in rows])
        fpr, tpr, _ = roc_curve(y, s)
        ax.plot(fpr, tpr, label=f"{k} (AUC {roc_auc_score(y, s):.3f})", lw=2 if k in ("tcn", "fusion") else 1.2)
    ax.plot([0, 1], [0, 1], "k:", lw=0.8)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC (out-of-fold, video level)")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_roc.png"), dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fps", type=float, default=25.0)
    ap.add_argument("--win", type=int, default=64)
    ap.add_argument("--stride", type=int, default=32)
    ap.add_argument("--max-windows", type=int, default=8)
    ap.add_argument("--dims", type=int, default=2)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rows = [r for r in D.read_index(args.cache) if r.get("status", "ok") == "ok"]
    audit(rows, args.out)
    cfg = {**D.DEFAULT_CFG, **dict(fps=args.fps, win=args.win, stride=args.stride,
                                    max_windows=args.max_windows, dims=args.dims)}
    vids, cfg = D.load_dataset(args.cache, cfg)
    res, Hv, y = feature_stats(vids, args.out)
    fig_features(res, Hv, y, args.out)
    fig_trajectory(args.cache, rows, cfg, args.out)
    fig_roc(args.out)
    print(f"\nFigures and tables written to {args.out}")


if __name__ == "__main__":
    main()
