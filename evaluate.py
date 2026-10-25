"""Step 4: evaluate a trained model on a DIFFERENT dataset or a degraded copy.

Use this for cross-dataset generalisation (e.g. train FF++, test Celeb-DF)
and robustness (compression / downscaling made with degrade.py).

Usage:
  python evaluate.py --model results/main/evelyn_final.joblib --cache cache/celebdf --out results/cross_celebdf
"""
import argparse
import csv
import json
import os

import numpy as np

from evelyn import data as D
from evelyn import metrics as MT
from evelyn.infer import Detector


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default=None, help="only evaluate this split (e.g. test)")
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args()

    det = Detector(args.model)
    vids, _ = D.load_dataset(args.cache, det.cfg)
    if args.split:
        vids = [v for v in vids if v["split"] == args.split]
    y = np.array([v["label"] for v in vids])
    g = np.array([v["group"] for v in vids])
    meth = np.array([v["method"] for v in vids])
    scores = det.score_videos(vids)

    os.makedirs(args.out, exist_ok=True)
    rows, full = [], {}
    for name, p in scores.items():
        m = MT.full_report(y, p, g, args.n_boot)
        full[name] = m
        rows.append(f"| {name}{' (primary)' if name == det.primary else ''} | "
                    f"{MT.fmt_ci(m.get('auc'), m.get('auc_ci'))} | "
                    f"{MT.fmt_ci(m.get('eer'), m.get('eer_ci'), pct=True)} | "
                    f"{MT.fmt_ci(m.get('bal_acc'), m.get('bal_acc_ci'), pct=True)} | {m['acc'] * 100:.1f} |")
    md = [f"# External evaluation\n\nModel: `{args.model}`  \nData: `{args.cache}` "
          f"({len(y)} videos, {(y == 0).sum()} real / {(y == 1).sum()} fake)\n",
          "| model | AUC [95% CI] | EER % | Bal. acc % | Acc % |\n|---|---|---|---|---|",
          *rows]
    if len(set(meth) - {"real"}) > 1:
        md.append("\n| method | n | " + " | ".join(scores) + " |\n|---|---|" + "---|" * len(scores))
        for mt in sorted(set(meth) - {"real"}):
            sel = (meth == mt) | (y == 0)
            md.append(f"| {mt} | {(meth == mt).sum()} | " + " | ".join(
                f"{MT.point_metrics(y[sel], p[sel]).get('auc', float('nan')):.3f}" for p in scores.values()) + " |")
    text = "\n".join(md)
    print(text)
    with open(os.path.join(args.out, "summary.md"), "w", encoding="utf-8") as fh:
        fh.write(text)
    with open(os.path.join(args.out, "results.json"), "w") as fh:
        json.dump(full, fh, indent=1, default=float)
    with open(os.path.join(args.out, "predictions.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["video", "label", "method", "group"] + list(scores))
        for i, v in enumerate(vids):
            w.writerow([v["id"], y[i], meth[i], g[i]] + [f"{scores[k][i]:.5f}" for k in scores])


if __name__ == "__main__":
    main()
