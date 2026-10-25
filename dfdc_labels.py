"""Turn DFDC metadata into the labels CSV that extract_landmarks.py reads.

DFDC does not put videos in real/ and fake/ folders. Each chunk folder holds a
metadata.json mapping every file name to its label, and every fake names the
real video it was made from ("original"). This script:

  * reads every metadata.json (or dataset.json) under the given folder
  * labels each video 0 = real, 1 = fake
  * groups each fake with the real video it was made from, so a fake and its
    source can never be split across train and test. Getting this wrong is the
    single easiest way to produce a meaningless score on DFDC.
  * optionally caps the number of fakes per real video, because DFDC has several
    fakes for every real one and an unbalanced set wastes extraction time

Usage:
  python dfdc_labels.py --data D:\\dfdc --out dfdc_labels.csv
  python dfdc_labels.py --data D:\\dfdc --out dfdc_labels.csv --max-fakes-per-real 1
  python extract_landmarks.py --data D:\\dfdc --labels dfdc_labels.csv --out cache\\dfdc
  python train.py --cache cache\\dfdc --out results\\dfdc
"""
import argparse
import csv
import json
import os
import random
from collections import defaultdict

LABEL_WORDS = {"real": 0, "fake": 1, "0": 0, "1": 1, "original": 0, "manipulated": 1}


def parse_label(v):
    if isinstance(v, (int, float)):
        return int(v)
    return LABEL_WORDS.get(str(v).strip().lower())


def stem(name):
    return os.path.splitext(os.path.basename(str(name)))[0]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="DFDC root (one or more chunk folders inside)")
    ap.add_argument("--out", required=True, help="labels CSV to write")
    ap.add_argument("--max-fakes-per-real", type=int, default=0,
                    help="keep at most N fakes per source video (0 = keep all)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    if not os.path.isdir(args.data):
        raise SystemExit(f"Folder not found: {args.data}\nDownload and extract the data first, then point --data at it.")

    rows, missing, meta_files = [], 0, 0
    for dp, _, files in os.walk(args.data):
        for fn in files:
            if fn.lower() not in ("metadata.json", "dataset.json"):
                continue
            meta_files += 1
            with open(os.path.join(dp, fn), encoding="utf-8") as fh:
                meta = json.load(fh)
            for vid, info in meta.items():
                if not isinstance(info, dict):
                    continue
                lab = parse_label(info.get("label"))
                if lab is None:
                    continue
                path = os.path.join(dp, vid)
                if not os.path.exists(path):
                    missing += 1
                    continue
                src = info.get("original") or info.get("source") or info.get("target")
                split = str(info.get("split") or info.get("set") or "").lower()
                split = {"train": "train", "training": "train", "test": "test",
                         "val": "val", "validation": "val"}.get(split, "")
                rows.append(dict(path=os.path.relpath(path, args.data), label=lab,
                                 source=stem(src) if (lab == 1 and src) else stem(vid),
                                 split=split))
    if not rows:
        raise SystemExit(f"No labelled videos found. Looked for metadata.json under {args.data} "
                         f"({meta_files} metadata files found, {missing} listed videos missing on disk).")

    # every fake joins the group of the real video it was made from
    by_src = defaultdict(list)
    for r in rows:
        by_src[r["source"]].append(r)
    kept = []
    for src, members in by_src.items():
        reals = [m for m in members if m["label"] == 0]
        fakes = [m for m in members if m["label"] == 1]
        if args.max_fakes_per_real and len(fakes) > args.max_fakes_per_real:
            fakes = rng.sample(fakes, args.max_fakes_per_real)
        for m in reals + fakes:
            m["group"] = src
            m["method"] = "real" if m["label"] == 0 else "dfdc"
            kept.append(m)

    # DFDC splits are chunk-level; a group must not straddle two splits
    split_of = defaultdict(set)
    for m in kept:
        split_of[m["group"]].add(m["split"])
    clashes = [g for g, s in split_of.items() if len(s - {""}) > 1]
    if clashes:
        for m in kept:
            if m["group"] in clashes:
                m["split"] = ""
        print(f"  NOTE: {len(clashes)} source videos had members in different splits; "
              f"their split labels were cleared so cross-validation handles them safely.")

    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["path", "label", "group", "method", "split"])
        w.writeheader()
        for m in kept:
            w.writerow({k: m[k] for k in ["path", "label", "group", "method", "split"]})

    n_real = sum(1 for m in kept if m["label"] == 0)
    n_fake = len(kept) - n_real
    print(f"Read {meta_files} metadata file(s). Wrote {len(kept)} videos to {args.out}")
    print(f"  real {n_real} | fake {n_fake} | source groups {len(by_src)}"
          + (f" | {missing} listed videos not on disk (skipped)" if missing else ""))
    if n_real and n_fake / n_real > 3:
        print(f"  NOTE: {n_fake / n_real:.1f} fakes per real. Consider --max-fakes-per-real 1 "
              f"to balance the set and cut extraction time.")


if __name__ == "__main__":
    main()
