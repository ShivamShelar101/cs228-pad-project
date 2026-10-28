"""
Summarizes the Kiosk Guard real-world test log (app/backend/test_log.csv) into a table and a chart:
for each condition, how often each model was RIGHT (live accepted for real faces, spoof rejected for fakes).

    python -m evaluation.summarize_test_log --log app/backend/test_log.csv --out results/realworld
"""
import argparse
import csv
import os
from collections import OrderedDict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default="app/backend/test_log.csv")
    ap.add_argument("--out", default="results/realworld")
    args = ap.parse_args()
    rows = list(csv.DictReader(open(args.log)))
    groups = OrderedDict()
    for r in rows:
        groups.setdefault((r["truth"], r["condition"].strip()), []).append(r)

    lines = ["| Truth | Condition | Tries | Baseline correct | Hardened correct |", "|---|---|---|---|---|"]
    labels, base_acc, hard_acc = [], [], []
    for (truth, cond), rs in sorted(groups.items()):
        n = len(rs)
        b = sum(r["baseline_says"] == truth for r in rs)
        h = sum(r["hardened_says"] == truth for r in rs)
        lines.append(f"| {truth} | {cond} | {n} | {b}/{n} ({100*b/n:.0f}%) | {h}/{n} ({100*h/n:.0f}%) |")
        labels.append(f"{cond} ({truth})"); base_acc.append(100 * b / n); hard_acc.append(100 * h / n)
    lines.append("")
    lines.append("Correct = a real face accepted as live, or a fake rejected as spoof; thresholds chosen on validation.")
    os.makedirs(args.out, exist_ok=True)
    md = "\n".join(lines)
    open(os.path.join(args.out, "realworld.md"), "w").write(md + "\n")
    print(md)

    fig, ax = plt.subplots(figsize=(7, 0.45 * len(labels) + 1.2))
    y = range(len(labels))
    ax.barh([i + 0.2 for i in y], base_acc, height=0.4, label="baseline", color="#E8705A")
    ax.barh([i - 0.2 for i in y], hard_acc, height=0.4, label="hardened", color="#2A9D8F")
    ax.set_yticks(list(y)); ax.set_yticklabels(labels, fontsize=8); ax.invert_yaxis()
    ax.set_xlim(0, 100); ax.set_xlabel("correct decisions (%)")
    ax.set_title("Kiosk Guard: real-world tests by condition"); ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(args.out, "realworld.png"), dpi=200)
    print("chart ->", os.path.join(args.out, "realworld.png"))


if __name__ == "__main__":
    main()
