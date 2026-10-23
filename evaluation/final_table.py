"""
Final comparison of every defense variant: one markdown table and one trade-off chart
(clean error vs attack success at 2/255), from evaluate.py outputs.

    python -m evaluation.final_table --run "Label=results/final/test_x.json" ... --out results/final/final
"""
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLS = [("APCER", "Clean APCER"), ("BPCER", "Clean BPCER"), ("ACER", "Clean ACER"),
        ("ASRc_eps1", "Attack success 1/255*"), ("ASRc_eps2", "Attack success 2/255*"),
        ("ASRc_eps4", "Attack success 4/255*"), ("ASRc_eps8", "Attack success 8/255*")]


def fmt(s, k):
    m, sd = s[k]["mean"] * 100, s[k]["std"] * 100
    return f"{m:.1f}%" + (f" ± {sd:.1f}" if sd > 0 else "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True, help='"label=path/to/eval.json"')
    ap.add_argument("--out", default="results/final/final")
    args = ap.parse_args()

    rows, pts, base_done = [], [], False
    for spec in args.run:
        label, path = spec.split("=", 1)
        rep = json.load(open(path))
        if not base_done and "baseline" in rep:
            b = rep["baseline"]
            rows.append(("Baseline (no defense)", len(b["per_seed"]), b["summary"]))
            pts.append(("Baseline", b["summary"]))
            base_done = True
        h = rep["hardened"]
        rows.append((label, len(h["per_seed"]), h["summary"]))
        pts.append((label, h["summary"]))

    lines = ["| Model | Seeds | " + " | ".join(c for _, c in COLS) + " |",
             "|---|---|" + "---|" * len(COLS)]
    for label, n, s in rows:
        lines.append(f"| {label} | {n} | " + " | ".join(fmt(s, k) for k, _ in COLS) + " |")
    lines.append("")
    lines.append("*Share of the fakes a model rejects without attack that a 10-step PGD attack then flips "
                 "(lower is better). Mean ± std over seeds; test set, threshold 0.5.")
    md = "\n".join(lines)
    open(args.out + ".md", "w").write(md + "\n")
    print(md)

    fig, ax = plt.subplots(figsize=(6, 4.2))
    for label, s in pts:
        x, y = s["ACER"]["mean"] * 100, s["ASRc_eps2"]["mean"] * 100
        ax.errorbar(x, y, xerr=s["ACER"]["std"] * 100, yerr=s["ASRc_eps2"]["std"] * 100, fmt="o", capsize=3)
        ax.annotate(label, (x, y), textcoords="offset points", xytext=(6, 4), fontsize=8)
    ax.set_xlabel("clean ACER on test (%)  (lower is better)")
    ax.set_ylabel("attack success at 2/255 on caught fakes (%)")
    ax.set_title("Clean error vs robustness of each model")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.out + "_tradeoff.png", dpi=200)
    print("chart ->", args.out + "_tradeoff.png")


if __name__ == "__main__":
    main()
