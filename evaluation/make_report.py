"""
Turns eval JSON files into results tables (markdown) and a bar chart for your
slides and report.

    python -m evaluation.make_report \
        --run "Full method=results/eval_full.json" \
        --run "Adversarial training only=results/eval_adv_only.json" \
        --run "Regularizer only=results/eval_reg_only.json" \
        --out results/report
"""
import argparse
import json
from pathlib import Path


def pct(stat):
    return f"{stat['mean']*100:.1f}% ± {stat['std']*100:.1f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True, help='"label=path/to/eval.json"')
    ap.add_argument("--out", default="results/report")
    args = ap.parse_args()

    rows = []  # (model name, summary)
    for spec in args.run:
        label, path = spec.split("=", 1)
        data = json.load(open(path))
        if "baseline" in data and not any(r[0] == "Baseline (no defense)" for r in rows):
            rows.append(("Baseline (no defense)", data["baseline"]["summary"]))
        if "hardened" in data:
            rows.append((label, data["hardened"]["summary"]))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    t1 = ["| Model | APCER (fakes accepted) | BPCER (real rejected) | ACER |",
          "|---|---|---|---|"]
    t2 = ["| Model | Attack success (8/255, 10 steps) | Attack success (stronger: 16/255, 40 steps) | "
          "Fakes accepted under random noise (control) | Real faces rejected after attack |",
          "|---|---|---|---|---|"]
    t3 = ["| Model | Accuracy on validation | Accuracy on test |", "|---|---|---|"]
    for name, s in rows:
        t1.append(f"| {name} | {pct(s['APCER'])} | {pct(s['BPCER'])} | {pct(s['ACER'])} |")
        t2.append(f"| {name} | {pct(s['ASR'])} | {pct(s['ASR_strong'])} | {pct(s['APCER_noise'])} | {pct(s['BPCER_adv'])} |")
        t3.append(f"| {name} | {pct(s['ACC_val'])} | {pct(s['ACC_test'])} |")
    md = ("**Clean performance on the test set**\n\n" + "\n".join(t1) +
          "\n\n**Robustness (lower is better in every column)**\n\n" + "\n".join(t2) +
          "\n\n**Validation vs test accuracy**\n\n" + "\n".join(t3) + "\n")
    Path(str(out) + ".md").write_text(md)
    print(md)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        names = [n for n, _ in rows]
        series = [("Attack success 8/255", "ASR", "#F5A623"),
                  ("Attack success, stronger attacker", "ASR_strong", "#D9534F"),
                  ("ACER on clean images", "ACER", "#2DD4BF")]
        x = range(len(names))
        width = 0.27
        fig, ax = plt.subplots(figsize=(10, 4.8))
        for k, (label, key, color) in enumerate(series):
            vals = [s[key]["mean"] * 100 for _, s in rows]
            errs = [s[key]["std"] * 100 for _, s in rows]
            ax.bar([i + (k - 1) * width for i in x], vals, width, yerr=errs, label=label, color=color)
        ax.set_xticks(list(x))
        ax.set_xticklabels(names, rotation=15, ha="right")
        ax.set_ylabel("% (lower is better)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(str(out) + ".png", dpi=200)
        print("chart ->", str(out) + ".png")
    except ImportError:
        print("matplotlib not installed: tables written, chart skipped")


if __name__ == "__main__":
    main()
