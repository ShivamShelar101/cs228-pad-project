"""
Turns eval JSON files into a results table (markdown) and a bar chart for your
slides and report.

    python -m evaluation.make_report \
        --run "Baseline vs full method=results/eval_full.json" \
        --run "Adversarial training only=results/eval_adv_only.json" \
        --run "Regularizer only=results/eval_reg_only.json" \
        --out results/report
"""
import argparse
import json
from pathlib import Path


def fmt(stat, pct=True):
    m, s = stat["mean"], stat["std"]
    return f"{m*100:.1f}% ± {s*100:.1f}" if pct else f"{m:.3f} ± {s:.3f}"


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
    lines = ["| Model | APCER (fakes accepted) | BPCER (real rejected) | ACER | Attack success rate |",
             "|---|---|---|---|---|"]
    for name, s in rows:
        lines.append(f"| {name} | {fmt(s['APCER'])} | {fmt(s['BPCER'])} | {fmt(s['ACER'])} | {fmt(s['ASR'])} |")
    md = "\n".join(lines)
    Path(str(out) + ".md").write_text(md + "\n")
    print(md)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        names = [n for n, _ in rows]
        asr = [s["ASR"]["mean"] * 100 for _, s in rows]
        acer = [s["ACER"]["mean"] * 100 for _, s in rows]
        asr_err = [s["ASR"]["std"] * 100 for _, s in rows]
        x = range(len(names))
        fig, ax = plt.subplots(figsize=(9, 4.5))
        ax.bar([i - 0.2 for i in x], asr, 0.4, yerr=asr_err, label="Attack success rate (lower is better)", color="#F5A623")
        ax.bar([i + 0.2 for i in x], acer, 0.4, label="ACER on clean images (lower is better)", color="#2DD4BF")
        ax.set_xticks(list(x))
        ax.set_xticklabels(names, rotation=15, ha="right")
        ax.set_ylabel("%")
        ax.legend()
        fig.tight_layout()
        fig.savefig(str(out) + ".png", dpi=200)
        print("chart ->", str(out) + ".png")
    except ImportError:
        print("matplotlib not installed: table written, chart skipped")


if __name__ == "__main__":
    main()
