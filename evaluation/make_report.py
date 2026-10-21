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
          "Fakes accepted under random noise (control) | Real faces rejected after attack | "
          "Worst case (fake fools it clean OR under any attack) |",
          "|---|---|---|---|---|---|"]
    t3 = ["| Model | Accuracy on validation | Accuracy on test |", "|---|---|---|"]
    for name, s in rows:
        t1.append(f"| {name} | {pct(s['APCER'])} | {pct(s['BPCER'])} | {pct(s['ACER'])} |")
        t2.append(f"| {name} | {pct(s['ASR'])} | {pct(s['ASR_strong'])} | {pct(s['APCER_noise'])} | {pct(s['BPCER_adv'])} | {pct(s['ASR_worst'])} |")
        t3.append(f"| {name} | {pct(s['ACC_val'])} | {pct(s['ACC_test'])} |")
    sweep_keys = sorted({k for _, s in rows for k in s if k.startswith("ASR_eps")},
                        key=lambda k: float(k[len("ASR_eps"):]))
    t4 = []
    if sweep_keys:
        t4 = ["| Model | " + " | ".join(f"{k[len('ASR_eps'):]}/255" for k in sweep_keys) + " |",
              "|---|" + "---|" * len(sweep_keys)]
        for name, s in rows:
            t4.append(f"| {name} | " + " | ".join(pct(s[k]) if k in s else "-" for k in sweep_keys) + " |")
    cond_keys = sorted({k for _, s in rows for k in s if k.startswith("ASRc_eps")},
                       key=lambda k: float(k[len("ASRc_eps"):]))
    t5 = []
    if cond_keys:
        t5 = ["| Model | " + " | ".join(f"{k[len('ASRc_eps'):]}/255" for k in cond_keys) + " |",
              "|---|" + "---|" * len(cond_keys)]
        for name, s in rows:
            t5.append(f"| {name} | " + " | ".join(pct(s[k]) if k in s else "-" for k in cond_keys) + " |")
    md = ("**Clean performance on the test set**\n\n" + "\n".join(t1) +
          "\n\n**Robustness (lower is better in every column)**\n\n" + "\n".join(t2) +
          "\n\n**Validation vs test accuracy**\n\n" + "\n".join(t3) + "\n" +
          ("\n**Attack success vs attack budget (10 steps)**\n\n" + "\n".join(t4) + "\n" if t4 else "") +
          ("\n**Attack success on fakes the model caught WITHOUT an attack (10 steps)**\n\n" +
           "\n".join(t5) + "\n" if t5 else ""))
    Path(str(out) + ".md").write_text(md)
    print(md)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        names = [n for n, _ in rows]
        series = [("Attack success 8/255", "ASR", "#F5A623"),
                  ("Attack success, stronger attacker", "ASR_strong", "#D9534F"),
                  ("Worst case", "ASR_worst", "#7C3AED"),
                  ("Real faces rejected after attack", "BPCER_adv", "#3B82F6"),
                  ("ACER on clean images", "ACER", "#2DD4BF")]
        x = range(len(names))
        width = 0.16
        fig, ax = plt.subplots(figsize=(11, 6.2))
        for k, (label, key, color) in enumerate(series):
            vals = [s[key]["mean"] * 100 for _, s in rows]
            errs = [s[key]["std"] * 100 for _, s in rows]
            ax.bar([i + (k - 2) * width for i in x], vals, width, yerr=errs, label=label, color=color)
        ax.set_xticks(list(x))
        ax.set_xticklabels(names, rotation=12, ha="right", fontsize=9)
        ax.set_ylabel("% (lower is better)")
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.22), ncol=2, frameon=False)
        fig.tight_layout()
        fig.savefig(str(out) + ".png", dpi=200)
        print("chart ->", str(out) + ".png")
    except ImportError:
        print("matplotlib not installed: tables written, chart skipped")


if __name__ == "__main__":
    main()
