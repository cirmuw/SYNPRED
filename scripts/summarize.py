"""Summarise test AUROC over pretraining seeds (mean and bootstrap 95% CI).

    python scripts/summarize.py --root results/synpred

Reads <root>/seed_*/{frozen_eval,finetune}/metrics.json (and the *_last folders of
last-epoch backbones, where present) and writes summary.csv to --root. The CI is a bootstrap over the per-seed values
(2000 resamples, seed 0), as reported in the paper.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def bootstrap_ci(values, ci=0.95, n_boot=2000, seed=0):
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = [rng.choice(values, size=len(values), replace=True).mean() for _ in range(n_boot)]
    alpha = (1 - ci) / 2
    return float(np.quantile(means, alpha)), float(np.quantile(means, 1 - alpha))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default="results/synpred")
    parser.add_argument("--metric", default="auroc")
    args = parser.parse_args()

    rows = []
    for setting in ("frozen_eval", "finetune", "frozen_eval_last", "finetune_last"):
        for path in sorted(Path(args.root).glob(f"seed_*/{setting}/metrics.json")):
            test = json.loads(path.read_text())["test"]
            for mode, m in test.items():
                rows.append({"setting": setting, "seed": path.parts[-3], "mode": mode, args.metric: m[args.metric]})
    if not rows:
        raise SystemExit(f"No metrics.json found under {args.root}/seed_*/")

    df = pd.DataFrame(rows)
    summary = []
    for (setting, mode), g in df.groupby(["setting", "mode"]):
        low, high = bootstrap_ci(g[args.metric])
        summary.append({"setting": setting, "mode": mode, "n_seeds": len(g),
                        "mean": g[args.metric].mean(), "ci_low": low, "ci_high": high})
    summary = pd.DataFrame(summary)
    summary.to_csv(Path(args.root) / "summary.csv", index=False)
    print(df.pivot_table(index="seed", columns=["setting", "mode"], values=args.metric)
          .to_string(float_format="{:.3f}".format))
    print()
    print(summary.to_string(index=False, float_format="{:.3f}".format))


if __name__ == "__main__":
    main()
