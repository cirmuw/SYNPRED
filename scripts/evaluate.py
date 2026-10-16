"""Frozen evaluation of a pretrained SYNPRED checkpoint (linear probe).

    python scripts/evaluate.py --checkpoint results/synpred/seed_42                    # model_best_auc.pt
    python scripts/evaluate.py --checkpoint results/synpred/seed_42 --backbone last

A logistic regression is fitted on multimodal train embeddings ([z_fusion ; z_uni],
fused across modalities by product of experts) and tested on multimodal, image-only
and RNA-only embeddings without retraining. Writes predictions_{val,test}_{mode}.csv
and metrics.json to --output-dir (default: <checkpoint dir>/frozen_eval, or
frozen_eval_last for the last-epoch backbone).
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from utils import pipeline as P


def main():
    parser = P.add_common_args(argparse.ArgumentParser(description=__doc__,
                                                       formatter_class=argparse.RawDescriptionHelpFormatter))
    parser.add_argument("--checkpoint", required=True, help="Backbone checkpoint file or run folder (seed_<seed>/).")
    parser.add_argument("--backbone", choices=sorted(P.BACKBONE_FILES), default="best_auc",
                        help="Checkpoint to use from a run folder.")
    parser.add_argument("--seed", type=int, default=42, help="Only affects the train-loader shuffle.")
    args = parser.parse_args()

    out_override = args.output_dir
    args.output_dir = None
    cfg = P.load_config(args.config, args)
    checkpoint, backbone = P.resolve_checkpoint(args.checkpoint, args.backbone)
    print(f"Backbone ({backbone}): {checkpoint}")
    out_dir = Path(out_override) if out_override else checkpoint.parent / P.output_name("frozen_eval", backbone)
    device = cfg["device"]

    P.set_seed(args.seed)
    datasets = P.build_datasets(cfg["data"])
    loaders = P.build_loaders(datasets, cfg["data"], args.seed)
    model_cfg = P.model_config_for_checkpoint(cfg, checkpoint)
    model = P.load_state(P.build_model(model_cfg, P.rna_dim(datasets), device=device),
                         checkpoint, device).eval()

    results = P.frozen_evaluation(model, loaders, device, out_dir)
    print(f"Test metrics (linear probe) -> {out_dir}")
    print(pd.DataFrame(results["test"]).T.to_string(float_format="{:.3f}".format))


if __name__ == "__main__":
    main()
