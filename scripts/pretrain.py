"""Pretrain SYNPRED, then run the frozen (linear-probe) evaluation.

    python scripts/pretrain.py --config configs/synpred.yaml            # seeds from the config
    python scripts/pretrain.py --config configs/synpred.yaml --seeds 42 # a single run

Each seed is an independent run (the RNGs are reset per seed), so any seed can be
rerun on its own. Writes to <output_dir>/seed_<seed>/:
    model_best_auc.pt   checkpoint selected by validation AUROC (multimodal linear probe)
    model_last.pt       final epoch
    config.json         resolved configuration
    history.csv         per-epoch train losses, validation loss and validation AUROC
    frozen_eval/        linear-probe predictions and metrics.json
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from utils import pipeline as P


def main():
    parser = P.add_common_args(argparse.ArgumentParser(description=__doc__,
                                                       formatter_class=argparse.RawDescriptionHelpFormatter))
    parser.add_argument("--seeds", type=int, nargs="+", help="Overrides pretrain.seeds, e.g. --seeds 42 43.")
    parser.add_argument("--epochs", type=int)
    args = parser.parse_args()

    cfg = P.load_config(args.config, args)
    pt = cfg["pretrain"]
    if args.seeds:
        pt["seeds"] = args.seeds
    if args.epochs is not None:
        pt["epochs"] = args.epochs

    datasets = P.build_datasets(cfg["data"])
    for seed in pt["seeds"]:
        out_dir = Path(cfg["output_dir"]) / f"seed_{seed}"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "config.json").write_text(json.dumps(
            {**cfg, "seed": seed, "rna_dim": P.rna_dim(datasets)}, indent=2))
        P.set_seed(seed)
        loaders = P.build_loaders(datasets, cfg["data"], seed)
        pretrain_seed(cfg, datasets, loaders, out_dir, seed)


def pretrain_seed(cfg, datasets, loaders, out_dir, seed):
    pt, device = cfg["pretrain"], cfg["device"]
    model = P.build_model(cfg["model"], P.rna_dim(datasets), beta=pt["beta"], device=device)
    img_params = list(model.img_branch.parameters())
    img_ids = {id(p) for p in img_params}
    optimizer = torch.optim.Adam([
        {"params": img_params, "lr": pt["img_lr"]},
        {"params": [p for p in model.parameters() if id(p) not in img_ids], "lr": pt["rna_lr"]},
    ])

    first_selectable = min(pt["min_selection_epoch"], pt["epochs"] - 1)
    best_auc, history = -np.inf, []
    progress = tqdm(range(pt["epochs"]), desc=f"pretrain seed {seed}")
    for epoch in progress:
        model.train()
        losses = []
        for batch in loaders["train"]:
            out = forward(model, batch, device, epoch)
            optimizer.zero_grad()
            out["loss_total"].backward()
            optimizer.step()
            losses.append({k: out[k].item() for k in
                           ("loss_total", "loss_rec_img", "loss_rec_rna", "loss_kl_cls", "loss_kl_style")})

        model.eval()
        with torch.no_grad():
            val_loss = np.mean([forward(model, batch, device, epoch)["loss_total"].item()
                                for batch in loaders["val"]])

        train_emb = P.extract_embeddings(model, loaders["train_clean"], device)
        val_emb = P.extract_embeddings(model, loaders["val"], device)
        val_auc = P.metrics(P.linear_probe(train_emb, {"mm": val_emb["mm"]})["mm"])["auroc"]

        row = {"epoch": epoch, **pd.DataFrame(losses).mean().to_dict(),
               "val_loss_total": val_loss, "val_auroc_mm": val_auc}
        history.append(row)
        pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)
        if epoch >= first_selectable and val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), out_dir / "model_best_auc.pt")
        progress.set_postfix(loss=f"{row['loss_total']:.4f}", val_auc=f"{val_auc:.3f}", best=f"{best_auc:.3f}")

    torch.save(model.state_dict(), out_dir / "model_last.pt")
    print(f"Best validation AUROC (mm) {best_auc:.3f}; checkpoint: {out_dir / 'model_best_auc.pt'}")

    P.load_state(model, out_dir / "model_best_auc.pt", device)
    results = P.frozen_evaluation(model, loaders, device, out_dir / "frozen_eval")
    print(pd.DataFrame(results["test"]).T.to_string(float_format="{:.3f}".format))


def forward(model, batch, device, epoch):
    """Training-objective forward pass; also used for the validation loss."""
    return model(
        img_x=batch["T0"].to(device), img_target=batch["target_T0"].to(device),
        rna_x=batch["rna"].to(device), rna_target=batch["target_rna"].to(device),
        labels=batch["pcr"].to(device), rna_missing=batch["rna_missing"].to(device),
        current_epoch=epoch, eval=False, warmup=False,
    )


if __name__ == "__main__":
    main()
