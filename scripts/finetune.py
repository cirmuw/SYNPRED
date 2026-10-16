"""Fine-tune MLP prediction heads on a pretrained SYNPRED checkpoint.

    python scripts/finetune.py --checkpoint results/synpred/seed_42                    # finetune.backbone
    python scripts/finetune.py --checkpoint results/synpred/seed_42 --backbone last
    python scripts/finetune.py --checkpoint results/synpred/seed_42/model_best_auc.pt

One head per evaluation mode (mm, img, rna). By default the backbone is frozen and
the img/rna heads are trained on multimodal embeddings, then applied to single-modality
inputs. Heads are selected by validation loss. Writes model_best_{mode}.pt,
predictions_{val,test}_{mode}.csv and metrics.json to --output-dir
(default: <checkpoint dir>/finetune, or finetune_last for the last-epoch backbone).

--checkpoint is a checkpoint file or a run folder. For a folder, the backbone is the
checkpoint selected by validation AUROC (model_best_auc.pt, the default) or the last
epoch (finetune.backbone in the config, or --backbone).
"""

import argparse
import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

from utils import pipeline as P


def finetune_mode(mode, backbone_ckpt, model_cfg, ft, datasets, loaders, device):
    P.set_seed(ft["seed"])
    base = P.load_state(P.build_model(model_cfg, P.rna_dim(datasets), device=device), backbone_ckpt, device)
    model = P.FinetuneModel(base, in_dim=2 * model_cfg["z_cls_dim"], hidden_dim=ft["head_hidden_dim"]).to(device)
    if ft["freeze_backbone"]:
        model.base_model.requires_grad_(False)
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=ft["lr"])
    train_mode = "mm" if ft["train_on_mm_features"] else mode

    best_loss, best_state = np.inf, None
    for _ in tqdm(range(ft["epochs"]), desc=f"finetune {mode}"):
        model.train()
        for batch in loaders["train"]:
            logits, rows = model.logits(batch, train_mode, mode, device, ft["mm_paired_only"])
            if logits is None:
                continue
            y = batch["pcr"].float().to(device)[rows]
            loss = F.binary_cross_entropy_with_logits(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        _, val_loss = P.predict_head(model, loaders["val"], mode, device, ft["mm_paired_only"])
        if val_loss < best_loss:
            best_loss, best_state = val_loss, copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return model


def main():
    parser = P.add_common_args(argparse.ArgumentParser(description=__doc__,
                                                       formatter_class=argparse.RawDescriptionHelpFormatter))
    parser.add_argument("--checkpoint", required=True, help="Backbone checkpoint file or run folder (seed_<seed>/).")
    parser.add_argument("--backbone", choices=sorted(P.BACKBONE_FILES),
                        help="Checkpoint to use from a run folder; overrides finetune.backbone.")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--modes", help="Comma-separated subset of mm,img,rna.")
    parser.add_argument("--unfreeze", action="store_true", help="Also update the backbone.")
    args = parser.parse_args()

    out_override = args.output_dir
    args.output_dir = None
    cfg = P.load_config(args.config, args)
    ft = cfg["finetune"]
    if args.seed is not None:
        ft["seed"] = args.seed
    if args.epochs is not None:
        ft["epochs"] = args.epochs
    if args.modes:
        ft["modes"] = [m.strip() for m in args.modes.split(",") if m.strip()]
    if args.unfreeze:
        ft["freeze_backbone"] = False
    checkpoint, backbone = P.resolve_checkpoint(args.checkpoint, args.backbone or ft.get("backbone", "best_auc"))
    print(f"Backbone ({backbone}): {checkpoint}")
    out_dir = Path(out_override) if out_override else checkpoint.parent / P.output_name("finetune", backbone)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = cfg["device"]

    P.set_seed(ft["seed"])
    datasets = P.build_datasets(cfg["data"])
    loaders = P.build_loaders(datasets, cfg["data"], ft["seed"])
    model_cfg = P.model_config_for_checkpoint(cfg, checkpoint)

    results = {"val": {}, "test": {}}
    for mode in ft["modes"]:
        model = finetune_mode(mode, checkpoint, model_cfg, ft, datasets, loaders, device)
        torch.save(model.state_dict(), out_dir / f"model_best_{mode}.pt")
        for split in ("val", "test"):
            preds, _ = P.predict_head(model, loaders[split], mode, device, ft["mm_paired_only"])
            results[split].update(P.save_predictions({mode: preds}, out_dir, split))
    (out_dir / "metrics.json").write_text(json.dumps(
        {"config": ft, "backbone": backbone, "checkpoint": str(checkpoint), **results}, indent=2))
    print(f"Test metrics (fine-tuned heads) -> {out_dir}")
    print(pd.DataFrame(results["test"]).T.to_string(float_format="{:.3f}".format))


if __name__ == "__main__":
    main()
