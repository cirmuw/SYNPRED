"""Shared data, model and evaluation code for the SYNPRED scripts and demo notebook."""

import copy
import json
import pickle
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from monai.transforms import Compose, RandGaussianNoise, RandGibbsNoise, ScaleIntensity
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from data.Dataset import BreastDCEDLMultimodal
from models.VAEMultiModal.synVAEMM import SynpredVAE

MODES = ("mm", "img", "rna")

IMAGE_AUGMENTATION = Compose([
    RandGibbsNoise(prob=1.0, alpha=(0.2, 0.9)),
    RandGaussianNoise(prob=1.0, mean=0.5, std=0.1),
    ScaleIntensity(minv=0.0, maxv=1.0),
])


# ---------------------------------------------------------------------------
# Config and reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def add_common_args(parser):
    parser.add_argument("--config", default="configs/synpred.yaml")
    parser.add_argument("--data-path", dest="data_path")
    parser.add_argument("--rna-data-path", dest="rna_data_path")
    parser.add_argument("--splits-path", dest="splits_path")
    parser.add_argument("--output-dir", dest="output_dir")
    parser.add_argument("--device")
    parser.add_argument("--batch-size", dest="batch_size", type=int)
    parser.add_argument("--drop-last-eval", dest="drop_last_eval", action="store_true",
                        help="Drop the last partial val/test batch, as in the paper runs.")
    return parser


def load_config(path, args=None):
    """Load the YAML config and apply common command-line overrides."""
    with open(path) as f:
        cfg = yaml.safe_load(f)
    if args is not None:
        for key, section, name in [
            ("data_path", "data", "data_path"),
            ("rna_data_path", "data", "rna_data_path"),
            ("splits_path", "data", "splits_path"),
            ("batch_size", "data", "batch_size"),
        ]:
            if getattr(args, key, None) is not None:
                cfg[section][name] = getattr(args, key)
        for key in ("output_dir", "device"):
            if getattr(args, key, None) is not None:
                cfg[key] = getattr(args, key)
        if getattr(args, "drop_last_eval", False):
            cfg["data"]["drop_last_eval"] = True
    if cfg["device"].startswith("cuda") and not torch.cuda.is_available():
        print("CUDA not available, using CPU.")
        cfg["device"] = "cpu"
    return cfg


def model_config_for_checkpoint(cfg, checkpoint):
    """Prefer the model settings saved next to a checkpoint (config.json)."""
    saved = Path(checkpoint).parent / "config.json"
    if saved.exists():
        model_cfg = json.loads(saved.read_text())["model"]
        if model_cfg != cfg["model"]:
            print(f"Using model settings from {saved}: {model_cfg}")
        return model_cfg
    return cfg["model"]


# Backbone checkpoints in a pretraining run folder. The released weights name the last
# epoch model.pt; scripts/pretrain.py writes model_last.pt.
BACKBONE_FILES = {"best_auc": ("model_best_auc.pt",), "last": ("model_last.pt", "model.pt")}


def resolve_checkpoint(path, backbone="best_auc"):
    """Return (checkpoint file, backbone) for a checkpoint file or a run folder.

    For a folder, `backbone` picks the checkpoint selected by validation AUROC
    ("best_auc") or the last epoch ("last"). For a file, it is inferred from the name.
    """
    path = Path(path)
    if path.is_file():
        return path, "last" if path.name in BACKBONE_FILES["last"] else "best_auc"
    for name in BACKBONE_FILES[backbone]:
        if (path / name).is_file():
            return path / name, backbone
    raise FileNotFoundError(f"No {' or '.join(BACKBONE_FILES[backbone])} in {path}")


def output_name(base, backbone):
    """frozen_eval / finetune for the selected checkpoint, *_last for the last epoch."""
    return base if backbone == "best_auc" else f"{base}_last"


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_rna_table(path) -> pd.DataFrame:
    """Return the RNA table as patients x genes.

    The GSE194040 file stores genes as rows and patient IDs in a header that is
    one field shorter than the data rows, so pandas uses gene symbols as the
    index. A single transpose gives patients x genes.
    """
    return pd.read_table(path).transpose()


def load_splits(data_path, splits_path=None):
    """Load train/val/test records from ``splits_path`` (default: ``data_path``).

    Image paths relative to ``data_path`` are made absolute; absolute paths are kept.
    """
    splits = {}
    for name in ("train", "val", "test"):
        with open(Path(splits_path or data_path) / f"{name}.pkl", "rb") as f:
            records = pickle.load(f)
        for record in records.values():
            for key in ("T0_early", "T0_late"):
                record[key] = str(Path(data_path) / record[key])
        splits[name] = records
    return splits


def split_summary(splits, rna_table) -> pd.DataFrame:
    return pd.DataFrame({
        name: {
            "patients": len(records),
            "pCR=1": sum(int(r["pcr"]) for r in records.values()),
            "with RNA": sum(pid in rna_table.index for pid in records),
        } for name, records in splits.items()
    }).T


def build_datasets(data_cfg, verbose=True):
    rna_table = load_rna_table(data_cfg["rna_data_path"])
    splits = load_splits(data_cfg["data_path"], data_cfg.get("splits_path"))
    summary = split_summary(splits, rna_table)
    if verbose:
        print(f"RNA table: {rna_table.shape[0]} patients x {rna_table.shape[1]} genes")
        print(summary.to_string())
    if summary["with RNA"].sum() == 0:
        raise ValueError(
            "No patient in the splits has RNA. Check that the pickle keys are strings "
            "matching the patient IDs in the RNA header."
        )
    datasets = {
        "train": BreastDCEDLMultimodal(splits["train"], rna_table, timepoints=["T0"],
                                       augmentation=IMAGE_AUGMENTATION, augment_rna=True),
        "val": BreastDCEDLMultimodal(splits["val"], rna_table, timepoints=["T0"]),
        "test": BreastDCEDLMultimodal(splits["test"], rna_table, timepoints=["T0"]),
    }
    # Same patients and tensors as "train", without augmentation (for embeddings).
    clean = copy.copy(datasets["train"])
    clean.augmentation, clean.augment_rna = None, False
    datasets["train_clean"] = clean
    return datasets


def build_loaders(datasets, data_cfg, seed):
    bs, drop = data_cfg["batch_size"], bool(data_cfg.get("drop_last_eval", False))
    return {
        "train": DataLoader(datasets["train"], batch_size=bs, shuffle=True, drop_last=True,
                            generator=torch.Generator().manual_seed(seed)),
        "train_clean": DataLoader(datasets["train_clean"], batch_size=bs, shuffle=False),
        "val": DataLoader(datasets["val"], batch_size=bs, shuffle=False, drop_last=drop),
        "test": DataLoader(datasets["test"], batch_size=bs, shuffle=False, drop_last=drop),
    }


def rna_dim(datasets) -> int:
    return datasets["train"].rna_data_mat.shape[1]


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def build_model(model_cfg, rna_dim, beta=1e-4, device="cpu"):
    return SynpredVAE(
        in_channels_img=2, out_channels_img=2, img_size=(256, 256),
        input_dim_rna=rna_dim, output_dim_rna=rna_dim,
        hidden_dim_rna=model_cfg["hidden_dim_rna"],
        skip4_dropout_p=model_cfg["skip4_dropout_p"],
        z_dim=model_cfg["z_dim"], z_cls_dim=model_cfg["z_cls_dim"],
        ablation_mode=model_cfg["latents"],
        beta_start=beta, beta_end=beta, total_epochs=1, warmup_start_epoch=0,
        trainable_priors=True, mm_kl_delay_epochs=0, p_drop_rna=0.0,
    ).to(device)


def load_state(module, path, device):
    state = torch.load(path, map_location=device)
    module.load_state_dict(state.get("state_dict", state))
    return module


def poe_diagonal_gaussians(mu_list, logvar_list, eps: float = 1e-8):
    """Product of experts of diagonal Gaussians."""
    precisions = [torch.exp(-lv) for lv in logvar_list]
    prec_sum = torch.stack(precisions).sum(0)
    mu = sum(p * m for p, m in zip(precisions, mu_list)) / (prec_sum + eps)
    return mu, -torch.log(prec_sum + eps)


def task_posterior(out, prefix):
    """Concatenate the task-relevant posteriors [fusion ; uni] of one modality."""
    mus, lvs = [], []
    if out.get("mm_mu_cls") is not None:
        mus.append(out["mm_mu_cls"]); lvs.append(out["mm_logvar_cls"])
    if out.get(f"mu_{prefix}_cls_single") is not None:
        mus.append(out[f"mu_{prefix}_cls_single"]); lvs.append(out[f"logvar_{prefix}_cls_single"])
    if not mus:
        return None, None
    return torch.cat(mus, 1), torch.cat(lvs, 1)


def embed(model, batch, mode, device, paired_only=True):
    """Prediction embedding (PoE mean) for one batch.

    Returns (embedding, row mask into the batch). RNA-only always uses patients
    with RNA; multimodal does too when ``paired_only``, otherwise patients
    without RNA fall back to the image expert.
    """
    img, rna = batch["T0"].to(device), batch["rna"].to(device)
    missing = batch["rna_missing"].to(device).bool()
    present = ~missing
    all_rows = torch.ones_like(present)

    if mode == "img":
        mu, lv = task_posterior(model(img_x=img, eval=True), "img")
        return poe_diagonal_gaussians([mu], [lv])[0], all_rows
    if not present.any() and (mode == "rna" or paired_only):
        return None, present
    if mode == "rna":
        mu, lv = task_posterior(model(rna_x=rna[present], eval=True), "rna")
        return poe_diagonal_gaussians([mu], [lv])[0], present
    if mode == "mm":
        out = model(img_x=img, rna_x=rna, rna_missing=missing, eval=True)
        mu_i, lv_i = task_posterior(out, "img")
        mu_r, lv_r = task_posterior(out, "rna")
        fused = poe_diagonal_gaussians([mu_i, mu_r], [lv_i, lv_r])[0]
        img_only = poe_diagonal_gaussians([mu_i], [lv_i])[0]
        z = torch.where(present[:, None], fused, img_only)
        return (z[present], present) if paired_only else (z, all_rows)
    raise ValueError(f"Unknown mode: {mode}")


# ---------------------------------------------------------------------------
# Frozen evaluation (linear probe)
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_embeddings(model, loader, device):
    """{mode: (X, y, ids)}; RNA and multimodal only for patients with RNA."""
    model.eval()
    acc = {m: ([], [], []) for m in MODES}
    for batch in loader:
        y, ids = batch["pcr"].numpy(), np.asarray(batch["id"])
        for mode in MODES:
            z, rows = embed(model, batch, mode, device, paired_only=True)
            if z is None:
                continue
            rows = rows.cpu().numpy()
            acc[mode][0].append(z.cpu().numpy()); acc[mode][1].append(y[rows]); acc[mode][2].append(ids[rows])
    return {m: tuple(np.concatenate(v) for v in acc[m]) for m in MODES if acc[m][0]}


def linear_probe(train_emb, eval_embs):
    """Fit logistic regression on multimodal train embeddings; predict every mode."""
    X, y, _ = train_emb["mm"]
    scaler = StandardScaler().fit(X)
    clf = LogisticRegression(max_iter=1000).fit(scaler.transform(X), y)
    return {
        m: pd.DataFrame({"id": ids, "y_true": yy.astype(int),
                         "prob": clf.predict_proba(scaler.transform(Xe))[:, 1]})
        for m, (Xe, yy, ids) in eval_embs.items()
    }


def metrics(pred: pd.DataFrame) -> dict:
    y, p = pred["y_true"].to_numpy(), pred["prob"].to_numpy()
    two_classes = len(np.unique(y)) > 1
    return {
        "n": int(len(y)),
        "auroc": float(roc_auc_score(y, p)) if two_classes else float("nan"),
        "prauc": float(average_precision_score(y, p)) if two_classes else float("nan"),
    }


def save_predictions(preds: dict, out_dir, split: str) -> dict:
    """Write predictions_{split}_{mode}.csv and return {mode: metrics}."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for mode, df in preds.items():
        df.to_csv(out_dir / f"predictions_{split}_{mode}.csv", index=False)
    return {mode: metrics(df) for mode, df in preds.items()}


def frozen_evaluation(model, loaders, device, out_dir):
    """Linear probe on frozen embeddings; writes predictions and metrics.json."""
    train_emb = extract_embeddings(model, loaders["train_clean"], device)
    results = {}
    for split in ("val", "test"):
        preds = linear_probe(train_emb, extract_embeddings(model, loaders[split], device))
        results[split] = save_predictions(preds, out_dir, split)
    (Path(out_dir) / "metrics.json").write_text(json.dumps(results, indent=2))
    return results


# ---------------------------------------------------------------------------
# Fine-tuning heads
# ---------------------------------------------------------------------------

class PredictionHeads(torch.nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 32):
        super().__init__()
        make = lambda: torch.nn.Sequential(
            torch.nn.Linear(in_dim, hidden_dim), torch.nn.ReLU(), torch.nn.Linear(hidden_dim, 1))
        self.prediction_head_mm = make()
        self.prediction_head_img = make()
        self.prediction_head_rna = make()


class FinetuneModel(torch.nn.Module):
    """Backbone plus one MLP head per evaluation mode (mm / img / rna)."""

    def __init__(self, base_model, in_dim: int, hidden_dim: int = 32):
        super().__init__()
        self.base_model = base_model
        self.heads = PredictionHeads(in_dim, hidden_dim)

    def logits(self, batch, mode, head, device, paired_only=True):
        z, rows = embed(self.base_model, batch, mode, device, paired_only)
        if z is None:
            return None, rows
        return getattr(self.heads, f"prediction_head_{head}")(z).view(-1), rows


@torch.no_grad()
def predict_head(model, loader, mode, device, paired_only=True):
    model.eval()
    frames, losses = [], []
    for batch in loader:
        logits, rows = model.logits(batch, mode, mode, device, paired_only)
        if logits is None:
            continue
        rows = rows.cpu().numpy()
        y = batch["pcr"].float().to(device)[torch.as_tensor(rows, device=device)]
        losses.append(F.binary_cross_entropy_with_logits(logits, y).item())
        frames.append(pd.DataFrame({"id": np.asarray(batch["id"])[rows],
                                    "y_true": y.cpu().numpy().astype(int),
                                    "prob": torch.sigmoid(logits).cpu().numpy()}))
    return pd.concat(frames, ignore_index=True), float(np.mean(losses))
