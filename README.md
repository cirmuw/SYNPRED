# SYNPRED: A Synergistic Approach to Multimodal Learning for Clinical Prediction

Code for the multimodal VAE SYNPRED, applied to pathological complete response (pCR)
prediction from breast DCE-MRI and gene expression (BreastDCEDL / I-SPY2).

SYNPRED splits each modality's latent space into three parts:

- shared task-relevant latents (`fusion`, combined across modalities with a mixture of experts)
- modality-specific task-relevant latents (`uni`)
- instance-specific latents (`inst`)

With the paper settings (`z_dim=256`, `z_cls_dim=16`) these are 16 + 16 + 224 dimensions.
For prediction, the task-relevant posteriors are combined with a product of experts, so
the same classifier works with image-only, RNA-only or multimodal input.

Patient data is not redistributed here. Obtain it under the applicable data-use terms.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Data layout

`data_path` must contain the official BreastDCEDL train/validation/test split:

```text
BC_pcr_ds/
├── train.pkl
├── val.pkl
└── test.pkl
```

Each pickle maps a patient ID (string) to its pCR label and the two T0
maximum-intensity projections (early and late subtraction, 2D NIfTI):

```python
{
    "756412": {
        "pcr": 0,
        "T0_early": "/data/756412/T0_early_mip.nii.gz",
        "T0_late": "/data/756412/T0_late_mip.nii.gz",
    },
}
```

Images are scaled to [0, 1] and resized to 256x256. RNA comes from
`GSE194040_ISPY2ResID_AgilentGeneExp_990_FrshFrzn_meanCol_geneLevel_n988.txt`
(GEO GSE194040). That file has genes as rows and patient IDs as columns, and is
transposed once on loading. Missing values are mean-imputed, then `log1p` and
z-scoring are applied. Patients without RNA (e.g. the I-SPY1 and Duke cohorts) are used for
the image branch and marked `rna_missing=True`. Each script prints how many patients
per split have RNA. If that number is 0, the patient IDs do not match.

All settings are in [configs/synpred.yaml](configs/synpred.yaml). Set the two data
paths there, or pass `--data-path` and `--rna-data-path` to any script.

## Reproducing one run

```bash
# 1. Pretrain one seed (300 epochs). Selects the checkpoint by validation AUROC of a
#    multimodal linear probe, then runs the frozen evaluation on the test set.
python scripts/pretrain.py --config configs/synpred.yaml --seeds 42

# 2. Frozen evaluation (linear probe) of any checkpoint, e.g. released weights.
python scripts/evaluate.py --config configs/synpred.yaml \
    --checkpoint results/synpred/seed_42/model_best_auc.pt

# 3. Fine-tuning: MLP heads for multimodal, image-only and RNA-only prediction.
#    Given a run folder, the backbone is model_best_auc.pt (finetune.backbone: best_auc).
python scripts/finetune.py --config configs/synpred.yaml \
    --checkpoint results/synpred/seed_42
```

`--checkpoint` takes a checkpoint file or a run folder. With a folder, `--backbone last`
(or `finetune.backbone: last` in the config) uses the last-epoch checkpoint instead of
the one selected by validation AUROC; its outputs go to `frozen_eval_last/` and
`finetune_last/`. `evaluate.py` takes the same `--backbone` option.

Outputs go to `results/synpred/seed_42/`:

```text
model_best_auc.pt  model_last.pt  config.json  history.csv
frozen_eval/  metrics.json, predictions_{val,test}_{mm,img,rna}.csv
finetune/     metrics.json, predictions_*.csv, model_best_{mm,img,rna}.pt
```

For a quick check of the environment, add `--epochs 2 --device cpu`.

## All five seeds (Table 1)

```bash
python scripts/pretrain.py              # pretrain.seeds: [42, 43, 44, 45, 46]
for seed in 42 43 44 45 46; do
    python scripts/finetune.py --checkpoint results/synpred/seed_$seed
done
python scripts/summarize.py --root results/synpred
```

`summarize.py` reports the mean test AUROC and a bootstrap 95% CI over seeds for the
linear probe and the fine-tuned heads.

The latent ablations in Table 2 use the same scripts with `model.latents` set to
`style_cls`, `mm_cls`, `style_mm` or `none`, and a separate `--output-dir`.

## Demo notebook

[demo_breastdcedl.ipynb](demo_breastdcedl.ipynb) uses the same code. Given the data
paths and checkpoints, it shows samples from the dataloader, reconstructions, the
frozen evaluation of the selected and the last-epoch checkpoint, fine-tuning, test
images with true and predicted labels, ROC curves and the summary over seeds
(last-epoch checkpoints). It can also start pretraining.

## Using your own architecture and data

`generic/` is a task-agnostic training interface for any PyTorch encoder that
returns `(mu, logvar)` and any decoder that takes a latent tensor. See
[examples/generic_mlp.py](examples/generic_mlp.py).

```python
from generic import GenericTrainer, TrainerConfig, build_model

model = build_model(encoder=my_encoder, decoder=my_decoder, latent_dim=128,
                    class_latent_dim=16, style_latent_dim=112, num_classes=2)
trainer = GenericTrainer(model, TrainerConfig(epochs=100, beta=1e-4))
trainer.fit(train_loader, val_loader, output_dir="results/my_task")
```

The default batch adapter accepts `(x, label)`, `(x, target, label)` or a dict with
`x`/`target`/`label` (or `image`, `T0`, `pcr`). Pass `batch_adapter=...` for other
batch structures.

## Reproducibility notes

- Each pretraining seed in `pretrain.seeds` is an independent run that controls
  initialisation, augmentation and the data order; `--seeds 42` runs only one. Each
  fine-tuning run uses seed 42.
- The released weights came from the original training scripts, which set seed 42
  once and trained five repetitions back to back. The seed-42 run here corresponds to
  their first repetition; seeds 43-46 are new runs. Bit-exact agreement is not
  expected, since this code draws random numbers in a different order.
- The paper's runs dropped the last partial validation and test batch. Pass
  `--drop-last-eval` to evaluate on the same patients. By default every patient is used.
- The linear probe is fitted on non-augmented training embeddings.
- Exact numbers also depend on the PyTorch/CUDA version and GPU kernels.

## Repository layout

```text
configs/synpred.yaml        paper settings
scripts/pretrain.py         pretraining + frozen evaluation (one run per seed)
scripts/evaluate.py         linear-probe evaluation of a checkpoint
scripts/finetune.py         MLP heads on a pretrained backbone
scripts/summarize.py        mean and CI over seeds
utils/pipeline.py           shared data, model and evaluation code
data/Dataset.py             BreastDCEDL + RNA dataset
models/VAEMultiModal/       SYNPRED model (SynpredVAE) and image/RNA branches
generic/                    architecture-independent training interface
```
