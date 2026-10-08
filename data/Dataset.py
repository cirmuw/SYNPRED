"""Dataset used by the BreastDCEDL ablation release.

The generic trainer does not depend on this module; it accepts any PyTorch
DataLoader through a user-provided batch adapter.
"""

from typing import Dict, List, Optional

import nibabel as nib
import numpy as np
import pandas as pd
import torch
from monai.transforms import Compose, Resize, ScaleIntensity
from torch.utils.data import Dataset
from tqdm import tqdm
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


class BreastDCEDLMultimodal(Dataset):
    """Load T0 early/late NIfTI images plus an RNA table.

    Each split record is ``{id: {pcr, T0_early, T0_late}}``. Missing RNA is
    represented by zeros and exposed as ``rna_missing``.
    """

    def __init__(
        self,
        data_list: Dict,
        rna_data_mat: pd.DataFrame,
        timepoints: Optional[List[str]] = None,
        augmentation=None,
        res: tuple = (256, 256),
        two_view_transform: bool = False,
        augment_rna: bool = False,
        three_channels: bool = False,
    ):
        self.timepoints = timepoints or ["T0"]
        self.augmentation = augmentation
        self.two_view_transform = two_view_transform
        self.augment_rna = augment_rna
        self.three_channels = three_channels
        self.rna_data_mat = self._preprocess_rna(rna_data_mat)
        self.preprocessing_transform = Compose([
            ScaleIntensity(minv=0.0, maxv=1.0),
            Resize(spatial_size=res),
        ])
        self.data_list = self._load_data(data_list)

    @staticmethod
    def _preprocess_rna(table: pd.DataFrame) -> pd.DataFrame:
        values = SimpleImputer(strategy="mean").fit_transform(table)
        values = np.log1p(values)
        values = StandardScaler().fit_transform(values)
        return pd.DataFrame(values, index=table.index, columns=table.columns)

    def _load_image(self, path: str) -> torch.Tensor:
        image = torch.as_tensor(nib.load(path).get_fdata(), dtype=torch.float32)
        return self.preprocessing_transform(image.unsqueeze(0))

    def _load_channels(self, record: Dict) -> torch.Tensor:
        early = self._load_image(record["T0_early"])
        late = self._load_image(record["T0_late"])
        channels = [early, late]
        if self.three_channels:
            channels.append(torch.zeros_like(early))
        return torch.stack(channels, dim=0).squeeze(1)

    def _load_data(self, records: Dict) -> list:
        loaded = []
        for patient_id in tqdm(records.keys(), desc="loading data"):
            record = records[patient_id]
            has_rna = patient_id in self.rna_data_mat.index
            rna_dim = self.rna_data_mat.shape[1]
            rna = (
                torch.tensor(self.rna_data_mat.loc[patient_id].values, dtype=torch.float32)
                if has_rna else torch.zeros(rna_dim, dtype=torch.float32)
            )
            item = {
                "id": patient_id,
                "pcr": record["pcr"],
                "rna": rna,
                "target_rna": rna.clone(),
                "rna_missing": torch.tensor(not has_rna, dtype=torch.bool),
            }
            for timepoint in self.timepoints:
                image = self._load_channels(record)
                item[timepoint] = image
                item[f"target_{timepoint}"] = image.clone()
            loaded.append(item)
        return loaded

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        item = self.data_list[index].copy()
        if self.augmentation is not None:
            for timepoint in self.timepoints:
                item[timepoint] = self.augmentation(item[timepoint])
        if self.augment_rna and not bool(item["rna_missing"]):
            item["rna"] = item["rna"] + 0.05 * torch.randn_like(item["rna"])
        return item

    def get_labels(self):
        return [item["pcr"] for item in self.data_list]
