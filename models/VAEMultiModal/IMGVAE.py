"""Image autoencoder branch used by the multimodal ablation model."""

import torch
from torch import nn

from models.VAEMultiModal.Layers import Down, Up, DoubleConv


class ImageBranchVAEAblation(nn.Module):
    """Image encoder/decoder with class, multimodal, and style latents."""

    def __init__(
        self,
        in_channels_img: int = 3,
        out_channels_img: int = 3,
        img_base=(16, 32, 64, 128, 256),
        img_tail: int = 32,
        img_size=(256, 256),
        z_cls_dim: int = 0,
        z_mm_dim: int = 0,
        z_style_dim: int = 0,
        skip4_dropout_p: float = 0.3,
    ):
        super().__init__()
        self.z_cls_dim = int(z_cls_dim)
        self.z_mm_dim = int(z_mm_dim)
        self.z_style_dim = int(z_style_dim)
        self.z_dim = self.z_cls_dim + self.z_mm_dim + self.z_style_dim
        if self.z_dim <= 0:
            raise ValueError("ImageBranchVAEAblation: z_dim must be > 0.")
        self.img_size = img_size

        c1, c2, c3, c4, c5 = img_base
        self.img_inc = DoubleConv(in_channels_img, c1)
        self.img_down1 = Down(c1, c2)
        self.img_down2 = Down(c2, c3)
        self.img_down3 = Down(c3, c4)
        self.img_down4 = Down(c4, c5)
        self.img_enc_pool = nn.AdaptiveAvgPool2d(1)

        self.cls_mu_img = nn.Linear(c5, self.z_cls_dim) if self.z_cls_dim > 0 else None
        self.cls_logvar_img = nn.Linear(c5, self.z_cls_dim) if self.z_cls_dim > 0 else None
        self.cls_mu_mm_img = nn.Linear(c5, self.z_mm_dim) if self.z_mm_dim > 0 else None
        self.cls_logvar_mm_img = nn.Linear(c5, self.z_mm_dim) if self.z_mm_dim > 0 else None
        self.cls_style_mu_img = nn.Linear(c5, self.z_style_dim) if self.z_style_dim > 0 else None
        self.cls_style_logvar_img = nn.Linear(c5, self.z_style_dim) if self.z_style_dim > 0 else None

        self.img_up1 = Up(c5, c4)
        self.img_up2 = Up(c4, c3)
        self.img_up3 = Up(c3, c2)
        self.img_up4 = Up(c2, c1)
        self.img_tail = nn.Sequential(
            nn.Conv2d(c1, img_tail, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(img_tail), nn.ReLU(inplace=True),
        )
        self.img_outc = nn.Conv2d(img_tail, out_channels_img, kernel_size=1)

        with torch.no_grad():
            dummy = torch.zeros(1, in_channels_img, *self.img_size)
            x1 = self.img_inc(dummy)
            x2 = self.img_down1(x1)
            x3 = self.img_down2(x2)
            x4 = self.img_down3(x3)
            x5 = self.img_down4(x4)
            _, c5_enc, h5, w5 = x5.shape
            d4 = self.img_up1(x5)
            d3 = self.img_up2(d4)
            d2 = self.img_up3(d3)
            d1 = self.img_up4(d2)
            _, c4_dec, h4, w4 = d4.shape
            _, c3_dec, h3, w3 = d3.shape
            _, c2_dec, h2, w2 = d2.shape
            _, c1_dec, h1, w1 = d1.shape
            _, c4_enc, h4_enc, w4_enc = x4.shape

        self.enc_out_channels = c5_enc
        self.enc_out_hw = (h5, w5)
        self.dec_shapes = {
            "c5": (c5_enc, h5, w5),
            "c4_enc": (c4_enc, h4_enc, w4_enc),
            "c4_dec": (c4_dec, h4, w4),
            "c3": (c3_dec, h3, w3),
            "c2": (c2_dec, h2, w2),
            "c1": (c1_dec, h1, w1),
        }
        c5_ch, h5_s, w5_s = self.dec_shapes["c5"]
        self.z_to_c5 = nn.Linear(self.z_dim, c5_ch * h5_s * w5_s)
        self.img_z_to_c5 = nn.Conv2d(c5_ch, c5_ch, kernel_size=1)
        c4_enc_ch, h4_enc, w4_enc = self.dec_shapes["c4_enc"]
        c4_dec_ch, h4_s, w4_s = self.dec_shapes["c4_dec"]
        assert (h4_enc, w4_enc) == (h4_s, w4_s)
        self.skip4 = nn.Sequential(
            nn.Conv2d(c4_enc_ch, c4_dec_ch, kernel_size=1, bias=False),
            nn.InstanceNorm2d(c4_dec_ch), nn.ReLU(inplace=True),
        )
        self.skip4_dropout = nn.Dropout2d(p=skip4_dropout_p)
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
            elif isinstance(module, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.LayerNorm)):
                if module.weight is not None:
                    nn.init.constant_(module.weight, 1.0)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

    def encode(self, img_x: torch.Tensor):
        x1 = self.img_inc(img_x)
        x2 = self.img_down1(x1)
        x3 = self.img_down2(x2)
        x4 = self.img_down3(x3)
        x5 = self.img_down4(x4)
        batch_size, channels, height, width = x5.shape
        assert (height, width) == self.enc_out_hw, "Input size mismatch vs img_size"
        pooled = self.img_enc_pool(x5).view(batch_size, channels)
        return {
            "mu_cls_single": self.cls_mu_img(pooled) if self.cls_mu_img else None,
            "logvar_cls_single": self.cls_logvar_img(pooled) if self.cls_logvar_img else None,
            "mu_cls_mm": self.cls_mu_mm_img(pooled) if self.cls_mu_mm_img else None,
            "logvar_cls_mm": self.cls_logvar_mm_img(pooled) if self.cls_logvar_mm_img else None,
            "mu_style": self.cls_style_mu_img(pooled) if self.cls_style_mu_img else None,
            "logvar_style": self.cls_style_logvar_img(pooled) if self.cls_style_logvar_img else None,
            "x4": x4,
            "batch_size": batch_size,
        }

    def _cat_parts(self, z_cls_single=None, z_cls_mm=None, z_style=None):
        parts = []
        for dim, value, name in (
            (self.z_cls_dim, z_cls_single, "z_cls_single"),
            (self.z_mm_dim, z_cls_mm, "z_cls_mm"),
            (self.z_style_dim, z_style, "z_style"),
        ):
            if dim > 0:
                if value is None:
                    raise ValueError(f"{name} is required but missing.")
                parts.append(value)
        return torch.cat(parts, dim=1) if len(parts) > 1 else parts[0]

    def decode(self, z_cls_single=None, z_cls_mm=None, z_style=None, x4=None, batch_size=None):
        z = self._cat_parts(z_cls_single, z_cls_mm, z_style)
        assert z.shape[1] == self.z_dim
        c5, h5, w5 = self.dec_shapes["c5"]
        z5 = self.img_z_to_c5(self.z_to_c5(z).view(batch_size, c5, h5, w5))
        d4 = self.img_up1(z5)
        c4, h4, w4 = self.dec_shapes["c4_enc"]
        assert x4.shape[1:] == (c4, h4, w4)
        skip = self.skip4(x4)
        if self.training:
            skip = self.skip4_dropout(skip)
        d1 = self.img_up4(self.img_up3(self.img_up2(d4 + skip)))
        return torch.sigmoid(self.img_outc(self.img_tail(d1)))
