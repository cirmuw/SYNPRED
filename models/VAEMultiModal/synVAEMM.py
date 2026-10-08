import torch
import torch.nn as nn
import torch.nn.functional as F
from models.VAEMultiModal.RNAVAE import RNADecoder, RNAEncoder, kl_divergence, reparameterize
from models.VAEMultiModal.Layers import Down, Up, DoubleConv
from models.VAEMultiModal.IMGVAE import ImageBranchVAEAblation


def decompose_kl(z, mu, log_var, **_kwargs):
    """Compatibility KL for legacy classes retained in this model file.

    The released ``MultiModalVAEDisMoEwProjectorsAblation`` path does not use
    this helper; it keeps the older classes importable without the removed
    beta-TCVAE utility module.
    """
    return kl_divergence(mu, log_var).mean()


# assumes DoubleConv, Down, Up exist in your codebase

class ImageBranchVAE(nn.Module):
    """
    Image-only branch for MultiModalVAEDisMoEwProjectors.
    Handles:
      - image encoder (UNet-style down path)
      - image decoder (UNet-style up path + skip)
      - splitting latent into [cls_single_img, cls_mm_img, style_img]
    """

    def __init__(
        self,
        in_channels_img: int = 3,
        out_channels_img: int = 3,
        img_base=(16, 32, 64, 128, 256),
        img_tail: int = 32,
        img_size=(256, 256),
        z_dim: int = 256,
        z_cls_dim: int = 16,
        skip4_dropout_p: float = 0.3,
    ):
        super().__init__()

        self.z_dim = int(z_dim)
        self.z_cls_dim = int(z_cls_dim)
        self.z_style_dim = self.z_dim - 2 * self.z_cls_dim
        if self.z_style_dim <= 0:
            raise ValueError(
                f"ImageBranchVAE: z_dim={z_dim} too small for two cls tokens of size {z_cls_dim}; "
                f"need z_dim > 2*z_cls_dim."
            )

        self.img_size = img_size

        # ---------------- IMAGE ENCODER ----------------
        c1, c2, c3, c4, c5 = img_base

        self.img_inc   = DoubleConv(in_channels_img, c1)
        self.img_down1 = Down(c1, c2)
        self.img_down2 = Down(c2, c3)
        self.img_down3 = Down(c3, c4)
        self.img_down4 = Down(c4, c5)

        self.img_enc_pool = nn.AdaptiveAvgPool2d(1)

        # heads for cls_single, cls_mm, style
        self.cls_mu_img         = nn.Linear(c5, self.z_cls_dim)
        self.cls_logvar_img     = nn.Linear(c5, self.z_cls_dim)
        self.cls_mu_mm_img      = nn.Linear(c5, self.z_cls_dim)
        self.cls_logvar_mm_img  = nn.Linear(c5, self.z_cls_dim)
        self.cls_style_mu_img   = nn.Linear(c5, self.z_style_dim)
        self.cls_style_logvar_img = nn.Linear(c5, self.z_style_dim)

        # ---------------- DECODER ----------------
        self.img_up1 = Up(c5, c4)
        self.img_up2 = Up(c4, c3)
        self.img_up3 = Up(c3, c2)
        self.img_up4 = Up(c2, c1)

        self.img_tail = nn.Sequential(
            nn.Conv2d(c1, img_tail, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(img_tail),
            nn.ReLU(inplace=True),
        )
        self.img_outc = nn.Conv2d(img_tail, out_channels_img, kernel_size=1)

        # --------- infer encoder/decoder spatial sizes and build z/skip maps ---------
        with torch.no_grad():
            H, W = self.img_size
            dummy = torch.zeros(1, in_channels_img, H, W)
            x1 = self.img_inc(dummy)
            x2 = self.img_down1(x1)
            x3 = self.img_down2(x2)
            x4 = self.img_down3(x3)
            x5 = self.img_down4(x4)  # [1, c5_enc, H5, W5]
            _, c5_enc, H5, W5 = x5.shape

            # run dummy through decoder to get shapes at each scale
            d5 = x5
            d4 = self.img_up1(d5)  # [1, c4_dec, H4, W4]
            d3 = self.img_up2(d4)  # [1, c3_dec, H3, W3]
            d2 = self.img_up3(d3)  # [1, c2_dec, H2, W2]
            d1 = self.img_up4(d2)  # [1, c1_dec, H1, W1]

            _, c4_dec, H4, W4 = d4.shape
            _, c3_dec, H3, W3 = d3.shape
            _, c2_dec, H2, W2 = d2.shape
            _, c1_dec, H1, W1 = d1.shape
            _, c4_enc, H4_enc, W4_enc = x4.shape

        # store shapes
        self.enc_out_channels = c5_enc
        self.enc_out_hw = (H5, W5)
        self.dec_shapes = {
            'c5': (c5_enc, H5, W5),
            'c4_enc': (c4_enc, H4_enc, W4_enc),
            'c4_dec': (c4_dec, H4, W4),
            'c3': (c3_dec, H3, W3),
            'c2': (c2_dec, H2, W2),
            'c1': (c1_dec, H1, W1),
        }

        # Latent -> bottleneck spatial feature map
        c5_ch, H5_s, W5_s = self.dec_shapes['c5']
        self.z_to_c5 = nn.Linear(self.z_dim, c5_ch * H5_s * W5_s)
        self.img_z_to_c5 = nn.Conv2d(c5_ch, c5_ch, kernel_size=1)

        # Single encoder skip from x4 (H/8)
        c4_enc_ch, H4_enc, W4_enc = self.dec_shapes['c4_enc']
        c4_dec_ch, H4_s, W4_s = self.dec_shapes['c4_dec']
        assert H4_enc == H4_s and W4_enc == W4_s, "x4 and decoder d4 spatial sizes must match"

        self.skip4 = nn.Sequential(
            nn.Conv2d(c4_enc_ch, c4_dec_ch, kernel_size=1, bias=False),
            nn.InstanceNorm2d(c4_dec_ch),
            nn.ReLU(inplace=True),
        )
        self.skip4_dropout = nn.Dropout2d(p=skip4_dropout_p)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.LayerNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1.0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    # ---------- ENCODE IMAGE ----------
    def encode(self, img_x: torch.Tensor):
        """
        img_x: [B, C, H, W]
        returns dict with:
          mu_cls_single, logvar_cls_single,
          mu_cls_mm, logvar_cls_mm,
          mu_style, logvar_style,
          x4, batch_size
        """
        x1 = self.img_inc(img_x)
        x2 = self.img_down1(x1)
        x3 = self.img_down2(x2)
        x4 = self.img_down3(x3)
        x5 = self.img_down4(x4)     # (B, c5, H5, W5)
        B, C5, H5, W5 = x5.shape

        H5_enc, W5_enc = self.enc_out_hw
        assert H5 == H5_enc and W5 == W5_enc, "Input size mismatch vs img_size used in __init__"

        pooled_img = self.img_enc_pool(x5).view(B, C5)

        mu_img_cls_single    = self.cls_mu_img(pooled_img)
        logvar_img_cls_single = self.cls_logvar_img(pooled_img)
        mu_img_cls_mm        = self.cls_mu_mm_img(pooled_img)
        logvar_img_cls_mm    = self.cls_logvar_mm_img(pooled_img)
        mu_img_style         = self.cls_style_mu_img(pooled_img)
        logvar_img_style     = self.cls_style_logvar_img(pooled_img)

        return {
            "mu_cls_single": mu_img_cls_single,
            "logvar_cls_single": logvar_img_cls_single,
            "mu_cls_mm": mu_img_cls_mm,
            "logvar_cls_mm": logvar_img_cls_mm,
            "mu_style": mu_img_style,
            "logvar_style": logvar_img_style,
            "x4": x4,
            "batch_size": B,
        }

    # ---------- DECODE IMAGE ----------
    def decode(
        self,
        z_cls_single: torch.Tensor,
        z_cls_mm: torch.Tensor,
        z_style: torch.Tensor,
        x4: torch.Tensor,
        batch_size: int,
    ):
        """
        Decode image from [cls_single_img, cls_mm, style_img] + encoder feature x4.
        Returns: img_rec in [0,1], shape [B, C, H, W]
        """
        z_img_full = torch.cat([z_cls_single, z_cls_mm, z_style], dim=1)
        assert z_img_full.shape[1] == self.z_dim

        c5_ch, H5_s, W5_s = self.dec_shapes['c5']
        z5 = self.z_to_c5(z_img_full).view(batch_size, c5_ch, H5_s, W5_s)
        z5 = self.img_z_to_c5(z5)       # [B, C5, H5, W5]

        d4 = self.img_up1(z5)           # [B, c4_dec, H4, W4]

        c4_enc_ch, H4_enc, W4_enc = self.dec_shapes['c4_enc']
        c4_dec_ch, H4_s, W4_s = self.dec_shapes['c4_dec']
        assert d4.shape[2:] == (H4_s, W4_s)
        assert x4.shape[2:] == (H4_enc, W4_enc)

        skip4 = self.skip4(x4)          # [B, c4_dec, H4, W4]
        if self.training:
            skip4 = self.skip4_dropout(skip4)
        d4 = d4 + skip4

        d3 = self.img_up2(d4)
        d2 = self.img_up3(d3)
        d1 = self.img_up4(d2)

        x_dec = self.img_tail(d1)
        logits_img = self.img_outc(x_dec)
        img_rec = torch.sigmoid(logits_img)

        return img_rec



# assumes RNAEncoder, RNADecoder exist in your codebase

class RNABranchVAE(nn.Module):
    """
    RNA-only branch for MultiModalVAEDisMoEwProjectors.
    Handles:
      - MLP-based encoder
      - MLP-based decoder
      - splitting latent into [cls_single_rna, cls_mm_rna, style_rna]
    """

    def __init__(
        self,
        input_dim_rna: int = 1000,
        hidden_dim_rna: int = 256,
        output_dim_rna: int = 1000,
        z_dim: int = 256,
        z_cls_dim: int = 16,
    ):
        super().__init__()

        self.z_dim = int(z_dim)
        self.z_cls_dim = int(z_cls_dim)
        self.z_style_dim = self.z_dim - 2 * self.z_cls_dim
        if self.z_style_dim <= 0:
            raise ValueError(
                f"RNABranchVAE: z_dim={z_dim} too small for two cls tokens of size {z_cls_dim}; "
                f"need z_dim > 2*z_cls_dim."
            )

        # encoder / decoder
        self.rna_encoder = RNAEncoder(input_dim_rna, hidden_dim_rna, self.z_dim, output='latent')
        self.rna_decoder = RNADecoder(self.z_dim, hidden_dim_rna, output_dim_rna)

        # heads for cls_single, cls_mm, style
        self.cls_mu_rna         = nn.Linear(hidden_dim_rna, self.z_cls_dim)
        self.cls_logvar_rna     = nn.Linear(hidden_dim_rna, self.z_cls_dim)
        self.cls_mu_mm_rna      = nn.Linear(hidden_dim_rna, self.z_cls_dim)
        self.cls_logvar_mm_rna  = nn.Linear(hidden_dim_rna, self.z_cls_dim)
        self.cls_style_mu_rna   = nn.Linear(hidden_dim_rna, self.z_style_dim)
        self.cls_style_logvar_rna = nn.Linear(hidden_dim_rna, self.z_style_dim)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    # ---------- ENCODE RNA ----------
    def encode(self, rna_x: torch.Tensor):
        """
        rna_x: [B, input_dim_rna]
        returns dict with:
          mu_cls_single, logvar_cls_single,
          mu_cls_mm, logvar_cls_mm,
          mu_style, logvar_style,
          h_rna
        """
        h_rna = self.rna_encoder(rna_x)  # [B, hidden_dim_rna]

        mu_rna_cls_single    = self.cls_mu_rna(h_rna)
        logvar_rna_cls_single = self.cls_logvar_rna(h_rna)
        mu_rna_cls_mm        = self.cls_mu_mm_rna(h_rna)
        logvar_rna_cls_mm    = self.cls_logvar_mm_rna(h_rna)
        mu_rna_style         = self.cls_style_mu_rna(h_rna)
        logvar_rna_style     = self.cls_style_logvar_rna(h_rna)

        return {
            "mu_cls_single": mu_rna_cls_single,
            "logvar_cls_single": logvar_rna_cls_single,
            "mu_cls_mm": mu_rna_cls_mm,
            "logvar_cls_mm": logvar_rna_cls_mm,
            "mu_style": mu_rna_style,
            "logvar_style": logvar_rna_style,
            "h_rna": h_rna,
        }

    # ---------- DECODE RNA ----------
    def decode(
        self,
        z_cls_single: torch.Tensor,
        z_cls_mm: torch.Tensor,
        z_style: torch.Tensor,
    ):
        """
        Decode RNA from [cls_single_rna, cls_mm, style_rna].
        Returns: rna_rec: [B, output_dim_rna]
        """
        z_rna_full = torch.cat([z_cls_single, z_cls_mm, z_style], dim=1)
        assert z_rna_full.shape[1] == self.z_dim

        rna_rec = self.rna_decoder(z_rna_full)
        return rna_rec



# assumes:
#   - DoubleConv, Down, Up
#   - RNAEncoder, RNADecoder
#   - reparameterize, kl_divergence, decompose_kl
#   - ImageBranchVAE, RNABranchVAE
# are defined/imported elsewhere


class MultiModalVAEDisMoEwProjectors(nn.Module):
    """
    Multimodal disentangled VAE with MoE fusion of class latents.

    Latent structure per modality:
      z_img = [z_cls_img, z_cls_mm_img, z_style_img]
      z_rna = [z_cls_rna, z_cls_mm_rna, z_style_rna]

    - z_cls_img / z_cls_rna: used for unimodal class KL (per-modality).
    - z_cls_mm_*: fused via MoE across modalities; fused z_cls_mm_fused gets its own KL.
    - Decoders use both tokens:
        img: [z_cls_img, z_cls_mm_fused, z_style_img]
        rna: [z_cls_rna, z_cls_mm_fused, z_style_rna]

    Outputs:
      - still only one 'z_cls' key:
          z_cls = concat([z_cls_img, z_cls_mm_fused]) when image is present
                (analogously for RNA-only eval).
    """

    def __init__(
        self,
        # image branch
        in_channels_img=3,
        out_channels_img=3,
        img_base=(16, 32, 64, 128, 256),
        img_tail=32,
        img_size=(256, 256),      # (H, W) – must match your input images
        skip4_dropout_p: float = 0.3,

        # RNA branch
        input_dim_rna=1000,
        hidden_dim_rna=256,
        output_dim_rna=1000,

        # shared latent
        z_dim=256,                 # total latent dim = 2*z_cls_dim + z_style_dim
        *,
        z_cls_dim: int = 16,       # dim of each cls token
        num_classes: int = 2,
        beta_start: float = 0.0,
        beta_end: float = 1.0,
        total_epochs: int = 100,
        warmup_start_epoch: int = 0,
        trainable_priors: bool = True,
        prior_means: torch.Tensor = None,      # [num_classes, z_cls_dim]
        prior_logvars: torch.Tensor = None,    # [num_classes, z_cls_dim]
        include_prior_in_poe: bool = False,    # kept for API compatibility, not used in MoE
        tc_weight_img: float = 6.0,
        tc_weight_rna: float = 6.0,
        mm_kl_delay_epochs: int = 100,         # delay after warmup before enabling mm KL
        p_drop_rna:float = 0.3,
    ):
        super().__init__()

        # ----- latent sizes -----
        assert 1 <= z_cls_dim, "z_cls_dim must be >= 1"
        self.z_dim = int(z_dim)
        self.z_cls_single_dim = int(z_cls_dim)   # for unimodal KL
        self.z_cls_mm_dim = int(z_cls_dim)       # for multimodal MoE
        self.z_style_dim = self.z_dim - self.z_cls_single_dim - self.z_cls_mm_dim
        if self.z_style_dim <= 0:
            raise ValueError(
                f"z_dim={z_dim} too small for two cls tokens of size {z_cls_dim}; "
                f"need z_dim > 2*z_cls_dim."
            )

        # keep original API attribute (interpreted as per-token dim)
        self.z_cls_dim = int(z_cls_dim)
        self.num_classes = int(num_classes)
        self.include_prior_in_poe = bool(include_prior_in_poe)  # unused, but kept for compatibility
        self.dataset_size = 508
        self.tc_weight_img = float(tc_weight_img)
        self.tc_weight_rna = float(tc_weight_rna)
        self.mm_kl_delay_epochs = int(mm_kl_delay_epochs)
        self.img_size = img_size
        self.p_drop_rna = p_drop_rna

        # ---------------- MODALITY-SPECIFIC BRANCHES ----------------
        self.img_branch = ImageBranchVAE(
            in_channels_img=in_channels_img,
            out_channels_img=out_channels_img,
            img_base=img_base,
            img_tail=img_tail,
            img_size=img_size,
            z_dim=self.z_dim,
            z_cls_dim=self.z_cls_single_dim,
            skip4_dropout_p=skip4_dropout_p,
        )

        self.rna_branch = RNABranchVAE(
            input_dim_rna=input_dim_rna,
            hidden_dim_rna=hidden_dim_rna,
            output_dim_rna=output_dim_rna,
            z_dim=self.z_dim,
            z_cls_dim=self.z_cls_single_dim,
        )


        # ---------------- CLASS PRIOR (for z_cls tokens) ----------------
        if prior_means is None:
            if self.num_classes == 2:
                prior_means = torch.stack([
                    torch.zeros(self.z_cls_dim),
                    torch.ones(self.z_cls_dim) * 2.0
                ])
            else:
                means = []
                for k in range(self.num_classes):
                    means.append(torch.ones(self.z_cls_dim) * (2.0 * k))
                prior_means = torch.stack(means, dim=0)
        else:
            assert prior_means.shape == (self.num_classes, self.z_cls_dim)

        if prior_logvars is None:
            prior_logvars = torch.zeros(self.num_classes, self.z_cls_dim)
        else:
            assert prior_logvars.shape == (self.num_classes, self.z_cls_dim)
        eps = 1e-2
        self.prior_means_img = nn.Parameter(
            prior_means.clone() + eps*torch.randn_like(prior_means),
            requires_grad=trainable_priors
        )
        self.prior_logvars_img = nn.Parameter(
            prior_logvars.clone() + eps*torch.randn_like(prior_logvars),
            requires_grad=trainable_priors
        )


        self.prior_means_rna = nn.Parameter(
            prior_means.clone() + eps*torch.randn_like(prior_means),
            requires_grad=trainable_priors
        )
        self.prior_logvars_rna = nn.Parameter(
            prior_logvars + eps*torch.randn_like(prior_logvars),
            requires_grad=trainable_priors
        )


        self.prior_means_mm = nn.Parameter(
            prior_means.clone() + eps*torch.randn_like(prior_means),
            requires_grad=trainable_priors
        )
        self.prior_logvars_mm = nn.Parameter(
            prior_logvars + eps*torch.randn_like(prior_logvars),
            requires_grad=trainable_priors
        )


        # ---------------- MoE fusion weights (learnable) ----------------
        self.moe_logits = nn.Parameter(torch.zeros(2))

        # ---------------- Beta schedule ----------------
        self.beta_start = float(beta_start)
        self.beta_end = float(beta_end)
        self.total_epochs = int(total_epochs)
        self.warmup_start_epoch = int(warmup_start_epoch)
        self.register_buffer("beta", torch.tensor(0.0, dtype=torch.float32))

        self._init_weights()

    def _init_weights(self):
        """
        Initialize weights of the network.

        Conv / ConvTranspose: Kaiming normal
        Linear: Xavier normal
        Norms: weight=1, bias=0
        Biases: 0
        """
        for m in self.modules():
            # Convolution layers
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

            # Linear layers
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

            # Normalization layers
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.LayerNorm)):
                if getattr(m, "weight", None) is not None:
                    nn.init.constant_(m.weight, 1.0)
                if getattr(m, "bias", None) is not None:
                    nn.init.constant_(m.bias, 0.0)

    # --------- beta scheduling ----------
    def _beta_at(self, current_epoch: int) -> float:
        if current_epoch < self.warmup_start_epoch:
            return 0.0
        warmup_epochs = self.total_epochs - self.warmup_start_epoch
        if warmup_epochs <= 1:
            progress = 1.0
        else:
            e = max(0, min(current_epoch - self.warmup_start_epoch, warmup_epochs - 1))
            progress = e / (warmup_epochs - 1)
        return self.beta_start + progress * (self.beta_end - self.beta_start)

    def _update_beta(self, current_epoch: int):
        new_beta = self._beta_at(current_epoch)
        self.beta.copy_(torch.tensor(new_beta, dtype=self.beta.dtype, device=self.beta.device))

    # --------- MoE fusion for multimodal class latents ----------
    def fuse_cls_moe(
        self,
        mu_img_cls_mm: torch.Tensor = None,
        logvar_img_cls_mm: torch.Tensor = None,
        mu_rna_cls_mm: torch.Tensor = None,
        logvar_rna_cls_mm: torch.Tensor = None,
    ):
        """Learnable MoE fusion of multimodal class Gaussians."""
        experts_mu = []
        experts_logvar = []
        expert_ids = []  # 0 for img, 1 for rna

        if mu_img_cls_mm is not None and logvar_img_cls_mm is not None:
            experts_mu.append(mu_img_cls_mm)
            experts_logvar.append(logvar_img_cls_mm)
            expert_ids.append(0)

        if mu_rna_cls_mm is not None and logvar_rna_cls_mm is not None:
            experts_mu.append(mu_rna_cls_mm)
            experts_logvar.append(logvar_rna_cls_mm)
            expert_ids.append(1)

        num_experts = len(experts_mu)
        if num_experts == 0:
            raise ValueError("fuse_cls_moe: at least one expert (img or rna) must be provided.")

        if num_experts == 1:
            return experts_mu[0], experts_logvar[0]

        w = F.softmax(self.moe_logits, dim=0)  # [2]
        w_img, w_rna = w[0], w[1]

        weights = []
        for eid in expert_ids:
            weights.append(w_img if eid == 0 else w_rna)

        vars_ = [torch.exp(lv) for lv in experts_logvar]  # list of [B, D]
        var_moe = sum(w * v for w, v in zip(weights, vars_))
        mu_moe  = sum(w * m for w, m in zip(weights, experts_mu))
        logvar_moe = torch.log(var_moe + 1e-8)
        return mu_moe, logvar_moe


    def fuse_cls_poe(
        self,
        mu_img_cls_mm=None, logvar_img_cls_mm=None,
        mu_rna_cls_mm=None, logvar_rna_cls_mm=None,
        labels: torch.Tensor = None,
    ):
        mus, logvars = [], []

        if mu_img_cls_mm is not None and logvar_img_cls_mm is not None:
            mus.append(mu_img_cls_mm); logvars.append(logvar_img_cls_mm)
        if mu_rna_cls_mm is not None and logvar_rna_cls_mm is not None:
            mus.append(mu_rna_cls_mm); logvars.append(logvar_rna_cls_mm)

        if len(mus) == 0:
            raise ValueError("fuse_cls_poe: need at least one modality.")

        # include prior expert if requested
        if self.include_prior_in_poe:
            if labels is None:
                raise ValueError("include_prior_in_poe=True requires labels for class-conditional prior.")
            labels = labels.long()
            prior_mu = self.prior_means[labels]        # [B,D]
            prior_logvar = self.prior_logvars[labels]  # [B,D]
            mus.append(prior_mu)
            logvars.append(prior_logvar)

        if len(mus) == 1:
            return mus[0], logvars[0]

        mu = torch.stack(mus, dim=0)                  # [E,B,D]
        logvar = torch.stack(logvars, dim=0)          # [E,B,D]

        var = torch.exp(logvar).clamp_min(1e-8)
        precision = 1.0 / var

        precision_sum = precision.sum(dim=0)
        mu_fused = (precision * mu).sum(dim=0) / precision_sum
        var_fused = 1.0 / precision_sum
        logvar_fused = torch.log(var_fused.clamp_min(1e-8))
        return mu_fused, logvar_fused

    # ------------- losses ---------------
    def compute_loss(
        self,
        img_rec,
        img_target,
        rna_rec,
        rna_target,
        mm_mu_cls_mm,         # fused multimodal class mean (mm token)
        mm_logvar_cls_mm,
        mu_img_style,
        logvar_img_style,
        mu_rna_style,
        logvar_rna_style,
        labels,
        current_epoch: int,
        mu_img_cls_single=None,
        logvar_img_cls_single=None,
        mu_rna_cls_single=None,
        logvar_rna_cls_single=None,
        mu_img_cls_mm=None,
        logvar_img_cls_mm=None,
        mu_rna_cls_mm=None,
        logvar_rna_cls_mm=None,
        drop_rna=False,
        warmup=False
    ):
        # reconstruction
        loss_rec_img = F.l1_loss(img_rec, img_target, reduction='mean')
        if not drop_rna:
            loss_rec_rna = F.l1_loss(rna_rec, rna_target, reduction='mean')
        else:
            loss_rec_rna = torch.zeros_like(loss_rec_img)
        loss_rec = 0.5 * (loss_rec_img + loss_rec_rna)

        labels = labels.long()
        prior_mu_img = self.prior_means_img[labels]        # [B, z_cls_dim]
        prior_logvar_img = self.prior_logvars_img[labels]  # [B, z_cls_dim]
        if drop_rna:
            prior_mu_rna = torch.zeros_like(prior_mu_img)
            prior_logvar_rna = torch.zeros_like(prior_logvar_img)
        else:
            prior_mu_rna = self.prior_means_rna[labels]        # [B, z_cls_dim]
            prior_logvar_rna = self.prior_logvars_rna[labels]  # [B, z_cls_dim]

        prior_mu_mm = self.prior_means_mm[labels]        # [B, z_cls_dim]
        prior_logvar_mm = self.prior_logvars_mm[labels]  # [B, z_cls_dim]

        # ----- CLASS KLs -----
        kl_terms = []

        # unimodal class KLs (cls_single tokens) active after warmup when beta>0
        if mu_img_cls_single is not None and logvar_img_cls_single is not None:
            kl_terms.append(kl_divergence(mu_img_cls_single, logvar_img_cls_single,
                                          prior_mu_img, prior_logvar_img).mean())
        if mu_rna_cls_single is not None and logvar_rna_cls_single is not None and not drop_rna:
            kl_terms.append(kl_divergence(mu_rna_cls_single, logvar_rna_cls_single,
                                          prior_mu_rna, prior_logvar_rna).mean())

        # multimodal class KL on fused mm token, delayed
        mm_kl_start = self.warmup_start_epoch + self.mm_kl_delay_epochs
        if current_epoch >= mm_kl_start:
            kl_terms.append(kl_divergence(mm_mu_cls_mm, mm_logvar_cls_mm,
                                          prior_mu_mm, prior_logvar_mm).mean())

        # ALIGMENT OF single modal part to multimodal part    
        kl_terms.append(kl_divergence(mu_img_cls_mm, logvar_img_cls_mm,
                            mm_mu_cls_mm.detach(), mm_logvar_cls_mm.detach()).mean())
        if not drop_rna:
            kl_terms.append(kl_divergence(mu_rna_cls_mm, logvar_rna_cls_mm,
                                        mm_mu_cls_mm.detach(), mm_logvar_cls_mm.detach()).mean())

        

        kl_cls = sum(kl_terms)

        # ----- Style KLs vs N(0, I) -----
        kl_style_img = decompose_kl(
            z=reparameterize(mu_img_style, logvar_img_style),
            mu=mu_img_style,
            log_var=logvar_img_style,
            dataset_size=self.dataset_size,
            use_mss=True,
            beta=self.tc_weight_img,
        )
        if drop_rna:
            kl_style_rna = torch.zeros_like(kl_style_img)
        else:
            kl_style_rna = decompose_kl(
                z=reparameterize(mu_rna_style, logvar_rna_style),
                mu=mu_rna_style,
                log_var=logvar_rna_style,
                dataset_size=self.dataset_size,
                use_mss=True,
                beta=self.tc_weight_rna,
            )
        
        kl_style = 0.5 * (kl_style_img + kl_style_rna)
        loss_kl = kl_cls + kl_style
        if warmup == False:
            loss = loss_rec + self.beta * kl_style + self.beta * 10 * kl_cls
        else:
            loss = loss_rec

        return (loss,
                loss_rec, loss_rec_img, loss_rec_rna,
                loss_kl, kl_cls, kl_style, kl_style_img, kl_style_rna)

    # ------------- forward --------------
    def forward(
        self,
        img_x=None,
        img_target=None,
        rna_x=None,
        rna_target=None,
        labels=None,
        current_epoch: int = 0,
        eval: bool = False,
        warmup: bool = False,
    ):
        """
        TRAINING (eval=False):
          - Warmup (beta=0): only recon
          - After warmup: unimodal KLs on cls_single + style
          - After warmup + mm_kl_delay_epochs: add multimodal KL on fused cls_mm

        Image decoding:
          - z_img_full = [z_cls_img, z_cls_mm_fused, z_style_img]
          - passed through image branch decoder

        RNA decoding:
          - z_rna_full = [z_cls_rna, z_cls_mm_fused, z_style_rna]
          - passed through RNA branch decoder
        """
        self._update_beta(current_epoch)

        # ---------------- TRAINING MODE ----------------
        if not eval:
            
            drop_rna = (torch.rand((), device=img_x.device) < self.p_drop_rna) if not warmup or rna_x is None else False
            
            # expecting img_x/img_target either [B,H,W] or [B,C,H,W]; original code unsqueezed
            if img_x is not None and img_x.dim() == 3:
                img_x = img_x.unsqueeze(1)
            if img_target is not None and img_target.dim() == 3:
                img_target = img_target.unsqueeze(1)

            # ===== IMAGE ENCODER =====
            img_enc = self.img_branch.encode(img_x)
            mu_img_cls_single = img_enc["mu_cls_single"]
            logvar_img_cls_single = img_enc["logvar_cls_single"]
            mu_img_cls_mm = img_enc["mu_cls_mm"]
            logvar_img_cls_mm = img_enc["logvar_cls_mm"]
            mu_img_style = img_enc["mu_style"]
            logvar_img_style = img_enc["logvar_style"]
            x4 = img_enc["x4"]
            N = img_enc["batch_size"]

            if not drop_rna:
                # ===== RNA ENCODER =====
                rna_enc = self.rna_branch.encode(rna_x)
                mu_rna_cls_single = rna_enc["mu_cls_single"]
                logvar_rna_cls_single = rna_enc["logvar_cls_single"]
                mu_rna_cls_mm = rna_enc["mu_cls_mm"]
                logvar_rna_cls_mm = rna_enc["logvar_cls_mm"]
                mu_rna_style = rna_enc["mu_style"]
                logvar_rna_style = rna_enc["logvar_style"]

                # ===== MoE FUSION (for cls_mm tokens) =====
                mm_mu_cls_mm, mm_logvar_cls_mm = self.fuse_cls_moe(
                    mu_img_cls_mm=mu_img_cls_mm,
                    logvar_img_cls_mm=logvar_img_cls_mm,
                    mu_rna_cls_mm=mu_rna_cls_mm,
                    logvar_rna_cls_mm=logvar_rna_cls_mm,
                )

            else: 
                mm_mu_cls_mm, mm_logvar_cls_mm = self.fuse_cls_moe(
                    mu_img_cls_mm=mu_img_cls_mm,
                    logvar_img_cls_mm=logvar_img_cls_mm,
                    mu_rna_cls_mm=None,
                    logvar_rna_cls_mm=None,
                )

            # ===== SAMPLE =====
            # unimodal cls tokens
            z_cls_img_single = reparameterize(mu_img_cls_single, logvar_img_cls_single)
            

            # multimodal fused cls token
            z_cls_mm = reparameterize(mm_mu_cls_mm, mm_logvar_cls_mm)
            z_img_style = reparameterize(mu_img_style, logvar_img_style)
            # ===== DECODE IMAGE =====
            img_rec = self.img_branch.decode(
                z_cls_single=z_cls_img_single,
                z_cls_mm=z_cls_mm,
                z_style=z_img_style,
                x4=x4,
                batch_size=N,
            )

            if not drop_rna:
            # styles
                z_cls_rna_single = reparameterize(mu_rna_cls_single, logvar_rna_cls_single)
                z_rna_style = reparameterize(mu_rna_style, logvar_rna_style)
                # ===== DECODE RNA =====
                rna_rec = self.rna_branch.decode(
                    z_cls_single=z_cls_rna_single,
                    z_cls_mm=z_cls_mm,
                    z_style=z_rna_style,
                )
            else:
                rna_rec = None
                z_rna_style = None
                z_cls_rna_single = None
                mu_rna_cls_mm = None
                logvar_rna_cls_mm = None
                mu_rna_cls_single = None
                logvar_rna_cls_single = None
                mu_rna_style = None
                logvar_rna_style = None

            # ===== LOSS =====
            (loss,
             loss_rec, loss_rec_img, loss_rec_rna,
             loss_kl, kl_cls, kl_style,
             kl_style_img, kl_style_rna) = self.compute_loss(
                img_rec=img_rec,
                img_target=img_target,
                rna_rec=rna_rec,
                rna_target=rna_target,
                mm_mu_cls_mm=mm_mu_cls_mm,
                mm_logvar_cls_mm=mm_logvar_cls_mm,
                mu_img_style=mu_img_style,
                logvar_img_style=logvar_img_style,
                mu_rna_style=mu_rna_style,
                logvar_rna_style=logvar_rna_style,
                labels=labels,
                current_epoch=current_epoch,
                mu_img_cls_single=mu_img_cls_single,
                logvar_img_cls_single=logvar_img_cls_single,
                mu_rna_cls_single=mu_rna_cls_single,
                logvar_rna_cls_single=logvar_rna_cls_single,
                mu_img_cls_mm=mu_img_cls_mm,
                logvar_img_cls_mm=logvar_img_cls_mm,
                mu_rna_cls_mm=mu_rna_cls_mm,
                logvar_rna_cls_mm=logvar_rna_cls_mm,
                drop_rna=drop_rna,
                warmup=warmup
            )

            # z_cls output = concat([z_cls_img_single, z_cls_mm]) for image
            z_cls_out = torch.cat([z_cls_img_single, z_cls_mm], dim=1)

            return {
                'img_rec': img_rec,
                'rna_rec': rna_rec,
                'z_cls': z_cls_out,                 # [B, 2*z_cls_dim]
                'mm_mu_cls': mm_mu_cls_mm,
                'mm_logvar_cls': mm_logvar_cls_mm,
                'mu_img_style': mu_img_style,
                'logvar_img_style': logvar_img_style,
                'mu_rna_style': mu_rna_style,
                'logvar_rna_style': logvar_rna_style,
                'mu_img_cls_single': mu_img_cls_single,
                'mu_rna_cls_single': mu_rna_cls_single,
                'logvar_img_cls_single': logvar_img_cls_single,
                'logvar_rna_cls_single': logvar_rna_cls_single,
                'loss_total': loss,
                'loss_rec': loss_rec,
                'loss_rec_img': loss_rec_img,
                'loss_rec_rna': loss_rec_rna,
                'loss_kl': loss_kl,
                'loss_kl_cls': kl_cls,
                'loss_kl_style': kl_style,
                'loss_kl_style_img': kl_style_img,
                'loss_kl_style_rna': kl_style_rna,
                'beta': self.beta.detach().item(),
                'drop_rna': drop_rna
            }

        # ---------------- EVAL MODE ----------------
        else:
            has_img = img_x is not None
            has_rna = rna_x is not None
            if not (has_img or has_rna):
                raise ValueError("At least one modality must be provided at eval time.")

            img_rec = None
            rna_rec = None
            z_cls_out = None

            # ----- Encode image if available -----
            if has_img:
                img_x = img_x.unsqueeze(1) if len(img_x.shape) == 3 else img_x
                if img_target is not None:
                    img_target = img_target.unsqueeze(1)  # kept for API symmetry, not used

                img_enc = self.img_branch.encode(img_x)
                mu_img_cls_single = img_enc["mu_cls_single"]
                logvar_img_cls_single = img_enc["logvar_cls_single"]
                mu_img_cls_mm = img_enc["mu_cls_mm"]
                logvar_img_cls_mm = img_enc["logvar_cls_mm"]
                mu_img_style = img_enc["mu_style"]
                logvar_img_style = img_enc["logvar_style"]
                x4 = img_enc["x4"]
                N = img_enc["batch_size"]
            else:
                mu_img_cls_single = mu_img_cls_mm = mu_img_style = None
                logvar_img_cls_single = logvar_img_cls_mm = logvar_img_style = None
                x4 = None
                N = None

            # ----- Encode RNA if available -----
            if has_rna:
                rna_enc = self.rna_branch.encode(rna_x)
                mu_rna_cls_single = rna_enc["mu_cls_single"]
                logvar_rna_cls_single = rna_enc["logvar_cls_single"]
                mu_rna_cls_mm = rna_enc["mu_cls_mm"]
                logvar_rna_cls_mm = rna_enc["logvar_cls_mm"]
                mu_rna_style = rna_enc["mu_style"]
                logvar_rna_style = rna_enc["logvar_style"]
            else:
                mu_rna_cls_single = mu_rna_cls_mm = mu_rna_style = None
                logvar_rna_cls_single = logvar_rna_cls_mm = logvar_rna_style = None

            # ----- Fuse multimodal cls_mm tokens (MoE degenerates if single expert) -----
            mm_mu_cls_mm, mm_logvar_cls_mm = self.fuse_cls_moe(
                mu_img_cls_mm=mu_img_cls_mm,
                logvar_img_cls_mm=logvar_img_cls_mm,
                mu_rna_cls_mm=mu_rna_cls_mm,
                logvar_rna_cls_mm=logvar_rna_cls_mm,
            )
            z_cls_mm = mm_mu_cls_mm  # deterministic at eval

            # ----- Decode image if available -----
            if has_img:
                z_cls_img_single = mu_img_cls_single  # deterministic
                z_img_style = mu_img_style

                img_rec = self.img_branch.decode(
                    z_cls_single=z_cls_img_single,
                    z_cls_mm=z_cls_mm,
                    z_style=z_img_style,
                    x4=x4,
                    batch_size=N,
                )

                # z_cls output for image = concat([cls_single_img, cls_mm_fused])
                z_cls_out = torch.cat([z_cls_img_single, z_cls_mm], dim=1)

            # ----- Decode RNA if available -----
            if has_rna:
                z_cls_rna_single = mu_rna_cls_single
                z_rna_style = mu_rna_style

                rna_rec = self.rna_branch.decode(
                    z_cls_single=z_cls_rna_single,
                    z_cls_mm=z_cls_mm,
                    z_style=z_rna_style,
                )

                if z_cls_out is None:  # no image case
                    z_cls_out = torch.cat([z_cls_rna_single, z_cls_mm], dim=1)

            out = {
                'img_rec': img_rec,
                'rna_rec': rna_rec,
                'z_cls': z_cls_out,                 # [B, 2*z_cls_dim]
                'mm_mu_cls': mm_mu_cls_mm,
                'mm_logvar_cls': mm_logvar_cls_mm,
            }
            if has_img:
                out.update({
                    'mu_img_style': mu_img_style,
                    'logvar_img_style': logvar_img_style,
                    'mu_img_cls_single': mu_img_cls_single,
                    'logvar_img_cls_single': logvar_img_cls_single,
                })
            if has_rna:
                out.update({
                    'mu_rna_style': mu_rna_style,
                    'logvar_rna_style': logvar_rna_style,
                    'mu_rna_cls_single': mu_rna_cls_single,
                    'logvar_rna_cls_single': logvar_rna_cls_single,
                })
            return out


class _LegacyImageBranchVAEAblation(nn.Module):
    """
    Image branch with configurable latent parts:
      - z_cls_single (optional)
      - z_cls_mm (optional)
      - z_style (optional)
    """

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
            nn.InstanceNorm2d(img_tail),
            nn.ReLU(inplace=True),
        )
        self.img_outc = nn.Conv2d(img_tail, out_channels_img, kernel_size=1)

        with torch.no_grad():
            H, W = self.img_size
            dummy = torch.zeros(1, in_channels_img, H, W)
            x1 = self.img_inc(dummy)
            x2 = self.img_down1(x1)
            x3 = self.img_down2(x2)
            x4 = self.img_down3(x3)
            x5 = self.img_down4(x4)
            _, c5_enc, H5, W5 = x5.shape

            d5 = x5
            d4 = self.img_up1(d5)
            d3 = self.img_up2(d4)
            d2 = self.img_up3(d3)
            d1 = self.img_up4(d2)

            _, c4_dec, H4, W4 = d4.shape
            _, c3_dec, H3, W3 = d3.shape
            _, c2_dec, H2, W2 = d2.shape
            _, c1_dec, H1, W1 = d1.shape
            _, c4_enc, H4_enc, W4_enc = x4.shape

        self.enc_out_channels = c5_enc
        self.enc_out_hw = (H5, W5)
        self.dec_shapes = {
            'c5': (c5_enc, H5, W5),
            'c4_enc': (c4_enc, H4_enc, W4_enc),
            'c4_dec': (c4_dec, H4, W4),
            'c3': (c3_dec, H3, W3),
            'c2': (c2_dec, H2, W2),
            'c1': (c1_dec, H1, W1),
        }

        c5_ch, H5_s, W5_s = self.dec_shapes['c5']
        self.z_to_c5 = nn.Linear(self.z_dim, c5_ch * H5_s * W5_s)
        self.img_z_to_c5 = nn.Conv2d(c5_ch, c5_ch, kernel_size=1)

        c4_enc_ch, H4_enc, W4_enc = self.dec_shapes['c4_enc']
        c4_dec_ch, H4_s, W4_s = self.dec_shapes['c4_dec']
        assert H4_enc == H4_s and W4_enc == W4_s, "x4 and decoder d4 spatial sizes must match"

        self.skip4 = nn.Sequential(
            nn.Conv2d(c4_enc_ch, c4_dec_ch, kernel_size=1, bias=False),
            nn.InstanceNorm2d(c4_dec_ch),
            nn.ReLU(inplace=True),
        )
        self.skip4_dropout = nn.Dropout2d(p=skip4_dropout_p)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.LayerNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1.0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def encode(self, img_x: torch.Tensor):
        x1 = self.img_inc(img_x)
        x2 = self.img_down1(x1)
        x3 = self.img_down2(x2)
        x4 = self.img_down3(x3)
        x5 = self.img_down4(x4)
        B, C5, H5, W5 = x5.shape

        H5_enc, W5_enc = self.enc_out_hw
        assert H5 == H5_enc and W5 == W5_enc, "Input size mismatch vs img_size used in __init__"

        pooled_img = self.img_enc_pool(x5).view(B, C5)

        mu_img_cls_single = self.cls_mu_img(pooled_img) if self.cls_mu_img is not None else None
        logvar_img_cls_single = self.cls_logvar_img(pooled_img) if self.cls_logvar_img is not None else None
        mu_img_cls_mm = self.cls_mu_mm_img(pooled_img) if self.cls_mu_mm_img is not None else None
        logvar_img_cls_mm = self.cls_logvar_mm_img(pooled_img) if self.cls_logvar_mm_img is not None else None
        mu_img_style = self.cls_style_mu_img(pooled_img) if self.cls_style_mu_img is not None else None
        logvar_img_style = self.cls_style_logvar_img(pooled_img) if self.cls_style_logvar_img is not None else None

        return {
            "mu_cls_single": mu_img_cls_single,
            "logvar_cls_single": logvar_img_cls_single,
            "mu_cls_mm": mu_img_cls_mm,
            "logvar_cls_mm": logvar_img_cls_mm,
            "mu_style": mu_img_style,
            "logvar_style": logvar_img_style,
            "x4": x4,
            "batch_size": B,
        }

    def _cat_parts(self, z_cls_single, z_cls_mm, z_style):
        parts = []
        if self.z_cls_dim > 0:
            if z_cls_single is None:
                raise ValueError("z_cls_single is required but missing.")
            parts.append(z_cls_single)
        if self.z_mm_dim > 0:
            if z_cls_mm is None:
                raise ValueError("z_cls_mm is required but missing.")
            parts.append(z_cls_mm)
        if self.z_style_dim > 0:
            if z_style is None:
                raise ValueError("z_style is required but missing.")
            parts.append(z_style)
        if not parts:
            raise ValueError("At least one latent part must be provided.")
        return torch.cat(parts, dim=1) if len(parts) > 1 else parts[0]

    def decode(
        self,
        z_cls_single: torch.Tensor = None,
        z_cls_mm: torch.Tensor = None,
        z_style: torch.Tensor = None,
        x4: torch.Tensor = None,
        batch_size: int = None,
    ):
        z_img_full = self._cat_parts(z_cls_single, z_cls_mm, z_style)
        assert z_img_full.shape[1] == self.z_dim

        c5_ch, H5_s, W5_s = self.dec_shapes['c5']
        z5 = self.z_to_c5(z_img_full).view(batch_size, c5_ch, H5_s, W5_s)
        z5 = self.img_z_to_c5(z5)

        d4 = self.img_up1(z5)

        c4_enc_ch, H4_enc, W4_enc = self.dec_shapes['c4_enc']
        c4_dec_ch, H4_s, W4_s = self.dec_shapes['c4_dec']
        assert d4.shape[2:] == (H4_s, W4_s)
        assert x4.shape[2:] == (H4_enc, W4_enc)

        skip4 = self.skip4(x4)
        if self.training:
            skip4 = self.skip4_dropout(skip4)
        d4 = d4 + skip4

        d3 = self.img_up2(d4)
        d2 = self.img_up3(d3)
        d1 = self.img_up4(d2)

        x_dec = self.img_tail(d1)
        logits_img = self.img_outc(x_dec)
        img_rec = torch.sigmoid(logits_img)

        return img_rec


class RNABranchVAEAblation(nn.Module):
    """
    RNA branch with configurable latent parts:
      - z_cls_single (optional)
      - z_cls_mm (optional)
      - z_style (optional)
    """

    def __init__(
        self,
        input_dim_rna: int = 1000,
        hidden_dim_rna: int = 256,
        output_dim_rna: int = 1000,
        z_cls_dim: int = 0,
        z_mm_dim: int = 0,
        z_style_dim: int = 0,
    ):
        super().__init__()

        self.z_cls_dim = int(z_cls_dim)
        self.z_mm_dim = int(z_mm_dim)
        self.z_style_dim = int(z_style_dim)
        self.z_dim = self.z_cls_dim + self.z_mm_dim + self.z_style_dim
        if self.z_dim <= 0:
            raise ValueError("RNABranchVAEAblation: z_dim must be > 0.")

        self.rna_encoder = RNAEncoder(input_dim_rna, hidden_dim_rna, self.z_dim, output='latent')
        self.rna_decoder = RNADecoder(self.z_dim, hidden_dim_rna, output_dim_rna)

        self.cls_mu_rna = nn.Linear(hidden_dim_rna, self.z_cls_dim) if self.z_cls_dim > 0 else None
        self.cls_logvar_rna = nn.Linear(hidden_dim_rna, self.z_cls_dim) if self.z_cls_dim > 0 else None
        self.cls_mu_mm_rna = nn.Linear(hidden_dim_rna, self.z_mm_dim) if self.z_mm_dim > 0 else None
        self.cls_logvar_mm_rna = nn.Linear(hidden_dim_rna, self.z_mm_dim) if self.z_mm_dim > 0 else None
        self.cls_style_mu_rna = nn.Linear(hidden_dim_rna, self.z_style_dim) if self.z_style_dim > 0 else None
        self.cls_style_logvar_rna = nn.Linear(hidden_dim_rna, self.z_style_dim) if self.z_style_dim > 0 else None

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def encode(self, rna_x: torch.Tensor):
        h_rna = self.rna_encoder(rna_x)

        mu_rna_cls_single = self.cls_mu_rna(h_rna) if self.cls_mu_rna is not None else None
        logvar_rna_cls_single = self.cls_logvar_rna(h_rna) if self.cls_logvar_rna is not None else None
        mu_rna_cls_mm = self.cls_mu_mm_rna(h_rna) if self.cls_mu_mm_rna is not None else None
        logvar_rna_cls_mm = self.cls_logvar_mm_rna(h_rna) if self.cls_logvar_mm_rna is not None else None
        mu_rna_style = self.cls_style_mu_rna(h_rna) if self.cls_style_mu_rna is not None else None
        logvar_rna_style = self.cls_style_logvar_rna(h_rna) if self.cls_style_logvar_rna is not None else None

        return {
            "mu_cls_single": mu_rna_cls_single,
            "logvar_cls_single": logvar_rna_cls_single,
            "mu_cls_mm": mu_rna_cls_mm,
            "logvar_cls_mm": logvar_rna_cls_mm,
            "mu_style": mu_rna_style,
            "logvar_style": logvar_rna_style,
            "h_rna": h_rna,
        }

    def _cat_parts(self, z_cls_single, z_cls_mm, z_style):
        parts = []
        if self.z_cls_dim > 0:
            if z_cls_single is None:
                raise ValueError("z_cls_single is required but missing.")
            parts.append(z_cls_single)
        if self.z_mm_dim > 0:
            if z_cls_mm is None:
                raise ValueError("z_cls_mm is required but missing.")
            parts.append(z_cls_mm)
        if self.z_style_dim > 0:
            if z_style is None:
                raise ValueError("z_style is required but missing.")
            parts.append(z_style)
        if not parts:
            raise ValueError("At least one latent part must be provided.")
        return torch.cat(parts, dim=1) if len(parts) > 1 else parts[0]

    def decode(
        self,
        z_cls_single: torch.Tensor = None,
        z_cls_mm: torch.Tensor = None,
        z_style: torch.Tensor = None,
    ):
        z_rna_full = self._cat_parts(z_cls_single, z_cls_mm, z_style)
        assert z_rna_full.shape[1] == self.z_dim
        rna_rec = self.rna_decoder(z_rna_full)
        return rna_rec


class MultiModalVAEDisMoEwProjectorsAblation(nn.Module):
    """
    MultiModalVAEDisMoEwProjectors ablations with missing-RNA support.

    Modes:
      - "none":         single z_mm only (no cls, no style)
      - "style_mm":     z_mm + z_style
      - "mm_cls":       z_mm + z_cls
      - "style_cls":    z_style + z_cls
      - "all":          z_mm + z_cls + z_style (default)
      - "no_mm_align":  like "all", but skips alignment KL from unimodal mm tokens
    """

    def __init__(
        self,
        in_channels_img=3,
        out_channels_img=3,
        img_base=(16, 32, 64, 128, 256),
        img_tail=32,
        img_size=(256, 256),
        skip4_dropout_p: float = 0.3,
        input_dim_rna=19134,
        hidden_dim_rna=256,
        output_dim_rna=19134,
        z_dim=256,
        *,
        z_cls_dim: int = 16,
        ablation_mode: str = "all",
        num_classes: int = 2,
        beta_start: float = 0.0,
        beta_end: float = 1.0,
        total_epochs: int = 100,
        warmup_start_epoch: int = 0,
        mm_kl_delay_epochs: int = 100,
        p_drop_rna: float = 0.3,
        trainable_priors: bool = True,
        prior_means: torch.Tensor = None,
        prior_logvars: torch.Tensor = None,
    ):
        super().__init__()

        mode = str(ablation_mode).lower()
        if mode not in {"none", "style_mm", "mm_cls", "style_cls", "all", "no_mm_align"}:
            raise ValueError(f"Unknown ablation_mode={ablation_mode}")

        self.z_dim = int(z_dim)
        base_cls_dim = int(z_cls_dim)

        if mode == "none":
            self.z_cls_single_dim = 0
            self.z_cls_mm_dim = self.z_dim
            self.z_style_dim = 0
        elif mode == "style_mm":
            self.z_cls_single_dim = 0
            self.z_cls_mm_dim = base_cls_dim
            self.z_style_dim = self.z_dim - self.z_cls_mm_dim
        elif mode == "mm_cls":
            self.z_cls_single_dim = base_cls_dim
            self.z_cls_mm_dim = self.z_dim - self.z_cls_single_dim
            self.z_style_dim = 0
        elif mode == "style_cls":
            self.z_cls_single_dim = base_cls_dim
            self.z_cls_mm_dim = 0
            self.z_style_dim = self.z_dim - self.z_cls_single_dim
        elif mode == "no_mm_align":
            self.z_cls_single_dim = base_cls_dim
            self.z_cls_mm_dim = base_cls_dim
            self.z_style_dim = self.z_dim - self.z_cls_single_dim - self.z_cls_mm_dim
        else:
            self.z_cls_single_dim = base_cls_dim
            self.z_cls_mm_dim = base_cls_dim
            self.z_style_dim = self.z_dim - self.z_cls_single_dim - self.z_cls_mm_dim

        if self.z_style_dim < 0 or self.z_cls_mm_dim < 0 or self.z_cls_single_dim < 0:
            raise ValueError("Invalid latent dimensions for ablation configuration.")
        if self.z_cls_single_dim + self.z_cls_mm_dim + self.z_style_dim != self.z_dim:
            raise ValueError("Latent dimensions must sum to z_dim.")

        self.z_cls_dim = self.z_cls_single_dim
        self.num_classes = int(num_classes)
        self.mm_kl_delay_epochs = int(mm_kl_delay_epochs)
        self.img_size = img_size
        self.p_drop_rna = float(p_drop_rna)
        self.ablation_mode = mode
        self.align_mm_tokens = mode != "no_mm_align"

        self.img_branch = ImageBranchVAEAblation(
            in_channels_img=in_channels_img,
            out_channels_img=out_channels_img,
            img_base=img_base,
            img_tail=img_tail,
            img_size=img_size,
            z_cls_dim=self.z_cls_single_dim,
            z_mm_dim=self.z_cls_mm_dim,
            z_style_dim=self.z_style_dim,
            skip4_dropout_p=skip4_dropout_p,
        )

        self.rna_branch = RNABranchVAEAblation(
            input_dim_rna=input_dim_rna,
            hidden_dim_rna=hidden_dim_rna,
            output_dim_rna=output_dim_rna,
            z_cls_dim=self.z_cls_single_dim,
            z_mm_dim=self.z_cls_mm_dim,
            z_style_dim=self.z_style_dim,
        )

        self.prior_means_img = None
        self.prior_logvars_img = None
        self.prior_means_rna = None
        self.prior_logvars_rna = None
        self.prior_means_mm = None
        self.prior_logvars_mm = None

        def _init_prior(dim):
            if prior_means is None:
                if self.num_classes == 2:
                    means = torch.stack([torch.zeros(dim), torch.ones(dim) * 2.0])
                else:
                    means = torch.stack([torch.ones(dim) * (2.0 * k) for k in range(self.num_classes)], dim=0)
            else:
                means = prior_means
            if prior_logvars is None:
                logvars = torch.zeros(self.num_classes, dim)
            else:
                logvars = prior_logvars
            return means, logvars

        eps = 1e-2
        if self.z_cls_single_dim > 0:
            means, logvars = _init_prior(self.z_cls_single_dim)
            self.prior_means_img = nn.Parameter(
                means.clone() + eps * torch.randn_like(means),
                requires_grad=trainable_priors
            )
            self.prior_logvars_img = nn.Parameter(
                logvars.clone() + eps * torch.randn_like(logvars),
                requires_grad=trainable_priors
            )
            self.prior_means_rna = nn.Parameter(
                means.clone() + eps * torch.randn_like(means),
                requires_grad=trainable_priors
            )
            self.prior_logvars_rna = nn.Parameter(
                logvars.clone() + eps * torch.randn_like(logvars),
                requires_grad=trainable_priors
            )

        if self.z_cls_mm_dim > 0:
            means, logvars = _init_prior(self.z_cls_mm_dim)
            self.prior_means_mm = nn.Parameter(
                means.clone() + eps * torch.randn_like(means),
                requires_grad=trainable_priors
            )
            self.prior_logvars_mm = nn.Parameter(
                logvars.clone() + eps * torch.randn_like(logvars),
                requires_grad=trainable_priors
            )

        self.moe_logits = nn.Parameter(torch.zeros(2)) if self.z_cls_mm_dim > 0 else None

        self.beta_start = float(beta_start)
        self.beta_end = float(beta_end)
        self.total_epochs = int(total_epochs)
        self.warmup_start_epoch = int(warmup_start_epoch)
        self.register_buffer("beta", torch.tensor(0.0, dtype=torch.float32))

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.LayerNorm)):
                if getattr(m, "weight", None) is not None:
                    nn.init.constant_(m.weight, 1.0)
                if getattr(m, "bias", None) is not None:
                    nn.init.constant_(m.bias, 0.0)

    def _beta_at(self, current_epoch: int) -> float:
        if current_epoch < self.warmup_start_epoch:
            return 0.0
        warmup_epochs = self.total_epochs - self.warmup_start_epoch
        if warmup_epochs <= 1:
            progress = 1.0
        else:
            e = max(0, min(current_epoch - self.warmup_start_epoch, warmup_epochs - 1))
            progress = e / (warmup_epochs - 1)
        return self.beta_start + progress * (self.beta_end - self.beta_start)

    def _update_beta(self, current_epoch: int):
        new_beta = self._beta_at(current_epoch)
        self.beta.copy_(torch.tensor(new_beta, dtype=self.beta.dtype, device=self.beta.device))

    def fuse_cls_moe(
        self,
        mu_img_cls_mm: torch.Tensor = None,
        logvar_img_cls_mm: torch.Tensor = None,
        mu_rna_cls_mm: torch.Tensor = None,
        logvar_rna_cls_mm: torch.Tensor = None,
    ):
        if self.z_cls_mm_dim == 0:
            raise ValueError("fuse_cls_moe called but z_cls_mm_dim is 0.")

        experts_mu = []
        experts_logvar = []
        expert_ids = []

        if mu_img_cls_mm is not None and logvar_img_cls_mm is not None:
            experts_mu.append(mu_img_cls_mm)
            experts_logvar.append(logvar_img_cls_mm)
            expert_ids.append(0)

        if mu_rna_cls_mm is not None and logvar_rna_cls_mm is not None:
            experts_mu.append(mu_rna_cls_mm)
            experts_logvar.append(logvar_rna_cls_mm)
            expert_ids.append(1)

        num_experts = len(experts_mu)
        if num_experts == 0:
            raise ValueError("fuse_cls_moe: at least one expert must be provided.")
        if num_experts == 1:
            return experts_mu[0], experts_logvar[0]

        w = F.softmax(self.moe_logits, dim=0)
        w_img, w_rna = w[0], w[1]
        weights = [(w_img if eid == 0 else w_rna) for eid in expert_ids]

        vars_ = [torch.exp(lv) for lv in experts_logvar]
        var_moe = sum(wi * vi for wi, vi in zip(weights, vars_))
        mu_moe = sum(wi * mi for wi, mi in zip(weights, experts_mu))
        logvar_moe = torch.log(var_moe + 1e-8)
        return mu_moe, logvar_moe

    def compute_loss(
        self,
        img_rec,
        img_target,
        rna_rec,
        rna_target,
        mm_mu_cls_mm,
        mm_logvar_cls_mm,
        mu_img_style,
        logvar_img_style,
        mu_rna_style,
        logvar_rna_style,
        labels,
        current_epoch: int,
        mu_img_cls_single=None,
        logvar_img_cls_single=None,
        mu_rna_cls_single=None,
        logvar_rna_cls_single=None,
        mu_img_cls_mm=None,
        logvar_img_cls_mm=None,
        mu_rna_cls_mm=None,
        logvar_rna_cls_mm=None,
        rna_present: torch.Tensor = None,
        warmup: bool = False,
    ):
        device = img_rec.device
        labels = labels.long()

        if rna_present is None:
            rna_present = torch.ones(labels.shape[0], dtype=torch.bool, device=device)

        loss_rec_img = F.l1_loss(img_rec, img_target, reduction="mean")
        if rna_present.any():
            idx = rna_present.nonzero(as_tuple=True)[0]
            loss_rec_rna = F.l1_loss(rna_rec[idx], rna_target[idx], reduction="mean")
        else:
            loss_rec_rna = torch.zeros((), device=device, dtype=loss_rec_img.dtype)
        loss_rec = 0.5 * (loss_rec_img + loss_rec_rna)

        kl_terms = []

        if self.z_cls_single_dim > 0:
            prior_mu_img = self.prior_means_img[labels]
            prior_lv_img = self.prior_logvars_img[labels]
            prior_mu_rna = self.prior_means_rna[labels]
            prior_lv_rna = self.prior_logvars_rna[labels]

            if mu_img_cls_single is not None and logvar_img_cls_single is not None:
                kl_terms.append(
                    kl_divergence(mu_img_cls_single, logvar_img_cls_single,
                                  prior_mu_img, prior_lv_img).mean()
                )

            if (mu_rna_cls_single is not None) and (logvar_rna_cls_single is not None) and rna_present.any():
                idx = rna_present.nonzero(as_tuple=True)[0]
                kl_terms.append(
                    kl_divergence(mu_rna_cls_single[idx], logvar_rna_cls_single[idx],
                                  prior_mu_rna[idx], prior_lv_rna[idx]).mean()
                )

        if self.z_cls_mm_dim > 0:
            prior_mu_mm = self.prior_means_mm[labels]
            prior_lv_mm = self.prior_logvars_mm[labels]

            mm_kl_start = self.warmup_start_epoch + self.mm_kl_delay_epochs
            if current_epoch >= mm_kl_start:
                kl_terms.append(
                    kl_divergence(mm_mu_cls_mm, mm_logvar_cls_mm,
                                  prior_mu_mm, prior_lv_mm).mean()
                )

            if self.align_mm_tokens:
                if mu_img_cls_mm is not None and logvar_img_cls_mm is not None:
                    kl_terms.append(
                        kl_divergence(mu_img_cls_mm, logvar_img_cls_mm,
                                      mm_mu_cls_mm.detach(), mm_logvar_cls_mm.detach()).mean()
                    )
                if (mu_rna_cls_mm is not None) and (logvar_rna_cls_mm is not None) and rna_present.any():
                    idx = rna_present.nonzero(as_tuple=True)[0]
                    kl_terms.append(
                        kl_divergence(mu_rna_cls_mm[idx], logvar_rna_cls_mm[idx],
                                      mm_mu_cls_mm.detach()[idx], mm_logvar_cls_mm.detach()[idx]).mean()
                    )

        kl_cls = sum(kl_terms) if len(kl_terms) > 0 else torch.zeros((), device=device)

        if self.z_style_dim > 0:
            kl_style_img = kl_divergence(mu_img_style, logvar_img_style).mean()
            if (mu_rna_style is not None) and (logvar_rna_style is not None) and rna_present.any():
                idx = rna_present.nonzero(as_tuple=True)[0]
                kl_style_rna = kl_divergence(mu_rna_style[idx], logvar_rna_style[idx]).mean()
            else:
                kl_style_rna = torch.zeros((), device=device, dtype=kl_style_img.dtype)
            kl_style = 0.5 * (kl_style_img + kl_style_rna)
        else:
            kl_style_img = torch.zeros((), device=device)
            kl_style_rna = torch.zeros((), device=device)
            kl_style = torch.zeros((), device=device)

        loss_kl = kl_cls + kl_style
        if warmup:
            loss = loss_rec
        else:
            loss = loss_rec + self.beta * kl_style + self.beta * kl_cls

        return (loss,
                loss_rec, loss_rec_img, loss_rec_rna,
                loss_kl, kl_cls, kl_style,
                kl_style_img, kl_style_rna)

    def forward(
        self,
        img_x=None,
        img_target=None,
        rna_x=None,
        rna_target=None,
        rna_missing=None,
        labels=None,
        current_epoch: int = 0,
        eval: bool = False,
        warmup: bool = False,
    ):
        self._update_beta(current_epoch)

        def _ensure_img_4d(x):
            if x is None:
                return None
            return x.unsqueeze(1) if x.dim() == 3 else x

        def _zeros_part(B, dim, device, like=None):
            if dim <= 0:
                return None
            if like is not None:
                return torch.zeros_like(like)
            return torch.zeros(B, dim, device=device)

        if eval:
            has_img = img_x is not None
            has_rna = rna_x is not None
            if not (has_img or has_rna):
                raise ValueError("At least one modality must be provided at eval time.")

            img_x = _ensure_img_4d(img_x)

            if has_img:
                B = img_x.shape[0]
                device = img_x.device
            else:
                B = rna_x.shape[0]
                device = rna_x.device

            if has_rna:
                if rna_missing is None:
                    rna_present = torch.ones(B, dtype=torch.bool, device=device)
                else:
                    rna_present = ~rna_missing.to(device).bool()
            else:
                rna_present = torch.zeros(B, dtype=torch.bool, device=device)

            if has_img:
                img_enc = self.img_branch.encode(img_x)
                mu_img_cls_single = img_enc["mu_cls_single"]
                logvar_img_cls_single = img_enc["logvar_cls_single"]
                mu_img_cls_mm = img_enc["mu_cls_mm"]
                logvar_img_cls_mm = img_enc["logvar_cls_mm"]
                mu_img_style = img_enc["mu_style"]
                logvar_img_style = img_enc["logvar_style"]
                x4 = img_enc["x4"]
                N = img_enc["batch_size"]
            else:
                mu_img_cls_single = logvar_img_cls_single = None
                mu_img_cls_mm = logvar_img_cls_mm = None
                mu_img_style = logvar_img_style = None
                x4 = None
                N = None

            if has_img:
                mu_rna_cls_single = _zeros_part(B, self.z_cls_single_dim, device, like=mu_img_cls_single)
                logvar_rna_cls_single = _zeros_part(B, self.z_cls_single_dim, device, like=logvar_img_cls_single)
                mu_rna_cls_mm = _zeros_part(B, self.z_cls_mm_dim, device, like=mu_img_cls_mm)
                logvar_rna_cls_mm = _zeros_part(B, self.z_cls_mm_dim, device, like=logvar_img_cls_mm)
                mu_rna_style = _zeros_part(B, self.z_style_dim, device, like=mu_img_style)
                logvar_rna_style = _zeros_part(B, self.z_style_dim, device, like=logvar_img_style)
            else:
                mu_rna_cls_single = logvar_rna_cls_single = None
                mu_rna_cls_mm = logvar_rna_cls_mm = None
                mu_rna_style = logvar_rna_style = None

            if has_rna and rna_present.any():
                idx = rna_present.nonzero(as_tuple=True)[0]
                rna_enc = self.rna_branch.encode(rna_x[idx])

                if has_img:
                    if mu_rna_cls_single is not None:
                        mu_rna_cls_single[idx] = rna_enc["mu_cls_single"]
                        logvar_rna_cls_single[idx] = rna_enc["logvar_cls_single"]
                    if mu_rna_cls_mm is not None:
                        mu_rna_cls_mm[idx] = rna_enc["mu_cls_mm"]
                        logvar_rna_cls_mm[idx] = rna_enc["logvar_cls_mm"]
                    if mu_rna_style is not None:
                        mu_rna_style[idx] = rna_enc["mu_style"]
                        logvar_rna_style[idx] = rna_enc["logvar_style"]
                else:
                    mu_rna_cls_single = rna_enc["mu_cls_single"]
                    logvar_rna_cls_single = rna_enc["logvar_cls_single"]
                    mu_rna_cls_mm = rna_enc["mu_cls_mm"]
                    logvar_rna_cls_mm = rna_enc["logvar_cls_mm"]
                    mu_rna_style = rna_enc["mu_style"]
                    logvar_rna_style = rna_enc["logvar_style"]

            mm_mu_cls = None
            mm_logvar_cls = None
            z_cls_mm = None
            if self.z_cls_mm_dim > 0:
                if has_img:
                    mm_mu_both, mm_lv_both = self.fuse_cls_moe(
                        mu_img_cls_mm=mu_img_cls_mm,
                        logvar_img_cls_mm=logvar_img_cls_mm,
                        mu_rna_cls_mm=mu_rna_cls_mm if has_rna and rna_present.any() else None,
                        logvar_rna_cls_mm=logvar_rna_cls_mm if has_rna and rna_present.any() else None,
                    )
                    mm_mu_img, mm_lv_img = self.fuse_cls_moe(
                        mu_img_cls_mm=mu_img_cls_mm,
                        logvar_img_cls_mm=logvar_img_cls_mm,
                        mu_rna_cls_mm=None,
                        logvar_rna_cls_mm=None,
                    )
                    mm_mu_cls = torch.where(rna_present[:, None], mm_mu_both, mm_mu_img)
                    mm_logvar_cls = torch.where(rna_present[:, None], mm_lv_both, mm_lv_img)
                    z_cls_mm = mm_mu_cls
                else:
                    if not (has_rna and rna_present.any()):
                        raise ValueError("RNA-only eval requires at least one present RNA sample.")
                    mm_mu_cls = mu_rna_cls_mm
                    mm_logvar_cls = logvar_rna_cls_mm
                    z_cls_mm = mm_mu_cls

            img_rec = None
            rna_rec = None
            z_cls_out = None

            if has_img:
                z_cls_img_single = mu_img_cls_single
                z_img_style = mu_img_style
                img_rec = self.img_branch.decode(
                    z_cls_single=z_cls_img_single,
                    z_cls_mm=z_cls_mm,
                    z_style=z_img_style,
                    x4=x4,
                    batch_size=N,
                )
                z_cls_parts = []
                if z_cls_img_single is not None:
                    z_cls_parts.append(z_cls_img_single)
                if z_cls_mm is not None:
                    z_cls_parts.append(z_cls_mm)
                z_cls_out = torch.cat(z_cls_parts, dim=1) if len(z_cls_parts) > 1 else z_cls_parts[0]

            if has_rna and rna_present.any():
                idx = rna_present.nonzero(as_tuple=True)[0]
                if has_img:
                    rna_rec_sub = self.rna_branch.decode(
                        z_cls_single=mu_rna_cls_single[idx] if mu_rna_cls_single is not None else None,
                        z_cls_mm=z_cls_mm[idx] if z_cls_mm is not None else None,
                        z_style=mu_rna_style[idx] if mu_rna_style is not None else None,
                    )
                    rna_rec = torch.zeros_like(rna_x)
                    rna_rec[idx] = rna_rec_sub
                else:
                    rna_rec_sub = self.rna_branch.decode(
                        z_cls_single=mu_rna_cls_single,
                        z_cls_mm=mu_rna_cls_mm,
                        z_style=mu_rna_style,
                    )
                    rna_rec = torch.zeros_like(rna_x)
                    rna_rec[idx] = rna_rec_sub
                    if z_cls_out is None:
                        z_cls_parts = []
                        if mu_rna_cls_single is not None:
                            z_cls_parts.append(mu_rna_cls_single)
                        if mu_rna_cls_mm is not None:
                            z_cls_parts.append(mu_rna_cls_mm)
                        z_cls_out = torch.cat(z_cls_parts, dim=1) if len(z_cls_parts) > 1 else z_cls_parts[0]

            return {
                "img_rec": img_rec,
                "rna_rec": rna_rec,
                "z_cls": z_cls_out,
                "mm_mu_cls": mm_mu_cls,
                "mm_logvar_cls": mm_logvar_cls,
                "rna_present": rna_present,
                "mu_img_cls_single": mu_img_cls_single,
                "logvar_img_cls_single": logvar_img_cls_single,
                "mu_rna_cls_single": mu_rna_cls_single,
                "logvar_rna_cls_single": logvar_rna_cls_single,
            }

        if img_x is None:
            raise ValueError("Training expects img_x to be present.")

        img_x = _ensure_img_4d(img_x)
        img_target = _ensure_img_4d(img_target)

        B = img_x.shape[0]
        device = img_x.device

        if rna_x is None:
            rna_present = torch.zeros(B, dtype=torch.bool, device=device)
        else:
            if rna_missing is None:
                rna_present = torch.ones(B, dtype=torch.bool, device=device)
            else:
                rna_present = ~rna_missing.to(device).bool()

        if (not warmup) and (self.p_drop_rna > 0) and rna_present.any():
            keep = (torch.rand(B, device=device) >= self.p_drop_rna)
            rna_present = rna_present & keep

        img_enc = self.img_branch.encode(img_x)
        mu_img_cls_single = img_enc["mu_cls_single"]
        logvar_img_cls_single = img_enc["logvar_cls_single"]
        mu_img_cls_mm = img_enc["mu_cls_mm"]
        logvar_img_cls_mm = img_enc["logvar_cls_mm"]
        mu_img_style = img_enc["mu_style"]
        logvar_img_style = img_enc["logvar_style"]
        x4 = img_enc["x4"]
        N = img_enc["batch_size"]

        mu_rna_cls_single = _zeros_part(B, self.z_cls_single_dim, device)
        logvar_rna_cls_single = _zeros_part(B, self.z_cls_single_dim, device)
        mu_rna_cls_mm = _zeros_part(B, self.z_cls_mm_dim, device)
        logvar_rna_cls_mm = _zeros_part(B, self.z_cls_mm_dim, device)
        mu_rna_style = _zeros_part(B, self.z_style_dim, device)
        logvar_rna_style = _zeros_part(B, self.z_style_dim, device)

        if (rna_x is not None) and rna_present.any():
            idx = rna_present.nonzero(as_tuple=True)[0]
            rna_enc = self.rna_branch.encode(rna_x[idx])
            if mu_rna_cls_single is not None:
                mu_rna_cls_single[idx] = rna_enc["mu_cls_single"]
                logvar_rna_cls_single[idx] = rna_enc["logvar_cls_single"]
            if mu_rna_cls_mm is not None:
                mu_rna_cls_mm[idx] = rna_enc["mu_cls_mm"]
                logvar_rna_cls_mm[idx] = rna_enc["logvar_cls_mm"]
            if mu_rna_style is not None:
                mu_rna_style[idx] = rna_enc["mu_style"]
                logvar_rna_style[idx] = rna_enc["logvar_style"]

        mm_mu_cls_mm = None
        mm_logvar_cls_mm = None
        if self.z_cls_mm_dim > 0:
            mm_mu_both, mm_lv_both = self.fuse_cls_moe(
                mu_img_cls_mm=mu_img_cls_mm,
                logvar_img_cls_mm=logvar_img_cls_mm,
                mu_rna_cls_mm=mu_rna_cls_mm if rna_present.any() else None,
                logvar_rna_cls_mm=logvar_rna_cls_mm if rna_present.any() else None,
            )
            mm_mu_img, mm_lv_img = self.fuse_cls_moe(
                mu_img_cls_mm=mu_img_cls_mm,
                logvar_img_cls_mm=logvar_img_cls_mm,
                mu_rna_cls_mm=None,
                logvar_rna_cls_mm=None,
            )
            mm_mu_cls_mm = torch.where(rna_present[:, None], mm_mu_both, mm_mu_img)
            mm_logvar_cls_mm = torch.where(rna_present[:, None], mm_lv_both, mm_lv_img)

        z_cls_img_single = reparameterize(mu_img_cls_single, logvar_img_cls_single) if self.z_cls_single_dim > 0 else None
        z_cls_mm = reparameterize(mm_mu_cls_mm, mm_logvar_cls_mm) if self.z_cls_mm_dim > 0 else None
        z_img_style = reparameterize(mu_img_style, logvar_img_style) if self.z_style_dim > 0 else None

        img_rec = self.img_branch.decode(
            z_cls_single=z_cls_img_single,
            z_cls_mm=z_cls_mm,
            z_style=z_img_style,
            x4=x4,
            batch_size=N,
        )

        if rna_target is not None:
            rna_rec = torch.zeros_like(rna_target)
        else:
            rna_rec = None

        if (rna_target is not None) and (rna_x is not None) and rna_present.any():
            idx = rna_present.nonzero(as_tuple=True)[0]
            z_cls_rna_single_sub = reparameterize(mu_rna_cls_single[idx], logvar_rna_cls_single[idx]) if self.z_cls_single_dim > 0 else None
            z_rna_style_sub = reparameterize(mu_rna_style[idx], logvar_rna_style[idx]) if self.z_style_dim > 0 else None
            rna_rec_sub = self.rna_branch.decode(
                z_cls_single=z_cls_rna_single_sub,
                z_cls_mm=z_cls_mm[idx] if z_cls_mm is not None else None,
                z_style=z_rna_style_sub,
            )
            rna_rec[idx] = rna_rec_sub

        (loss,
         loss_rec, loss_rec_img, loss_rec_rna,
         loss_kl, kl_cls, kl_style,
         kl_style_img, kl_style_rna) = self.compute_loss(
            img_rec=img_rec,
            img_target=img_target,
            rna_rec=rna_rec,
            rna_target=rna_target,
            mm_mu_cls_mm=mm_mu_cls_mm,
            mm_logvar_cls_mm=mm_logvar_cls_mm,
            mu_img_style=mu_img_style,
            logvar_img_style=logvar_img_style,
            mu_rna_style=mu_rna_style,
            logvar_rna_style=logvar_rna_style,
            current_epoch=current_epoch,
            mu_img_cls_single=mu_img_cls_single,
            logvar_img_cls_single=logvar_img_cls_single,
            mu_rna_cls_single=mu_rna_cls_single,
            logvar_rna_cls_single=logvar_rna_cls_single,
            mu_img_cls_mm=mu_img_cls_mm,
            logvar_img_cls_mm=logvar_img_cls_mm,
            mu_rna_cls_mm=mu_rna_cls_mm,
            logvar_rna_cls_mm=logvar_rna_cls_mm,
            rna_present=rna_present,
            warmup=warmup,
            labels=labels,
        )

        z_cls_parts = []
        if z_cls_img_single is not None:
            z_cls_parts.append(z_cls_img_single)
        if z_cls_mm is not None:
            z_cls_parts.append(z_cls_mm)
        z_cls_out = torch.cat(z_cls_parts, dim=1) if len(z_cls_parts) > 1 else z_cls_parts[0]

        return {
            "img_rec": img_rec,
            "rna_rec": rna_rec,
            "z_cls": z_cls_out,
            "mm_mu_cls": mm_mu_cls_mm,
            "mm_logvar_cls": mm_logvar_cls_mm,
            "mu_img_cls_single": mu_img_cls_single,
            "logvar_img_cls_single": logvar_img_cls_single,
            "mu_rna_cls_single": mu_rna_cls_single,
            "logvar_rna_cls_single": logvar_rna_cls_single,
            "loss_total": loss,
            "loss_rec": loss_rec,
            "loss_rec_img": loss_rec_img,
            "loss_rec_rna": loss_rec_rna,
            "loss_kl": loss_kl,
            "loss_kl_cls": kl_cls,
            "loss_kl_style": kl_style,
            "loss_kl_style_img": kl_style_img,
            "loss_kl_style_rna": kl_style_rna,
            "beta": float(self.beta.detach().item()),
            "rna_present": rna_present,
        }






import torch
import torch.nn as nn
import torch.nn.functional as F

# Assumed existing in your project:
# - ImageBranchVAE
# - RNABranchVAE
# - reparameterize(mu, logvar)
# - kl_divergence(mu, logvar)   # KL to standard Normal internally


class MultiModalVAEDisMoEwProjectorswMissingRNA(nn.Module):
    """
    Multimodal disentangled VAE with MoE fusion of class latents.
    Supports missing RNA per-sample using `rna_missing` mask.

    Expected tensors:
      img_x:        [B,H,W] or [B,C,H,W]
      img_target:   same as img_x
      rna_x:        [B,19134] (can be zeros for missing)
      rna_target:   [B,19134]
      rna_missing:  [B] with 1=missing, 0=present
      labels:       [B] (kept for API; not used in loss here)
    """

    def __init__(
        self,
        # image branch
        in_channels_img=3,
        out_channels_img=3,
        img_base=(16, 32, 64, 128, 256),
        img_tail=32,
        img_size=(256, 256),
        skip4_dropout_p: float = 0.3,

        # RNA branch
        input_dim_rna=19134,
        hidden_dim_rna=256,
        output_dim_rna=19134,

        # shared latent
        z_dim=256,
        *,
        z_cls_dim: int = 16,
        num_classes: int = 2,          # kept for API compatibility
        beta_start: float = 0.0,
        beta_end: float = 1.0,
        total_epochs: int = 100,
        warmup_start_epoch: int = 0,
        mm_kl_delay_epochs: int = 100,
        p_drop_rna: float = 0.3,
        trainable_priors: bool = True,
        prior_means: torch.Tensor = None,      # [num_classes, z_cls_dim]
        prior_logvars: torch.Tensor = None,    # [num_classes, z_cls_dim]
    ):
        super().__init__()

        # ----- latent sizes -----
        self.z_dim = int(z_dim)
        self.z_cls_single_dim = int(z_cls_dim)
        self.z_cls_mm_dim = int(z_cls_dim)
        self.z_style_dim = self.z_dim - self.z_cls_single_dim - self.z_cls_mm_dim
        if self.z_style_dim <= 0:
            raise ValueError(
                f"z_dim={z_dim} too small for two cls tokens of size {z_cls_dim}; "
                f"need z_dim > 2*z_cls_dim."
            )

        self.z_cls_dim = int(z_cls_dim)
        self.num_classes = int(num_classes)
        self.mm_kl_delay_epochs = int(mm_kl_delay_epochs)
        self.img_size = img_size
        self.p_drop_rna = float(p_drop_rna)

        # ---------------- MODALITY BRANCHES ----------------
        self.img_branch = ImageBranchVAE(
            in_channels_img=in_channels_img,
            out_channels_img=out_channels_img,
            img_base=img_base,
            img_tail=img_tail,
            img_size=img_size,
            z_dim=self.z_dim,
            z_cls_dim=self.z_cls_single_dim,
            skip4_dropout_p=skip4_dropout_p,
        )

        self.rna_branch = RNABranchVAE(
            input_dim_rna=input_dim_rna,
            hidden_dim_rna=hidden_dim_rna,
            output_dim_rna=output_dim_rna,
            z_dim=self.z_dim,
            z_cls_dim=self.z_cls_single_dim,
        )

        # ---------------- MoE weights (learnable) ----------------
        self.moe_logits = nn.Parameter(torch.zeros(2))

        # ---------------- Beta schedule ----------------
        self.beta_start = float(beta_start)
        self.beta_end = float(beta_end)
        self.total_epochs = int(total_epochs)
        self.warmup_start_epoch = int(warmup_start_epoch)
        self.register_buffer("beta", torch.tensor(0.0, dtype=torch.float32))

        self._init_weights()


       # ---------------- CLASS PRIOR (for z_cls tokens) ----------------
        if prior_means is None:
            if self.num_classes == 2:
                prior_means = torch.stack([
                    torch.zeros(self.z_cls_dim),
                    torch.ones(self.z_cls_dim) * 2.0
                ])
            else:
                means = []
                for k in range(self.num_classes):
                    means.append(torch.ones(self.z_cls_dim) * (2.0 * k))
                prior_means = torch.stack(means, dim=0)
        else:
            assert prior_means.shape == (self.num_classes, self.z_cls_dim)

        if prior_logvars is None:
            prior_logvars = torch.zeros(self.num_classes, self.z_cls_dim)
        else:
            assert prior_logvars.shape == (self.num_classes, self.z_cls_dim)
        eps = 1e-2
        self.prior_means_img = nn.Parameter(
            prior_means.clone() + eps*torch.randn_like(prior_means),
            requires_grad=trainable_priors
        )
        self.prior_logvars_img = nn.Parameter(
            prior_logvars.clone() + eps*torch.randn_like(prior_logvars),
            requires_grad=trainable_priors
        )


        self.prior_means_rna = nn.Parameter(
            prior_means.clone() + eps*torch.randn_like(prior_means),
            requires_grad=trainable_priors
        )
        self.prior_logvars_rna = nn.Parameter(
            prior_logvars + eps*torch.randn_like(prior_logvars),
            requires_grad=trainable_priors
        )


        self.prior_means_mm = nn.Parameter(
            prior_means.clone() + eps*torch.randn_like(prior_means),
            requires_grad=trainable_priors
        )
        self.prior_logvars_mm = nn.Parameter(
            prior_logvars + eps*torch.randn_like(prior_logvars),
            requires_grad=trainable_priors
        )

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.LayerNorm)):
                if getattr(m, "weight", None) is not None:
                    nn.init.constant_(m.weight, 1.0)
                if getattr(m, "bias", None) is not None:
                    nn.init.constant_(m.bias, 0.0)

    def _beta_at(self, current_epoch: int) -> float:
        if current_epoch < self.warmup_start_epoch:
            return 0.0
        warmup_epochs = self.total_epochs - self.warmup_start_epoch
        if warmup_epochs <= 1:
            progress = 1.0
        else:
            e = max(0, min(current_epoch - self.warmup_start_epoch, warmup_epochs - 1))
            progress = e / (warmup_epochs - 1)
        return self.beta_start + progress * (self.beta_end - self.beta_start)

    def _update_beta(self, current_epoch: int):
        new_beta = self._beta_at(current_epoch)
        self.beta.copy_(torch.tensor(new_beta, dtype=self.beta.dtype, device=self.beta.device))

    # --------- MoE fusion for multimodal class latents ----------
    def fuse_cls_moe(
        self,
        mu_img_cls_mm: torch.Tensor = None,
        logvar_img_cls_mm: torch.Tensor = None,
        mu_rna_cls_mm: torch.Tensor = None,
        logvar_rna_cls_mm: torch.Tensor = None,
    ):
        experts_mu = []
        experts_logvar = []
        expert_ids = []  # 0=img, 1=rna

        if mu_img_cls_mm is not None and logvar_img_cls_mm is not None:
            experts_mu.append(mu_img_cls_mm)
            experts_logvar.append(logvar_img_cls_mm)
            expert_ids.append(0)

        if mu_rna_cls_mm is not None and logvar_rna_cls_mm is not None:
            experts_mu.append(mu_rna_cls_mm)
            experts_logvar.append(logvar_rna_cls_mm)
            expert_ids.append(1)

        num_experts = len(experts_mu)
        if num_experts == 0:
            raise ValueError("fuse_cls_moe: at least one expert must be provided.")
        if num_experts == 1:
            return experts_mu[0], experts_logvar[0]

        w = F.softmax(self.moe_logits, dim=0)  # [2]
        w_img, w_rna = w[0], w[1]

        weights = [(w_img if eid == 0 else w_rna) for eid in expert_ids]

        vars_ = [torch.exp(lv) for lv in experts_logvar]
        var_moe = sum(wi * vi for wi, vi in zip(weights, vars_))
        mu_moe = sum(wi * mi for wi, mi in zip(weights, experts_mu))
        logvar_moe = torch.log(var_moe + 1e-8)
        return mu_moe, logvar_moe

    def compute_loss(
        self,
        img_rec,
        img_target,
        rna_rec,
        rna_target,
        mm_mu_cls_mm,
        mm_logvar_cls_mm,
        mu_img_style,
        logvar_img_style,
        mu_rna_style,
        logvar_rna_style,
        labels,
        current_epoch: int,
        mu_img_cls_single=None,
        logvar_img_cls_single=None,
        mu_rna_cls_single=None,
        logvar_rna_cls_single=None,
        mu_img_cls_mm=None,
        logvar_img_cls_mm=None,
        mu_rna_cls_mm=None,
        logvar_rna_cls_mm=None,
        rna_present: torch.Tensor = None,  # [B] bool
        warmup: bool = False,
    ):
        device = img_rec.device
        labels = labels.long()

        if rna_present is None:
            # default: assume RNA present for all if not provided
            rna_present = torch.ones(labels.shape[0], dtype=torch.bool, device=device)

        # ---------------- recon ----------------
        loss_rec_img = F.l1_loss(img_rec, img_target, reduction="mean")

        if rna_present.any():
            idx = rna_present.nonzero(as_tuple=True)[0]
            loss_rec_rna = F.l1_loss(rna_rec[idx], rna_target[idx], reduction="mean")
        else:
            loss_rec_rna = torch.zeros((), device=device, dtype=loss_rec_img.dtype)

        loss_rec = 0.5 * (loss_rec_img + loss_rec_rna)

        # ---------------- class priors ----------------
        prior_mu_img = self.prior_means_img[labels]        # [B, D]
        prior_lv_img = self.prior_logvars_img[labels]      # [B, D]

        prior_mu_mm = self.prior_means_mm[labels]          # [B, D]
        prior_lv_mm = self.prior_logvars_mm[labels]        # [B, D]

        # only needed where RNA is present
        prior_mu_rna = self.prior_means_rna[labels]        # [B, D]
        prior_lv_rna = self.prior_logvars_rna[labels]      # [B, D]

        # ---------------- KL class terms ----------------
        kl_terms = []

        # unimodal cls_single KLs
        if mu_img_cls_single is not None and logvar_img_cls_single is not None:
            kl_terms.append(
                kl_divergence(mu_img_cls_single, logvar_img_cls_single,
                            prior_mu_img, prior_lv_img).mean()
            )

        if (mu_rna_cls_single is not None) and (logvar_rna_cls_single is not None) and rna_present.any():
            idx = rna_present.nonzero(as_tuple=True)[0]
            kl_terms.append(
                kl_divergence(mu_rna_cls_single[idx], logvar_rna_cls_single[idx],
                            prior_mu_rna[idx], prior_lv_rna[idx]).mean()
            )

        # multimodal fused cls_mm KL (delayed)
        mm_kl_start = self.warmup_start_epoch + self.mm_kl_delay_epochs
        if current_epoch >= mm_kl_start:
            kl_terms.append(
                kl_divergence(mm_mu_cls_mm, mm_logvar_cls_mm,
                            prior_mu_mm, prior_lv_mm).mean()
            )

        # alignment: unimodal mm tokens -> fused mm token
        # (this is the original "KL(q || p)" with p = fused mm distribution)
        if mu_img_cls_mm is not None and logvar_img_cls_mm is not None:
            kl_terms.append(
                kl_divergence(mu_img_cls_mm, logvar_img_cls_mm,
                            mm_mu_cls_mm.detach(), mm_logvar_cls_mm.detach()).mean()
            )

        if (mu_rna_cls_mm is not None) and (logvar_rna_cls_mm is not None) and rna_present.any():
            idx = rna_present.nonzero(as_tuple=True)[0]
            kl_terms.append(
                kl_divergence(mu_rna_cls_mm[idx], logvar_rna_cls_mm[idx],
                            mm_mu_cls_mm.detach()[idx], mm_logvar_cls_mm.detach()[idx]).mean()
            )

        kl_cls = sum(kl_terms) if len(kl_terms) > 0 else torch.zeros((), device=device)

        # ---------------- style KL to N(0, I) ----------------
        # Replace decompose_kl with standard Normal KL.
        # If you already have a 2-arg kl_divergence that does this, use it here.

        kl_style_img = kl_divergence(mu_img_style, logvar_img_style).mean()

        if (mu_rna_style is not None) and (logvar_rna_style is not None) and rna_present.any():
            idx = rna_present.nonzero(as_tuple=True)[0]
            kl_style_rna = kl_divergence(mu_rna_style[idx], logvar_rna_style[idx]).mean()
        else:
            kl_style_rna = torch.zeros((), device=device, dtype=kl_style_img.dtype)

        kl_style = 0.5 * (kl_style_img + kl_style_rna)

        # ---------------- total ----------------
        loss_kl = kl_cls + kl_style

        if warmup:
            loss = loss_rec
        else:
            loss = loss_rec + self.beta * kl_style + self.beta * kl_cls

        return (loss,
                loss_rec, loss_rec_img, loss_rec_rna,
                loss_kl, kl_cls, kl_style,
                kl_style_img, kl_style_rna)

    # ------------- forward --------------
    def forward(
        self,
        img_x=None,
        img_target=None,
        rna_x=None,
        rna_target=None,
        rna_missing=None,  # [B] (1=missing, 0=present)
        labels=None,
        current_epoch: int = 0,
        eval: bool = False,
        warmup: bool = False,
    ):
        """
        Key change vs your version:
        - NEVER returns None for mu/logvar tensors when batch size B is known.
        - Always returns:
            mu_img_cls_single, logvar_img_cls_single
            mu_rna_cls_single, logvar_rna_cls_single   (zeros where RNA missing)
            mm_mu_cls, mm_logvar_cls                   (always defined from image expert at least)
            rna_present                                (bool mask)
        - In eval mode, supports:
            img-only, rna-only, or both.
        In rna-only eval, mm_* come from RNA expert.
        """
        self._update_beta(current_epoch)

        # ---------------- HELPERS ----------------
        def _ensure_img_4d(x):
            if x is None:
                return None
            return x.unsqueeze(1) if x.dim() == 3 else x

        # ---------------- EVAL MODE ----------------
        if eval:
            has_img = img_x is not None
            has_rna = rna_x is not None
            if not (has_img or has_rna):
                raise ValueError("At least one modality must be provided at eval time.")

            img_x = _ensure_img_4d(img_x)

            # determine B/device
            if has_img:
                B = img_x.shape[0]
                device = img_x.device
            else:
                B = rna_x.shape[0]
                device = rna_x.device

            # rna_present mask
            if has_rna:
                if rna_missing is None:
                    rna_present = torch.ones(B, dtype=torch.bool, device=device)
                else:
                    rna_present = ~rna_missing.to(device).bool()
            else:
                rna_present = torch.zeros(B, dtype=torch.bool, device=device)

            # ---------- Encode Image (or allocate zeros) ----------
            if has_img:
                img_enc = self.img_branch.encode(img_x)
                mu_img_cls_single = img_enc["mu_cls_single"]
                logvar_img_cls_single = img_enc["logvar_cls_single"]
                mu_img_cls_mm = img_enc["mu_cls_mm"]
                logvar_img_cls_mm = img_enc["logvar_cls_mm"]
                mu_img_style = img_enc["mu_style"]
                logvar_img_style = img_enc["logvar_style"]
                x4 = img_enc["x4"]
                N = img_enc["batch_size"]
            else:
                # if no image, we will later allocate these based on RNA encoder outputs
                mu_img_cls_single = logvar_img_cls_single = None
                mu_img_cls_mm = logvar_img_cls_mm = None
                mu_img_style = logvar_img_style = None
                x4 = None
                N = None

            # ---------- Encode RNA subset (or allocate zeros if has_img) ----------
            if has_img:
                # always allocate tensors so keys never None
                mu_rna_cls_single = torch.zeros_like(mu_img_cls_single)
                logvar_rna_cls_single = torch.zeros_like(logvar_img_cls_single)
                mu_rna_cls_mm = torch.zeros_like(mu_img_cls_mm)
                logvar_rna_cls_mm = torch.zeros_like(logvar_img_cls_mm)
                mu_rna_style = torch.zeros_like(mu_img_style)
                logvar_rna_style = torch.zeros_like(logvar_img_style)
            else:
                # rna-only eval: will use compact tensors from RNA encoder
                mu_rna_cls_single = logvar_rna_cls_single = None
                mu_rna_cls_mm = logvar_rna_cls_mm = None
                mu_rna_style = logvar_rna_style = None

            if has_rna and rna_present.any():
                idx = rna_present.nonzero(as_tuple=True)[0]
                rna_enc = self.rna_branch.encode(rna_x[idx])

                if has_img:
                    mu_rna_cls_single[idx] = rna_enc["mu_cls_single"]
                    logvar_rna_cls_single[idx] = rna_enc["logvar_cls_single"]
                    mu_rna_cls_mm[idx] = rna_enc["mu_cls_mm"]
                    logvar_rna_cls_mm[idx] = rna_enc["logvar_cls_mm"]
                    mu_rna_style[idx] = rna_enc["mu_style"]
                    logvar_rna_style[idx] = rna_enc["logvar_style"]
                else:
                    # rna-only: keep as-is
                    mu_rna_cls_single = rna_enc["mu_cls_single"]
                    logvar_rna_cls_single = rna_enc["logvar_cls_single"]
                    mu_rna_cls_mm = rna_enc["mu_cls_mm"]
                    logvar_rna_cls_mm = rna_enc["logvar_cls_mm"]
                    mu_rna_style = rna_enc["mu_style"]
                    logvar_rna_style = rna_enc["logvar_style"]

            # ---------- Fuse mm token (deterministic: z=mu) ----------
            if has_img:
                mm_mu_both, mm_lv_both = self.fuse_cls_moe(
                    mu_img_cls_mm=mu_img_cls_mm,
                    logvar_img_cls_mm=logvar_img_cls_mm,
                    mu_rna_cls_mm=mu_rna_cls_mm if has_rna and rna_present.any() else None,
                    logvar_rna_cls_mm=logvar_rna_cls_mm if has_rna and rna_present.any() else None,
                )
                mm_mu_img, mm_lv_img = self.fuse_cls_moe(
                    mu_img_cls_mm=mu_img_cls_mm,
                    logvar_img_cls_mm=logvar_img_cls_mm,
                    mu_rna_cls_mm=None,
                    logvar_rna_cls_mm=None,
                )
                mm_mu_cls = torch.where(rna_present[:, None], mm_mu_both, mm_mu_img)
                mm_logvar_cls = torch.where(rna_present[:, None], mm_lv_both, mm_lv_img)
                z_cls_mm = mm_mu_cls
            else:
                # rna-only: fused is rna expert (only defined where present)
                if not (has_rna and rna_present.any()):
                    raise ValueError("RNA-only eval requires at least one present RNA sample.")
                mm_mu_cls = mu_rna_cls_mm
                mm_logvar_cls = logvar_rna_cls_mm
                z_cls_mm = mm_mu_cls

            # ---------- Decode ----------
            img_rec = None
            rna_rec = None
            z_cls_out = None

            if has_img:
                img_rec = self.img_branch.decode(
                    z_cls_single=mu_img_cls_single,
                    z_cls_mm=z_cls_mm,
                    z_style=mu_img_style,
                    x4=x4,
                    batch_size=N,
                )
                z_cls_out = torch.cat([mu_img_cls_single, z_cls_mm], dim=1)

            if has_rna and rna_present.any():
                idx = rna_present.nonzero(as_tuple=True)[0]
                if has_img:
                    rna_rec_sub = self.rna_branch.decode(
                        z_cls_single=mu_rna_cls_single[idx],
                        z_cls_mm=z_cls_mm[idx],
                        z_style=mu_rna_style[idx],
                    )
                    rna_rec = torch.zeros_like(rna_x)
                    rna_rec[idx] = rna_rec_sub
                else:
                    # rna-only
                    rna_rec_sub = self.rna_branch.decode(
                        z_cls_single=mu_rna_cls_single,
                        z_cls_mm=mu_rna_cls_mm,
                        z_style=mu_rna_style,
                    )
                    rna_rec = torch.zeros_like(rna_x)
                    rna_rec[idx] = rna_rec_sub
                    if z_cls_out is None:
                        z_cls_out = torch.cat([mu_rna_cls_single, mu_rna_cls_mm], dim=1)

            return {
                "img_rec": img_rec,
                "rna_rec": rna_rec,
                "z_cls": z_cls_out,
                "mm_mu_cls": mm_mu_cls,
                "mm_logvar_cls": mm_logvar_cls,
                "rna_present": rna_present,
                "mu_img_cls_single": mu_img_cls_single,
                "logvar_img_cls_single": logvar_img_cls_single,
                "mu_rna_cls_single": mu_rna_cls_single,
                "logvar_rna_cls_single": logvar_rna_cls_single,
            }

        # ---------------- TRAINING MODE ----------------
        # (keeps your training behavior but returns non-None tensors always)
        if img_x is None:
            raise ValueError("Training expects img_x to be present.")

        img_x = _ensure_img_4d(img_x)
        img_target = _ensure_img_4d(img_target)

        B = img_x.shape[0]
        device = img_x.device

        # rna_present from dataset
        if rna_x is None:
            rna_present = torch.zeros(B, dtype=torch.bool, device=device)
        else:
            if rna_missing is None:
                rna_present = torch.ones(B, dtype=torch.bool, device=device)
            else:
                rna_present = ~rna_missing.to(device).bool()

        # per-sample stochastic dropout (only after warmup)
        if (not warmup) and (self.p_drop_rna > 0) and rna_present.any():
            keep = (torch.rand(B, device=device) >= self.p_drop_rna)
            rna_present = rna_present & keep

        # ===== IMAGE ENCODER =====
        img_enc = self.img_branch.encode(img_x)
        mu_img_cls_single = img_enc["mu_cls_single"]
        logvar_img_cls_single = img_enc["logvar_cls_single"]
        mu_img_cls_mm = img_enc["mu_cls_mm"]
        logvar_img_cls_mm = img_enc["logvar_cls_mm"]
        mu_img_style = img_enc["mu_style"]
        logvar_img_style = img_enc["logvar_style"]
        x4 = img_enc["x4"]
        N = img_enc["batch_size"]

        # ===== RNA ENCODER (always allocate, fill subset) =====
        mu_rna_cls_single = torch.zeros_like(mu_img_cls_single)
        logvar_rna_cls_single = torch.zeros_like(logvar_img_cls_single)
        mu_rna_cls_mm = torch.zeros_like(mu_img_cls_mm)
        logvar_rna_cls_mm = torch.zeros_like(logvar_img_cls_mm)
        mu_rna_style = torch.zeros_like(mu_img_style)
        logvar_rna_style = torch.zeros_like(logvar_img_style)

        if (rna_x is not None) and rna_present.any():
            idx = rna_present.nonzero(as_tuple=True)[0]
            rna_enc = self.rna_branch.encode(rna_x[idx])
            mu_rna_cls_single[idx] = rna_enc["mu_cls_single"]
            logvar_rna_cls_single[idx] = rna_enc["logvar_cls_single"]
            mu_rna_cls_mm[idx] = rna_enc["mu_cls_mm"]
            logvar_rna_cls_mm[idx] = rna_enc["logvar_cls_mm"]
            mu_rna_style[idx] = rna_enc["mu_style"]
            logvar_rna_style[idx] = rna_enc["logvar_style"]

        # ===== MoE FUSION per sample =====
        mm_mu_both, mm_lv_both = self.fuse_cls_moe(
            mu_img_cls_mm=mu_img_cls_mm,
            logvar_img_cls_mm=logvar_img_cls_mm,
            mu_rna_cls_mm=mu_rna_cls_mm if rna_present.any() else None,
            logvar_rna_cls_mm=logvar_rna_cls_mm if rna_present.any() else None,
        )
        mm_mu_img, mm_lv_img = self.fuse_cls_moe(
            mu_img_cls_mm=mu_img_cls_mm,
            logvar_img_cls_mm=logvar_img_cls_mm,
            mu_rna_cls_mm=None,
            logvar_rna_cls_mm=None,
        )
        mm_mu_cls_mm = torch.where(rna_present[:, None], mm_mu_both, mm_mu_img)
        mm_logvar_cls_mm = torch.where(rna_present[:, None], mm_lv_both, mm_lv_img)

        # ===== SAMPLE =====
        z_cls_img_single = reparameterize(mu_img_cls_single, logvar_img_cls_single)
        z_cls_mm = reparameterize(mm_mu_cls_mm, mm_logvar_cls_mm)
        z_img_style = reparameterize(mu_img_style, logvar_img_style)

        # ===== DECODE IMAGE =====
        img_rec = self.img_branch.decode(
            z_cls_single=z_cls_img_single,
            z_cls_mm=z_cls_mm,
            z_style=z_img_style,
            x4=x4,
            batch_size=N,
        )

        # ===== DECODE RNA (subset) =====
        if rna_target is not None:
            rna_rec = torch.zeros_like(rna_target)
        else:
            rna_rec = None

        if (rna_target is not None) and (rna_x is not None) and rna_present.any():
            idx = rna_present.nonzero(as_tuple=True)[0]
            z_cls_rna_single_sub = reparameterize(mu_rna_cls_single[idx], logvar_rna_cls_single[idx])
            z_rna_style_sub = reparameterize(mu_rna_style[idx], logvar_rna_style[idx])
            rna_rec_sub = self.rna_branch.decode(
                z_cls_single=z_cls_rna_single_sub,
                z_cls_mm=z_cls_mm[idx],
                z_style=z_rna_style_sub,
            )
            rna_rec[idx] = rna_rec_sub

        # ===== LOSS =====
        (loss,
        loss_rec, loss_rec_img, loss_rec_rna,
        loss_kl, kl_cls, kl_style,
        kl_style_img, kl_style_rna) = self.compute_loss(
            img_rec=img_rec,
            img_target=img_target,
            rna_rec=rna_rec,
            rna_target=rna_target,
            mm_mu_cls_mm=mm_mu_cls_mm,
            mm_logvar_cls_mm=mm_logvar_cls_mm,
            mu_img_style=mu_img_style,
            logvar_img_style=logvar_img_style,
            mu_rna_style=mu_rna_style,
            logvar_rna_style=logvar_rna_style,
            current_epoch=current_epoch,
            mu_img_cls_single=mu_img_cls_single,
            logvar_img_cls_single=logvar_img_cls_single,
            mu_rna_cls_single=mu_rna_cls_single,
            logvar_rna_cls_single=logvar_rna_cls_single,
            mu_img_cls_mm=mu_img_cls_mm,
            logvar_img_cls_mm=logvar_img_cls_mm,
            mu_rna_cls_mm=mu_rna_cls_mm,
            logvar_rna_cls_mm=logvar_rna_cls_mm,
            rna_present=rna_present,
            warmup=warmup,
            labels=labels,
        )

        z_cls_out = torch.cat([z_cls_img_single, z_cls_mm], dim=1)

        return {
            "img_rec": img_rec,
            "rna_rec": rna_rec,
            "z_cls": z_cls_out,
            "mm_mu_cls": mm_mu_cls_mm,
            "mm_logvar_cls": mm_logvar_cls_mm,
            "mu_img_cls_single": mu_img_cls_single,
            "logvar_img_cls_single": logvar_img_cls_single,
            "mu_rna_cls_single": mu_rna_cls_single,
            "logvar_rna_cls_single": logvar_rna_cls_single,
            "loss_total": loss,
            "loss_rec": loss_rec,
            "loss_rec_img": loss_rec_img,
            "loss_rec_rna": loss_rec_rna,
            "loss_kl": loss_kl,
            "loss_kl_cls": kl_cls,
            "loss_kl_style": kl_style,
            "loss_kl_style_img": kl_style_img,
            "loss_kl_style_rna": kl_style_rna,
            "beta": float(self.beta.detach().item()),
            "rna_present": rna_present,
        }


# Public name used by the SYNPRED scripts.
SynpredVAE = MultiModalVAEDisMoEwProjectorsAblation
