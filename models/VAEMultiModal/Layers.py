import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.LeakyReLU(0.1, inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.maxpool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x):
        return self.conv(self.maxpool(x))



class Up(nn.Module):
    """
    Upscaling: ConvTranspose2d(2x) + (optional) skip connection + DoubleConv.

    Usage:
        x = Up(c5, c4)(x5, x4)         # with skip
        x = Up(c5, c4)(x5)             # without skip

    If skip is not provided, DoubleConv is applied only on upsampled feature.
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        # 2x transposed convolution upsamples in_channels → out_channels
        self.up = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2,
        )

        # DoubleConv for:
        #   - with skip:  (out_channels * 2) → out_channels
        #   - without skip: out_channels → out_channels
        self.conv_with_skip = DoubleConv(out_channels * 2, out_channels)
        self.conv_no_skip   = DoubleConv(out_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor = None) -> torch.Tensor:
        x = self.up(x)

        if skip is not None:
            # align shapes (in case of odd dimensions)
            diff_y = skip.size(-2) - x.size(-2)
            diff_x = skip.size(-1) - x.size(-1)

            if diff_y != 0 or diff_x != 0:
                x = F.pad(
                    x,
                    [
                        diff_x // 2,
                        diff_x - diff_x // 2,
                        diff_y // 2,
                        diff_y - diff_y // 2,
                    ]
                )

            x = torch.cat([skip, x], dim=1)
            return self.conv_with_skip(x)

        # No skip given → simpler block
        return self.conv_no_skip(x)


