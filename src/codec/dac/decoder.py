"""Decoder components for DAC codec."""

import math

from torch import nn

from libs.audiocodec.nn.layers import Snake1d, WNConv1d, WNConvTranspose1d

from .encoder import ResidualUnit


class DecoderBlock(nn.Module):
    """Decoder block with upsampling and multiple residual units."""

    def __init__(self, input_dim: int = 16, output_dim: int = 8, stride: int = 1, out_pad=0):
        super().__init__()
        self.block = nn.Sequential(
            Snake1d(input_dim),
            WNConvTranspose1d(
                input_dim,
                output_dim,
                kernel_size=2 * stride,
                stride=stride,
                padding=math.ceil(stride / 2),
                output_padding=stride % 2,  # out_pad,
            ),
            ResidualUnit(output_dim, dilation=1),
            ResidualUnit(output_dim, dilation=3),
            ResidualUnit(output_dim, dilation=9),
        )

    def forward(self, x):
        return self.block(x)


class Decoder(nn.Module):
    """Main decoder network for DAC codec."""

    def __init__(
        self,
        input_channel,
        channels,
        rates,
        d_out: int = 1,
    ):
        super().__init__()

        # Add first conv layer
        layers = [WNConv1d(input_channel, channels, kernel_size=7, padding=3)]

        # Add upsampling + MRF blocks
        for i, stride in enumerate(rates):
            input_dim = channels // 2**i
            output_dim = channels // 2 ** (i + 1)
            if i == 1:
                out_pad = 1
            else:
                out_pad = 0
            layers += [DecoderBlock(input_dim, output_dim, stride, out_pad)]

        # Add final conv layer
        layers += [
            Snake1d(output_dim),
            WNConv1d(output_dim, d_out, kernel_size=7, padding=3),
            # nn.Tanh(),
        ]

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

