"""Utility functions for DAC codec."""

from torch import nn


def init_weights(m):
    """Initialize weights for Conv1d layers."""
    if isinstance(m, nn.Conv1d):
        nn.init.trunc_normal_(m.weight, std=0.02)
        nn.init.constant_(m.bias, 0)

