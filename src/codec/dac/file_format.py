"""File format handling for DAC codec."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

SUPPORTED_VERSIONS = ["1.0.0"]


@dataclass
class DACFile:
    """Container for DAC compressed audio file data and metadata."""

    codes: torch.Tensor

    # Metadata
    chunk_length: int
    original_length: int
    input_db: float
    channels: int
    sample_rate: int
    padding: bool
    dac_version: str

    def save(self, path):
        """Save DAC file to disk."""
        artifacts = {
            "codes": self.codes.numpy().astype(np.uint16),
            "metadata": {
                "input_db": self.input_db.numpy().astype(np.float32),
                "original_length": self.original_length,
                "sample_rate": self.sample_rate,
                "chunk_length": self.chunk_length,
                "channels": self.channels,
                "padding": self.padding,
                "dac_version": SUPPORTED_VERSIONS[-1],
            },
        }
        path = Path(path).with_suffix(".dac")
        with open(path, "wb") as f:
            np.save(f, artifacts)
        return path

    @classmethod
    def load(cls, path):
        """Load DAC file from disk."""
        artifacts = np.load(path, allow_pickle=True)[()]
        codes = torch.from_numpy(artifacts["codes"].astype(int))
        if artifacts["metadata"].get("dac_version", None) not in SUPPORTED_VERSIONS:
            raise RuntimeError(f"Given file {path} can't be loaded with this version of descript-audio-codec.")
        return cls(codes=codes, **artifacts["metadata"])

