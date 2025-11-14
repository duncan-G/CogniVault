import torch
from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelInput:
    input_ids: torch.LongTensor
    audio_ids_concat: torch.LongTensor
    audio_ids_start: torch.LongTensor
    label_ids: torch.LongTensor
    audio_waveforms_concat: Optional[torch.Tensor] = None
    audio_waveforms_start: Optional[torch.LongTensor] = None
    audio_sample_rate: Optional[torch.Tensor] = None
    audio_speaker_indices: Optional[torch.LongTensor] = None
    audio_label_ids_concat: Optional[torch.LongTensor] = None
    reward: Optional[float] = None
