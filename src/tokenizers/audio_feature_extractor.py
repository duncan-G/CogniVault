import torch
import numpy as np
from typing import Dict

class AudioFeatureExtractor(torch.nn.Module):
    def __init__(self, sampling_rate: int = 16000):
        super().__init__()
        self.sampling_rate = sampling_rate

    def forward(self, raw_audio: np.ndarray) -> Dict[str, torch.Tensor]:
        # Convert from librosa to torch
        audio_signal = torch.tensor(raw_audio)
        audio_signal = audio_signal.unsqueeze(0)
        if len(audio_signal.shape) < 3:
            audio_signal = audio_signal.unsqueeze(0)
        return {"input_values": audio_signal}