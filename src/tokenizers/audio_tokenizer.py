import math
import torch
import torch.nn.functional as F
import librosa
import torchaudio
import numpy as np
from typing import Sequence, Union, Optional
from transformers import AutoModel
from vector_quantize_pytorch import ResidualFSQ

import src.codec.dac as dac2
from src.tokenizers.encoded_result import EncodedResult
from src.codec.xcodec import Encoder, Decoder
from src.quantization.vq import ResidualVectorQuantizer
from src.tokenizers.audio_feature_extractor import AudioFeatureExtractor

class AudioTokenizer(torch.nn.Module):
    """
    Audio tokenizer that converts audio waveforms into token IDs.
    Args:
        n_filters: Number of filters for the encoder/decoder.
            Used in Descript Audio Codec (DAC).
        D: Dimension of the encoder output
        target_bandwidths: Target bitrates/bandwidths (kbps) for variable rate encoding.
            Used by the quantizer to select codebook layers
        ratios: Downsampling ratios for the encoder/decoder.
            Used in Descript Audio Codec (DAC).
        sample_rate: Sampling rate of the audio.
        bins: Number of entries per codebook
        n_q: Number of quantizer layers. More layers increase capacity and bitrate.
        codebook_dim: Dimension of the codebook vector. If None, uses quantizer_dim.
        normalize: Whether to normalize the audio.
        causal: Whether to use causal autoregressive processing.
        semantic_techer: The semantic techer to use.
        last_layer_semantic: Whether to use the last layer of the semantic model.
        merge_mode: The mode to merge the semantic and acoustic features.
        downsample_mode: The mode to downsample the semantic features.
        semantic_mode: The mode to use the semantic features.
        vq_scale: The scale to use for the quantizer.
        semantic_sample_rate: The sampling rate of the semantic model.
        device: The device to use.
    """
    def __init__(
        self,
        n_filters: int = 32,
        D: int = 128,
        target_bandwidths: Sequence[Union[int, float]] = [1, 1.5, 2, 4, 6],
        ratios: Sequence[int] = [8, 5, 4, 2],  #  downsampling by 320
        sample_rate: int = 16000,
        bins: int = 1024,
        n_q: int = 8,
        codebook_dim: int = None,
        normalize: bool = False,
        causal: bool = False,
        semantic_techer: str = "hubert_base_general",
        last_layer_semantic: bool = True,
        merge_mode: str = "concat",
        downsample_mode: str = "step_down",
        semantic_mode: str = "classic",
        vq_scale: int = 1,
        semantic_sample_rate: int = None,
        device: str = "cuda",
    ):
        super().__init__()
        self.hop_length = np.prod(ratios)
        self.semantic_techer = semantic_techer

        self.frame_rate = math.ceil(sample_rate / np.prod(ratios))  # 50 Hz

        self.target_bandwidths = target_bandwidths
        self.n_q = n_q
        self.sample_rate = sample_rate
        self.encoder = dac2.Encoder(64, ratios, D)

        self.decoder_2 = dac2.Decoder(D, 1024, ratios)
        self.last_layer_semantic = last_layer_semantic
        self.device = device
        if semantic_techer == "hubert_base":
            self.semantic_model = AutoModel.from_pretrained("facebook/hubert-base-ls960", cache_dir="./.models")
            self.semantic_sample_rate = 16000
            self.semantic_dim = 768
            self.encoder_semantic_dim = 768

        elif semantic_techer == "wavlm_base_plus":
            self.semantic_model = AutoModel.from_pretrained("microsoft/wavlm-base-plus", cache_dir="./.models")
            self.semantic_sample_rate = 16000
            self.semantic_dim = 768
            self.encoder_semantic_dim = 768

        elif semantic_techer == "hubert_base_general":
            self.semantic_model = AutoModel.from_pretrained("bosonai/hubert_base", trust_remote_code=True, cache_dir="./models")
            self.semantic_sample_rate = 16000
            self.semantic_dim = 768
            self.encoder_semantic_dim = 768

        # Overwrite semantic model sr to ensure semantic_downsample_factor is an integer
        if semantic_sample_rate is not None:
            self.semantic_sample_rate = semantic_sample_rate

        self.semantic_model.eval()

        # make the semantic model parameters do not need gradient
        for param in self.semantic_model.parameters():
            param.requires_grad = False

        self.semantic_downsample_factor = int(self.hop_length / (self.sample_rate / self.semantic_sample_rate) / 320)

        self.quantizer_dim = int((D + self.encoder_semantic_dim) // vq_scale)
        self.encoder_semantic = Encoder(input_channels=self.semantic_dim, encode_channels=self.encoder_semantic_dim)
        self.decoder_semantic = Decoder(
            code_dim=self.encoder_semantic_dim, output_channels=self.semantic_dim, decode_channels=self.semantic_dim
        )

        # out_D=D+768
        if isinstance(bins, int):  # RVQ
            self.quantizer = ResidualVectorQuantizer(
                dimension=self.quantizer_dim, codebook_dim=codebook_dim, n_q=n_q, bins=bins
            )
            self.quantizer_type = "RVQ"
        else:  # RFSQ
            self.quantizer = ResidualFSQ(dim=self.quantizer_dim, levels=bins, num_quantizers=n_q)
            self.quantizer_type = "RFSQ"

        self.fc_prior = torch.nn.Linear(D + self.encoder_semantic_dim, self.quantizer_dim)
        self.fc_post1 = torch.nn.Linear(self.quantizer_dim, self.encoder_semantic_dim)
        self.fc_post2 = torch.nn.Linear(self.quantizer_dim, D)

        self.downsample_mode = downsample_mode
        if downsample_mode == "avg":
            self.semantic_pooling = torch.nn.AvgPool1d(
                kernel_size=self.semantic_downsample_factor, stride=self.semantic_downsample_factor
            )

        self.audio_tokenizer_feature_extractor = AudioFeatureExtractor(sampling_rate=self.sample_rate)

    @property
    def tps(self):
        return self.frame_rate

    @property
    def sampling_rate(self):
        return self.sample_rate

    @property
    def num_codebooks(self):
        return self.n_q

    @property
    def codebook_size(self):
        return self.quantizer_dim

    def get_last_layer(self):
        return self.decoder.layers[-1].weight

    def calculate_rec_loss(self, rec, target):
        target = target / target.norm(dim=-1, keepdim=True)
        rec = rec / rec.norm(dim=-1, keepdim=True)
        rec_loss = (1 - (target * rec).sum(-1)).mean()

        return rec_loss

    @torch.no_grad()
    def get_regress_target(self, x):
        x = torchaudio.functional.resample(x, self.sample_rate, self.semantic_sample_rate)

        if (
            self.semantic_techer == "hubert_base"
            or self.semantic_techer == "hubert_base_general"
            or self.semantic_techer == "wavlm_base_plus"
        ):
            x = x[:, 0, :]
            x = F.pad(x, (160, 160))
            target = self.semantic_model(x, output_hidden_states=True).hidden_states
            target = torch.stack(target, dim=1)  # .transpose(-1, -2)#.flatten(start_dim=1, end_dim=2)

            # average for all layers
            target = target.mean(1)
            # target = target[9]
            # if self.hop_length > 320:
            #     target = self.semantic_pooling(target.transpose(1, 2)).transpose(1, 2)

        elif self.semantic_techer == "w2v_bert2":
            target = self.semantic_model(x)

        elif self.semantic_techer.startswith("whisper"):
            if self.last_layer_semantic:
                target = self.semantic_model(x, avg_layers=False)
            else:
                target = self.semantic_model(x, avg_layers=True)

        elif self.semantic_techer.startswith("mert_music"):
            if self.last_layer_semantic:
                target = self.semantic_model(x, avg_layers=False)
            else:
                target = self.semantic_model(x, avg_layers=True)

        elif self.semantic_techer.startswith("qwen_audio_omni"):
            target = self.semantic_model(x)

        if self.downsample_mode == "step_down":
            if self.semantic_downsample_factor > 1:
                target = target[:, :: self.semantic_downsample_factor, :]

        elif self.downsample_mode == "avg":
            target = self.semantic_pooling(target.transpose(1, 2)).transpose(1, 2)
        return target

    def forward(self, x: torch.Tensor, bw: int):
        e_semantic_input = self.get_regress_target(x).detach()

        e_semantic = self.encoder_semantic(e_semantic_input.transpose(1, 2))
        e_acoustic = self.encoder(x)

        e = torch.cat([e_acoustic, e_semantic], dim=1)

        e = self.fc_prior(e.transpose(1, 2))

        if self.quantizer_type == "RVQ":
            e = e.transpose(1, 2)
            quantized, codes, bandwidth, commit_loss = self.quantizer(e, self.frame_rate, bw)
            quantized = quantized.transpose(1, 2)
        else:
            quantized, codes = self.quantizer(e)
            commit_loss = torch.tensor(0.0)

        quantized_semantic = self.fc_post1(quantized).transpose(1, 2)
        quantized_acoustic = self.fc_post2(quantized).transpose(1, 2)

        o = self.decoder_2(quantized_acoustic)

        o_semantic = self.decoder_semantic(quantized_semantic)
        semantic_recon_loss = F.mse_loss(e_semantic_input.transpose(1, 2).detach(), o_semantic)

        return o, commit_loss, semantic_recon_loss, None

    def encode(self, audio_path_or_wv, sr=None, loudness_normalize=False, loudness_threshold=-23.0):
        if isinstance(audio_path_or_wv, str):
            wv, sr = librosa.load(audio_path_or_wv, mono=True, sr=None)
        else:
            wv = audio_path_or_wv
            assert sr is not None
        if loudness_normalize:
            import pyloudnorm as pyln

            meter = pyln.Meter(sr)
            l = meter.integrated_loudness(wv)
            wv = pyln.normalize.loudness(wv, l, loudness_threshold)
        if sr != self.sampling_rate:
            wv = librosa.resample(wv, orig_sr=sr, target_sr=self.sampling_rate)
        if self.audio_tokenizer_feature_extractor is not None:
            inputs = self.audio_tokenizer_feature_extractor(
                raw_audio=wv, sampling_rate=self.audio_tokenizer_feature_extractor.sampling_rate, return_tensors="pt"
            )
            input_values = inputs["input_values"].to(self.device)
        else:
            input_values = torch.from_numpy(wv).float().unsqueeze(0)
        with torch.no_grad():
            encoder_outputs = self._xcodec_encode(input_values)
            vq_code = encoder_outputs.audio_codes[0]
        return vq_code

    def _xcodec_encode(self, x: torch.Tensor, target_bw: Optional[int] = None) -> torch.Tensor:
        bw = target_bw

        e_semantic_input = self.get_regress_target(x).detach()

        e_semantic = self.encoder_semantic(e_semantic_input.transpose(1, 2))
        e_acoustic = self.encoder(x)

        if e_acoustic.shape[2] != e_semantic.shape[2]:
            pad_size = 160 * self.semantic_downsample_factor
            e_acoustic = self.encoder(F.pad(x[:, 0, :], (pad_size, pad_size)).unsqueeze(0))

        if e_acoustic.shape[2] != e_semantic.shape[2]:
            if e_acoustic.shape[2] > e_semantic.shape[2]:
                e_acoustic = e_acoustic[:, :, : e_semantic.shape[2]]
            else:
                e_semantic = e_semantic[:, :, : e_acoustic.shape[2]]

        e = torch.cat([e_acoustic, e_semantic], dim=1)

        e = self.fc_prior(e.transpose(1, 2))

        if self.quantizer_type == "RVQ":
            e = e.transpose(1, 2)
            quantized, codes, bandwidth, commit_loss = self.quantizer(e, self.frame_rate, bw)
            codes = codes.permute(1, 0, 2)
        else:
            quantized, codes = self.quantizer(e)
            codes = codes.permute(0, 2, 1)

        # return codes
        return EncodedResult(codes)

    def decode(self, vq_code: torch.Tensor) -> torch.Tensor:
        vq_code = vq_code.to(self.device)

        if self.quantizer_type == "RVQ":
            vq_code = vq_code.permute(1, 0, 2)
            quantized = self.quantizer.decode(vq_code)
            quantized = quantized.transpose(1, 2)
        else:
            vq_code = vq_code.permute(0, 2, 1)
            quantized = self.quantizer.get_output_from_indices(vq_code)
        quantized_acoustic = self.fc_post2(quantized).transpose(1, 2)

        o = self.decoder_2(quantized_acoustic)
        return o.detach().cpu().numpy()

