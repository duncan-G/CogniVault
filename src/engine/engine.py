import torch
import os
import json
import numpy as np
from transformers import WhisperProcessor, AutoTokenizer
from huggingface_hub import snapshot_download
from transformers.cache_utils import StaticCache
from copy import deepcopy
from dataclasses import asdict

from src.input.chat_history import ChatHistory
from src.input.input_processor import InputProcessor
from typing import Optional, List
from transformers import AutoTokenizer
from src.tokenizers.audio_tokenizer import AudioTokenizer
from src.input.input_collator import InputCollator
from src.engine.audio_response import AudioResponse
from src.audio_model.model import HiggsAudioModel

def _revert_delay_pattern(data):
    """Convert samples encoded with delay pattern back to the original form.

    Args:
        data (:obj:`torch.Tensor`):
            The data with delay pattern applied. It will have shape (num_codebooks, seq_len + num_codebooks - 1).

    Returns:
        ret (:obj:`torch.Tensor`):
            Recovered data with delay pattern removed. It will have shape (num_codebooks, seq_len).
    """
    assert len(data.shape) == 2
    out_l = []
    num_codebooks = data.shape[0]
    for i in range(num_codebooks):
        out_l.append(data[i : (i + 1), i : (data.shape[1] - num_codebooks + 1 + i)])
    return torch.cat(out_l, dim=0)

class AudioEngine:
    def __init__(
        self,
        kv_cache_lengths: List[int] = [1024, 4096, 8192]
    ):
        self.torch_dtype = torch.float16
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = self._create_tokenizer()
        self.model = self._load_model()
        self.audio_tokenizer = self._create_audio_tokenizer()
        self.audio_codebook_size = self.model.config.audio_codebook_size
        self.input_processor = self._create_input_processor()

        # Store cache config for lazy creation
        self.cache_config = deepcopy(self.model.config.text_config)
        self.cache_config.num_hidden_layers = self.model.config.text_config.num_hidden_layers
        if self.model.config.audio_dual_ffn_layers is not None:
            self.cache_config.num_hidden_layers += len(self.model.config.audio_dual_ffn_layers)
        self.kv_cache_lengths = sorted(kv_cache_lengths)
        # Create KV caches lazily to save memory - only create when needed
        self.kv_caches = {}

    def generate(
        self,
        chat: ChatHistory,
        max_new_tokens: int,
        temperature: float = 0.7,
        top_k: Optional[int] = None,
        top_p: float = 0.95,
        force_audio_gen: bool = False,
        ras_win_len: Optional[int] = 7,
        ras_win_max_num_repeat: int = 2,
        seed: Optional[int] = None,
    ):
        """
        Generate audio from a chatML sample.
        Args:
            chat: A chat history.
            max_new_tokens: The maximum number of new tokens to generate.
            temperature: The temperature to use for the generation.
            top_k: The top k to use for the generation.
            top_p: The top p to use for the generation.
            stop_strings: A list of strings to stop the generation.
            force_audio_gen: Whether to force audio generation. This ensures the model generates audio tokens rather than text tokens.
            ras_win_len: The length of the RAS window. We use 7 by default. You can disable it by setting it to None or <=0.
            ras_win_max_num_repeat: The maximum number of times to repeat the RAS window.
            seed: The seed to use for the generation.
        Returns:
            A dictionary with the following keys:
                audio: The generated audio.
                sampling_rate: The sampling rate of the generated audio.
        """
        
        with torch.no_grad():
            inputs = self.input_processor.process(chat)
            prompt_token_ids = inputs.input_ids[0].cpu().numpy()

            self._prepare_kv_caches()

            outputs = self.model.generate(
                **asdict(inputs),
                max_new_tokens=max_new_tokens,
                use_cache=True,
                stop_strings=["<|end_of_text|>", "<|eot_id|>"],
                tokenizer=self.tokenizer,
                do_sample=False if temperature == 0.0 else True,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                past_key_values_buckets=self.kv_caches,
                ras_win_len=ras_win_len,
                ras_win_max_num_repeat=ras_win_max_num_repeat,
                seed=seed,
            )
            
            # Clean up after generation to prevent memory leaks
            self._cleanup_after_generation()

            if len(outputs[1]) > 0:
                wv_list = []
                for output_audio in outputs[1]:
                    vq_code = _revert_delay_pattern(output_audio).clip(0, self.audio_codebook_size - 1)[:, 1:-1]
                    wv_numpy = self.audio_tokenizer.decode(vq_code.unsqueeze(0))[0, 0]
                    wv_list.append(wv_numpy)
                wv_numpy = np.concatenate(wv_list)
            else:
                wv_numpy = None

            # We only support one request at a time now
            generated_text_tokens = outputs[0][0].cpu().numpy()[len(prompt_token_ids) :]
            generated_text = self.tokenizer.decode(generated_text_tokens)
            generated_audio_tokens = outputs[1][0].cpu().numpy()
            return AudioResponse(
                audio=wv_numpy,
                generated_audio_tokens=generated_audio_tokens,
                sampling_rate=self.audio_tokenizer.sampling_rate,
                generated_text=generated_text,
                generated_text_tokens=generated_text_tokens,
                usage={
                    "prompt_tokens": prompt_token_ids.shape[0],
                    "completion_tokens": generated_text_tokens.shape[0] + generated_audio_tokens.shape[1],
                    "total_tokens": (
                        prompt_token_ids.shape[0] + generated_text_tokens.shape[0] + generated_audio_tokens.shape[1]
                    ),
                    "cached_tokens": 0,
                },
            )
 
            
    def _load_model(self):
        return HiggsAudioModel.from_pretrained(
            "bosonai/higgs-audio-v2-generation-3B-base",
            torch_dtype=self.torch_dtype,
            cache_dir="./.models"
        ).to(self.device)

    def _prepare_kv_caches(self):
        """Prepare KV caches, creating them lazily if they don't exist."""
        for length in self.kv_cache_lengths:
            if length not in self.kv_caches:
                # Create cache lazily when first needed
                self.kv_caches[length] = StaticCache(
                    config=self.cache_config,
                    max_batch_size=1,
                    max_cache_len=length,
                    device=self.model.device,
                    dtype=self.model.dtype,
                )
            else:
                self.kv_caches[length].reset()
    
    def _cleanup_after_generation(self):
        """Clean up model state and CUDA graphs after generation to prevent memory leaks."""
        # Reset model's past_key_values_bucket tracking
        if hasattr(self.model, 'current_past_key_values_bucket'):
            self.model.current_past_key_values_bucket = None
        
        # Clear CUDA graph runners to free memory
        # These accumulate during generation and can consume significant memory
        if hasattr(self.model, 'decode_graph_runners'):
            self.model.decode_graph_runners.clear()
        
        # Optionally: Clear KV caches to free memory (uncomment if memory is tight)
        # Note: They will be recreated on next generation, which adds a small overhead
        # self.kv_caches.clear()
        
        # Force garbage collection and clear CUDA cache
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _create_tokenizer(self):
        return AutoTokenizer.from_pretrained("bosonai/higgs-audio-v2-generation-3B-base", cache_dir="./.models")

    def _create_input_processor(self):
        return InputProcessor(
            tokenizer=self.tokenizer,
            audio_tokenizer=self.audio_tokenizer,
            collator=self._create_input_collator(),
            device=self.device
        )

    def _create_audio_tokenizer(self)->AudioTokenizer:
        tokenizer_name_or_path = "bosonai/higgs-audio-v2-tokenizer"
        is_local = os.path.exists(tokenizer_name_or_path)
        if not is_local:
            tokenizer_path = snapshot_download(tokenizer_name_or_path)
        else:
            tokenizer_path = tokenizer_name_or_path
        config_path = os.path.join(tokenizer_path, "config.json")
        model_path = os.path.join(tokenizer_path, "model.pth")
        config = json.load(open(config_path))
        model = AudioTokenizer(
            **config,
            device=self.device,
        )
        parameter_dict = torch.load(model_path, map_location=self.device)
        model.load_state_dict(parameter_dict, strict=False)
        model.to(self.device)
        model.eval()
        return model

    def _create_input_collator(self)->InputCollator:
        return InputCollator(
            whisper_processor=WhisperProcessor.from_pretrained("openai/whisper-large-v3-turbo", cache_dir="./.models"),
            audio_in_token_id=100,
            audio_out_token_id=101,
            pad_token_id=100,
            audio_stream_bos_id=100,
            audio_stream_eos_id=101,
            round_to=8,
            pad_left=False,
            encode_whisper_embed=True,
            return_audio_in_tokens=True,
            audio_num_codebooks=None,
            use_delay_pattern=False,
        )