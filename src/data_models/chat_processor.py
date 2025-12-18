from src.data_models.message import Message


import base64
import json
from io import BytesIO
from typing import Union, List, Dict, Tuple, Optional, Any
from dataclasses import fields

import librosa
import numpy as np
import pandas as pd
import dacite
import torch
from transformers import AutoTokenizer

from .chat import Chat
from .message_content import TextContent, AudioContent
from .constants import (
    AUDIO_IN_TOKEN,
    AUDIO_OUT_TOKEN,
    AUDIO_BOS,
    AUDIO_EOS,
    AUDIO_OUT_BOS,
    BEGIN_OF_TEXT,
    START_HEADER_ID,
    END_HEADER_ID,
    RECIPIENT,
    EOT_ID,
    EOM_ID,
)
from .model_input import HiggsAudioModelInput
from src.audio_tokenizer.higgs_audio_tokenizer import HiggsAudioTokenizer


class ChatProcessor:
    """
    Converts Chat instances into model-ready tensors for a multimodal (text + audio) model.
    """

    def __init__(
        self,
        text_tokenizer: AutoTokenizer,
        audio_tokenizer: HiggsAudioTokenizer,
        device: Optional[torch.device] = None,
    ) -> None:
        self.text_tokenizer = text_tokenizer
        self.audio_tokenizer = audio_tokenizer
        self.device = device if device is not None else torch.device('cpu')

    def process_chats(self, chats: List[Union[Chat, Dict[str, Any]]]) -> List[HiggsAudioModelInput]:
        """Batch process a list of chats."""
        return [self.process_chat(chat) for chat in chats]

    def process_chat(self, chat: Union[Chat, Dict[str, Any]]) -> HiggsAudioModelInput:
        """Process a single chat into a model-ready input."""
        # 1. Validation and Conversion
        if not isinstance(chat, Chat):
            chat = self._dict_to_chat(chat)
        
            if chat is None:
                raise ValueError("Failed to convert input to Chat object")

        # 2. Tokenize Text & Collect Audio Content
        input_tokens = []
        label_tokens = []
        audio_contents = []
        
        try:
            for turn_id, message in enumerate[Message](chat.messages):
                # A. Role Headers
                role_prefix_tokens = self._tokenize_role_prefix(message.role, turn_id, self.text_tokenizer)
                input_tokens.extend(role_prefix_tokens)
                label_tokens.extend([-100] * len(role_prefix_tokens))

                # B. Recipient Handling
                recipient_tokens = self._tokenize_recipient(message, self.text_tokenizer)
                if recipient_tokens:
                    input_tokens.extend(recipient_tokens)
                    label_tokens.extend(recipient_tokens)

                # C. Content Processing (Text & Audio)
                content_in, content_label, turn_audio = self._process_message_content(
                    message, chat, turn_id
                )
                input_tokens.extend(content_in)
                label_tokens.extend(content_label)
                audio_contents.extend(turn_audio)

                # D. Termination
                termination_tokens = self._tokenize_termination(message, chat, turn_id, self.text_tokenizer)
                input_tokens.extend(termination_tokens)
                # Apply teacher forcing for termination tokens
                start_index = getattr(chat, "start_index", None)
                if message.role == "assistant" and (start_index is None or turn_id >= start_index):
                    label_tokens.extend(termination_tokens)
                else:
                    label_tokens.extend([-100] * len(termination_tokens))

        except Exception as e:
            print(f"Error processing chat messages: {str(e)}")
            raise ValueError("Failed to process chat sample tokens") from e

        if not input_tokens:
             raise ValueError("Chat produced no tokens")

        # 3. Audio Processing
        # Bulk process all collected audio content
        (
            audio_ids_concat,
            audio_ids_start,
            audio_waveforms_concat,
            audio_waveforms_start,
            audio_sample_rate
        ) = self._process_audio_content(audio_contents)

        # 4. Tensor Creation
        input_ids = torch.tensor(input_tokens, dtype=torch.long, device=self.device)
        label_ids = (
            torch.tensor(label_tokens, dtype=torch.long, device=self.device)
            if label_tokens is not None
            else torch.full_like(input_ids, -100)
        )

        # 5. Model Input Assembly
        return HiggsAudioModelInput(
            input_ids=input_ids,
            label_ids=label_ids,
            audio_ids_concat=audio_ids_concat,
            audio_ids_start=audio_ids_start,
            audio_waveforms_concat=audio_waveforms_concat,
            audio_waveforms_start=audio_waveforms_start,
            audio_sample_rate=audio_sample_rate,
            audio_speaker_indices=torch.tensor([], dtype=torch.long, device=self.device),
        )

    def _process_audio_content(
        self,
        audio_contents: List[AudioContent],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Loads, encodes, and concatenates all audio content into model tensors.
        """
        # Create empty tensors inline to avoid shared state issues
        empty_tensor = lambda dtype: torch.tensor([], dtype=dtype, device=self.device)
        
        if not audio_contents or self.audio_tokenizer is None:
            return (
                torch.tensor([[]], dtype=torch.long, device=self.device),
                empty_tensor(torch.long),
                empty_tensor(torch.float32),
                empty_tensor(torch.long),
                empty_tensor(torch.float32)
            )

        audio_ids_list: List[torch.Tensor] = []
        audio_waveforms_list: List[torch.Tensor] = []
        sample_rates: List[float] = []
        target_sr = self.audio_tokenizer.sampling_rate

        for audio_content in audio_contents:
            raw_audio = None
            sr = target_sr

            try:
                if audio_content.audio_url and audio_content.audio_url not in ["placeholder", ""]:
                    raw_audio, sr = librosa.load(audio_content.audio_url, sr=target_sr)
                elif audio_content.raw_audio:
                    decoded = base64.b64decode(audio_content.raw_audio)
                    raw_audio, sr = librosa.load(BytesIO(decoded), sr=target_sr)
            except Exception as e:
                print(f"Failed to load audio content: {e}")
                continue

            if raw_audio is not None:
                audio_waveforms_list.append(torch.tensor(raw_audio, dtype=torch.float32))
                sample_rates.append(float(sr))
                
                # Encode to tokens [Codebooks, Time]
                audio_ids = self.audio_tokenizer.encode(raw_audio, sr)
                audio_ids_list.append(audio_ids.squeeze(0).cpu())

        if not audio_ids_list:
            return (
                torch.tensor([[]], dtype=torch.long, device=self.device),
                empty_tensor(torch.long),
                empty_tensor(torch.float32),
                empty_tensor(torch.long),
                empty_tensor(torch.float32)
            )

        # Calculate Start Indices (Offsets)
        audio_ids_lengths = [x.shape[1] for x in audio_ids_list]
        audio_ids_start = torch.tensor(
            np.cumsum([0] + audio_ids_lengths)[:-1], dtype=torch.long, device=self.device
        )

        waveform_lengths = [len(x) for x in audio_waveforms_list]
        audio_waveforms_start = torch.tensor(
            np.cumsum([0] + waveform_lengths)[:-1], dtype=torch.long, device=self.device
        )

        # Concatenate
        audio_ids_concat = torch.cat(audio_ids_list, dim=1).to(self.device)
        audio_waveforms_concat = torch.cat(audio_waveforms_list, dim=0).to(self.device)
        audio_sample_rate = torch.tensor(sample_rates, dtype=torch.float32, device=self.device)

        return audio_ids_concat, audio_ids_start, audio_waveforms_concat, audio_waveforms_start, audio_sample_rate

    def _dict_to_chat(self, sample: Dict[str, Any]) -> Optional[Chat]:
        """Safely convert a raw dictionary to a Chat object, handling NaNs."""
        def clean_value(obj):
            if isinstance(obj, (pd.Series, np.ndarray)):
                return obj.tolist()
            if pd.api.types.is_scalar(obj) and pd.isna(obj):
                return None
            if isinstance(obj, dict):
                return {k: clean_value(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [clean_value(item) for item in obj]
            return obj

        clean_sample = clean_value(sample)
        if not isinstance(clean_sample, dict):
             return None

        # Defaults
        if "speaker" not in clean_sample: clean_sample["speaker"] = None
        if "content" not in clean_sample: clean_sample["content"] = ""
        
        valid_keys = {f.name for f in fields(Chat)}
        filtered_sample = {k: v for k, v in clean_sample.items() if k in valid_keys}

        try:
            return dacite.from_dict(
                data_class=Chat, 
                data=filtered_sample, 
                config=dacite.Config(strict=False, check_types=True)
            )
        except Exception as e:
            print(f"Failed to convert dict to Chat: {e}")
            return None

    # ---------------------------------------------------------------------
    # Message Processing Helper Methods
    # ---------------------------------------------------------------------
    
    def _tokenize_role_prefix(self, role: str, turn_id: int, tokenizer: AutoTokenizer) -> List[int]:
        """Tokenize role prefix with special tokens (step 3 & 4: Text tokenization & Message formatting)."""
        if turn_id == 0:
            prefix = f"{BEGIN_OF_TEXT}{START_HEADER_ID}{role}{END_HEADER_ID}\n\n"
        else:
            prefix = f"{START_HEADER_ID}{role}{END_HEADER_ID}\n\n"
        return tokenizer.encode(prefix, add_special_tokens=False)

    def _tokenize_recipient(self, message: Message, tokenizer: AutoTokenizer) -> List[int]:
        """Tokenize recipient tokens if present (step 7: Recipient handling)."""
        if message.recipient and message.role == "assistant":
            recipient_text = f"{message.recipient}{RECIPIENT}"
            return tokenizer.encode(recipient_text, add_special_tokens=False)
        return []

    def _process_message_content(
        self, message: Message, chat: Chat, turn_id: int
    ) -> Tuple[List[int], List[int], List[AudioContent]]:
        """Process message content: text and audio with teacher forcing (step 5: Content processing, step 6: Label generation)."""
        input_tokens = []
        label_tokens = []
        audio_contents = []
        
        role = message.role
        content = message.content
        start_index = getattr(chat, "start_index", None)
        
        # Normalize content to list
        content_list = []
        if isinstance(content, str):
            content_list.append(TextContent(text=content))
        elif isinstance(content, TextContent):
            content_list.append(content)
        elif isinstance(content, AudioContent):
            content_list.append(content)
        elif isinstance(content, list):
            for ele in content:
                if isinstance(ele, str):
                    content_list.append(TextContent(text=ele))
                else:
                    content_list.append(ele)
        
        # Process each content item
        for content_item in content_list:
            if content_item.type == "text":
                # Tokenize text content
                text_tokens = self.text_tokenizer.encode(content_item.text, add_special_tokens=False)
                input_tokens.extend(text_tokens)
                
                # Apply teacher forcing with start_index handling
                if role == "assistant" and (start_index is None or turn_id >= start_index):
                    label_tokens.extend(text_tokens)
                else:
                    label_tokens.extend([-100] * len(text_tokens))
            
            elif content_item.type == "audio":
                # Collect audio content for later processing
                audio_contents.append(content_item)
                
                # Tokenize audio placeholder tokens
                if role == "user" or role == "system":
                    audio_tokens = self.text_tokenizer.encode(
                        f"{AUDIO_BOS}{AUDIO_IN_TOKEN}{AUDIO_EOS}", add_special_tokens=False
                    )
                else:  # assistant
                    audio_tokens = self.text_tokenizer.encode(
                        f"{AUDIO_OUT_BOS}{AUDIO_OUT_TOKEN}{AUDIO_EOS}", add_special_tokens=False
                    )
                
                input_tokens.extend(audio_tokens)
                
                # Apply teacher forcing for audio tokens
                if role == "assistant" and (start_index is None or turn_id >= start_index):
                    label_tokens.extend(audio_tokens)
                else:
                    label_tokens.extend([-100] * len(audio_tokens))
        
        return input_tokens, label_tokens, audio_contents

    def _tokenize_termination(
        self, message: Message, chat: Chat, turn_id: int, tokenizer: AutoTokenizer
    ) -> List[int]:
        """Tokenize termination tokens (step 8: Message termination).
        
        Note: The caller should apply teacher forcing to the returned tokens when adding to label_tokens.
        """
        role = message.role
        total_m = len(chat.messages)
        next_id = turn_id + 1
        
        # Determine termination token type
        if role == "assistant" and next_id < total_m and chat.messages[next_id].role == "assistant":
            termination_text = EOM_ID
        else:
            termination_text = EOT_ID
        
        return tokenizer.encode(termination_text, add_special_tokens=False)