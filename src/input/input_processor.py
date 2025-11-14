import base64
from io import BytesIO
from typing import List, Tuple, Optional

import librosa
import numpy as np
import torch
from transformers import AutoTokenizer

from src.input.chat_history import ChatHistory
from src.input.input_collator import InputCollator
from src.input.message import Message
from src.input.message_content import AudioContent, TextContent
from src.input.model_input import ModelInput
from src.input.model_batch_input import ModelBatchInput
from src.tokenizers.audio_tokenizer import AudioTokenizer

# Special tokens / templates
EOT = "<|eot_id|>"  # End of Turn
EOM = "<|eom_id|>"  # End of Message
BOS = "<|begin_of_text|>"  # Beginning of Sequence
HEADER_TEMPLATE = "<|start_header_id|>{role}<|end_header_id|>\n\n"
RECIPIENT_SUFFIX = "<|recipient|>"
AUDIO_IN = "<|audio_bos|><|AUDIO|><|audio_eos|>"
AUDIO_OUT = "<|audio_out_bos|><|AUDIO_OUT|><|audio_eos|>"


class InputProcessor:
    """
    Converts a ChatHistory instance into model-ready tensors for a multimodal (text + audio) model.

    Responsibilities:
        - Format messages with role headers and special tokens.
        - Build input_ids and label_ids with teacher forcing on assistant outputs.
        - Collect and encode audio into a concatenated token buffer plus offsets.
        - Batch the result via the provided InputCollator.
    """

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        audio_tokenizer: AudioTokenizer,
        collator: InputCollator,
        device: torch.device,
    ) -> None:
        self.tokenizer = tokenizer
        self.audio_tokenizer = audio_tokenizer
        self.collator = collator
        self.device = device

    def process(self, chat: ChatHistory) -> ModelBatchInput:
        """Convert a ChatHistory into a batched ModelBatchInput."""
        input_tokens, label_tokens, audio_contents = self._generate_tokens(chat)

        input_ids = torch.tensor(input_tokens, dtype=torch.long, device=self.device)
        label_ids = torch.tensor(label_tokens, dtype=torch.long, device=self.device)

        audio_ids_concat, audio_ids_start = self._generate_audio_ids(audio_contents)

        model_input = ModelInput(
            input_ids=input_ids,
            label_ids=label_ids,
            audio_ids_concat=audio_ids_concat,
            audio_ids_start=audio_ids_start,
        )
        # Collator expects a list of single-example inputs.
        return self.collator([model_input])

    # ---------------------------------------------------------------------
    # Token stream construction
    # ---------------------------------------------------------------------

    def _generate_tokens(
        self,
        chat: ChatHistory,
    ) -> Tuple[List[int], List[int], List[AudioContent]]:
        """
        Build parallel input and label token sequences and collect audio contents.

        - User/system content and structural tokens are context-only (labels = -100).
        - Assistant content and EOM tokens are supervised (labels = tokens).
        """
        input_tokens: List[int] = []
        label_tokens: List[int] = []
        audio_contents: List[AudioContent] = []

        total_messages = len(chat.messages)

        for idx, message in enumerate[Message](chat.messages):
            # Prefix: BOS (for first message) + role header (always masked)
            prefix_text = (BOS if idx == 0 else "") + HEADER_TEMPLATE.format(role=message.role)
            self._append_segment(prefix_text, teach=False, dst_inputs=input_tokens, dst_labels=label_tokens)

            # Optional recipient (assistant only; supervised)
            recipient = getattr(message, "recipient", None)
            if recipient is not None:
                if message.role != "assistant":
                    raise ValueError("Recipient is only valid for assistant messages.")
                rec_text = f"{recipient}{RECIPIENT_SUFFIX}"
                self._append_segment(rec_text, teach=True, dst_inputs=input_tokens, dst_labels=label_tokens)

            # Main content
            self._append_content(
                message=message,
                dst_inputs=input_tokens,
                dst_labels=label_tokens,
                audio_contents=audio_contents,
            )

            # Postfix: EOM for final/isolated assistant turn (supervised), else EOT (masked)
            is_assistant = message.role == "assistant"
            is_last_or_next_not_assistant = (
                idx == total_messages - 1
                or chat.messages[idx + 1].role != "assistant"
            )

            if is_assistant and is_last_or_next_not_assistant:
                # Train the model to produce EOM at the end of assistant turns.
                self._append_segment(EOM, teach=True, dst_inputs=input_tokens, dst_labels=label_tokens)
            else:
                # Non-terminal assistant turns or non-assistant turns just end with EOT (masked).
                self._append_segment(EOT, teach=False, dst_inputs=input_tokens, dst_labels=label_tokens)

        return input_tokens, label_tokens, audio_contents

    def _append_content(
        self,
        message: Message,
        dst_inputs: List[int],
        dst_labels: List[int],
        audio_contents: List[AudioContent],
    ) -> None:
        """Append content-specific tokens for a single message."""
        content = message.content

        if isinstance(content, AudioContent):
            audio_contents.append(content)

            if message.role in {"user", "system"}:
                # Audio from user/system is context only.
                self._append_segment(
                    AUDIO_IN,
                    teach=False,
                    dst_inputs=dst_inputs,
                    dst_labels=dst_labels,
                )
            elif message.role == "assistant":
                # Audio from assistant is a supervised target.
                self._append_segment(
                    AUDIO_OUT,
                    teach=True,
                    dst_inputs=dst_inputs,
                    dst_labels=dst_labels,
                )
            else:
                raise ValueError(f"Unsupported role for audio content: {message.role}")

        elif isinstance(content, TextContent):
            self._append_segment(
                content.text,
                teach=True,
                dst_inputs=dst_inputs,
                dst_labels=dst_labels,
            )
        else:
            raise TypeError(f"Unsupported content type: {type(content)}")

    # ---------------------------------------------------------------------
    # Audio encoding
    # ---------------------------------------------------------------------

    def _generate_audio_ids(
        self,
        audio_contents: List[AudioContent],
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Encode each AudioContent into token IDs.

        Returns:
            audio_ids_concat: [1, sum(T_i)] concatenated audio token tensor, or None if no audio.
            audio_ids_start:  [N+1] tensor of cumulative start indices, or None if no audio.
                              For N clips, audio_ids_start[k] is the start index of clip k.
        """
        if not audio_contents:
            return None, None

        audio_ids_list: List[torch.Tensor] = []

        for audio_content in audio_contents:
            raw_audio = self._load_raw_audio(audio_content)
            if raw_audio is None:
                continue

            # Expect encoder to return shape [1, T]
            audio_ids: torch.Tensor = self.audio_tokenizer.encode(
                raw_audio,
                self.audio_tokenizer.sampling_rate,
            )
            audio_ids_list.append(audio_ids.cpu())

        if not audio_ids_list:
            return None, None

        lengths = [a.shape[1] for a in audio_ids_list]
        starts_np = np.cumsum([0] + lengths)
        audio_ids_start = torch.tensor(starts_np, dtype=torch.long, device=self.device)

        audio_ids_concat = torch.cat(audio_ids_list, dim=1).to(self.device)

        return audio_ids_concat, audio_ids_start

    def _load_raw_audio(self, audio_content: AudioContent) -> Optional[np.ndarray]:
        """Load raw audio samples from either a URL/path or base64-encoded bytes."""
        # Prefer an explicit URL/path if provided.
        if getattr(audio_content, "audio_url", "") not in ("", "placeholder"):
            return librosa.load(
                audio_content.audio_url,
                sr=self.audio_tokenizer.sampling_rate,
            )[0]

        # Fall back to base64-encoded raw audio bytes.
        if audio_content.raw_audio is not None:
            decoded = base64.b64decode(audio_content.raw_audio)
            return librosa.load(
                BytesIO(decoded),
                sr=self.audio_tokenizer.sampling_rate,
            )[0]

        return None

    # ---------------------------------------------------------------------
    # Low-level helpers
    # ---------------------------------------------------------------------

    def _encode(self, text: str) -> List[int]:
        """Tokenize a text segment into token IDs without adding extra special tokens."""
        return self.tokenizer.encode(text, add_special_tokens=False)

    def _append_segment(
        self,
        text: str,
        teach: bool,
        dst_inputs: List[int],
        dst_labels: List[int],
    ) -> None:
        """Encode text and append tokens to input/label sequences."""
        tokens = self._encode(text)
        dst_inputs.extend(tokens)

        if teach:
            dst_labels.extend(tokens)
        else:
            # Mask out context-only tokens with -100 so the loss ignores them.
            dst_labels.extend([-100] * len(tokens))
