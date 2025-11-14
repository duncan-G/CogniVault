import torch
import numpy as np
from dataclasses import dataclass
from typing import Optional, List, Tuple, Union

# Whisper processor, 30 sec -> 3000 features
# Then we divide 4 in the audio tokenizer, we decrease 3000 features to 750, which gives 25 Hz
WHISPER_EMBED_NUM_HIDDEN_STATE_PER_SEC = 25


@dataclass
class ModelBatchInput:
    input_ids: torch.LongTensor  # shape (bsz, seq_len).
    attention_mask: torch.Tensor  # shape (bsz, seq_len).
    audio_features: Optional[torch.Tensor] = None  # shape (num_audio_in, feature_dim, max_mel_seq_len).
    audio_feature_attention_mask: Optional[torch.Tensor] = None  # shape (num_audio_in, max_mel_seq_len).
    audio_out_ids: Optional[torch.LongTensor] = None  # shape (num_codebooks, audio_out_total_length)
    audio_out_ids_start: Optional[torch.LongTensor] = None  # shape (num_audio_out,)
    # The audio_out_ids_start_group_loc has the same length as audio_out_ids_start. It is used to recover group location in a batch for an audio segment
    # Currently, we concatenante audio segments along dim 0 to handle variadic audio segment length. However, in the alignment stage, we need the location information
    # For example,
    #  audio_out_ids_start = [0, 2, 4, 8]; and the first two audio segments come from the same sample in a batch, and other two come from different samples.
    #  This is a batch of 3 samples, then we will have the group location as:
    #  audio_out_ids_start_group_loc = [0, 0, 1, 2]
    audio_out_ids_start_group_loc: Optional[
        torch.LongTensor
    ] = None  # shape (num_audio_out,), specify which a sample's group location in the batch
    audio_in_ids: Optional[torch.LongTensor] = None  # shape (num_codebooks, audio_in_total_length)
    audio_in_ids_start: Optional[torch.LongTensor] = None  # shape (num_audio_in,)
    label_ids: Optional[torch.LongTensor] = None  # shape (bsz, seq_len)
    label_audio_ids: Optional[torch.LongTensor] = None  # shape (num_codebooks, audio_out_total_length)
    reward: Optional[float] = None

    def num_audios(self, audio_type: str = "both"):
        """Returns the number of audio inputs/outputs.
        
        Args:
            audio_type: "in", "out", or "both" (default). If "both", returns max of in and out counts.
        
        Returns:
            Number of audio segments.
        """
        num_in = len(self.audio_in_ids_start) if self.audio_in_ids_start is not None else 0
        num_out = len(self.audio_out_ids_start) if self.audio_out_ids_start is not None else 0
        
        if audio_type == "in":
            return num_in
        elif audio_type == "out":
            return num_out
        else:  # "both"
            return max(num_in, num_out)

    def get_audio_out_codes(self, idx: int) -> Optional[torch.LongTensor]:
        """Get audio output codes for a specific index.
        
        Args:
            idx: Index of the audio output segment.
        
        Returns:
            Audio codes with shape (num_codebooks, segment_length) or None if not available.
        """
        if self.audio_out_ids is None or self.audio_out_ids_start is None:
            return None
        
        if idx < 0 or idx >= len(self.audio_out_ids_start):
            raise IndexError(f"Index {idx} out of range for audio_out_ids_start (length {len(self.audio_out_ids_start)})")
        
        code_start = self.audio_out_ids_start[idx]
        if idx < len(self.audio_out_ids_start) - 1:
            code_end = self.audio_out_ids_start[idx + 1]
        else:
            code_end = self.audio_out_ids.shape[-1]
        
        return self.audio_out_ids[:, code_start:code_end]

    def get_audio_in_codes(self, idx: int) -> Optional[torch.LongTensor]:
        """Get audio input codes for a specific index.
        
        Args:
            idx: Index of the audio input segment.
        
        Returns:
            Audio codes with shape (num_codebooks, segment_length) or None if not available.
        """
        if self.audio_in_ids is None or self.audio_in_ids_start is None:
            return None
        
        if idx < 0 or idx >= len(self.audio_in_ids_start):
            raise IndexError(f"Index {idx} out of range for audio_in_ids_start (length {len(self.audio_in_ids_start)})")
        
        code_start = self.audio_in_ids_start[idx]
        if idx < len(self.audio_in_ids_start) - 1:
            code_end = self.audio_in_ids_start[idx + 1]
        else:
            code_end = self.audio_in_ids.shape[-1]
        
        return self.audio_in_ids[:, code_start:code_end]

    def get_audio_codes_labels(self, idx: int) -> Optional[torch.LongTensor]:
        """Get audio code labels for a specific audio output index.
        
        Args:
            idx: Index of the audio output segment (corresponds to audio_out_ids_start).
        
        Returns:
            Label audio codes with shape (num_codebooks, segment_length) or None if not available.
        """
        if self.label_audio_ids is None or self.audio_out_ids_start is None:
            return None
        
        if idx < 0 or idx >= len(self.audio_out_ids_start):
            raise IndexError(f"Index {idx} out of range for audio_out_ids_start (length {len(self.audio_out_ids_start)})")
        
        code_start = self.audio_out_ids_start[idx]
        if idx < len(self.audio_out_ids_start) - 1:
            code_end = self.audio_out_ids_start[idx + 1]
        else:
            code_end = self.label_audio_ids.shape[-1]
        
        return self.label_audio_ids[:, code_start:code_end]

    def get_audio_features(self, idx: int) -> Optional[Tuple[torch.Tensor, Optional[torch.Tensor]]]:
        """Get audio features for a specific audio input index.
        
        Args:
            idx: Index of the audio input segment.
        
        Returns:
            Tuple of (features, mask) where features has shape (feature_dim, max_mel_seq_len)
            and mask has shape (max_mel_seq_len,) or None. Returns None if audio_features is not available.
        """
        if self.audio_features is None:
            return None
        
        if idx < 0 or idx >= self.audio_features.shape[0]:
            raise IndexError(f"Index {idx} out of range for audio_features (shape {self.audio_features.shape})")
        
        features = self.audio_features[idx]
        mask = None
        if self.audio_feature_attention_mask is not None:
            mask = self.audio_feature_attention_mask[idx]
        
        return features, mask

    def cal_num_tokens(
        self,
        encode_whisper_embed: bool = True,
        encode_audio_in_tokens: bool = False,
        encode_audio_out_tokens: bool = True,
        audio_in_token_id: int = 128015,
        audio_out_token_id: int = 128016,
    ) -> int:
        """Calculate the total number of tokens in the batch.
        
        Args:
            encode_whisper_embed: Whether to count Whisper embedding tokens from audio_features.
            encode_audio_in_tokens: Whether to count audio input tokens.
            encode_audio_out_tokens: Whether to count audio output tokens.
            audio_in_token_id: Token ID for audio input markers in input_ids.
            audio_out_token_id: Token ID for audio output markers in input_ids.
        
        Returns:
            Total number of tokens.
        """
        # Start with text tokens (exclude audio marker tokens)
        num_tokens = 0
        batch_size = self.input_ids.shape[0]
        
        for b in range(batch_size):
            seq_len = self.attention_mask[b].sum().item() if self.attention_mask is not None else self.input_ids.shape[1]
            input_seq = self.input_ids[b, :seq_len]
            
            # Count non-audio-marker tokens
            non_audio_mask = (input_seq != audio_in_token_id) & (input_seq != audio_out_token_id)
            num_tokens += non_audio_mask.sum().item()
        
        # Add Whisper embedding tokens from audio_features
        if encode_whisper_embed and self.audio_features is not None:
            # audio_features shape: (num_audio_in, feature_dim, max_mel_seq_len)
            # Each feature represents ~1/25 seconds of audio
            num_audio_in = self.audio_features.shape[0]
            for idx in range(num_audio_in):
                if self.audio_feature_attention_mask is not None:
                    # Count actual features (non-padded)
                    feature_length = self.audio_feature_attention_mask[idx].sum().item()
                else:
                    # Use full sequence length
                    feature_length = self.audio_features.shape[-1]
                num_tokens += feature_length
        
        # Add audio input tokens
        if encode_audio_in_tokens and self.audio_in_ids is not None and self.audio_in_ids_start is not None:
            if self.audio_in_ids.size(1) > 0:
                # Count tokens for each audio input segment
                for idx in range(len(self.audio_in_ids_start)):
                    code_start = self.audio_in_ids_start[idx]
                    if idx < len(self.audio_in_ids_start) - 1:
                        code_end = self.audio_in_ids_start[idx + 1]
                    else:
                        code_end = self.audio_in_ids.shape[-1]
                    num_tokens += (code_end - code_start) * self.audio_in_ids.shape[0]  # num_codebooks
        
        # Add audio output tokens
        if encode_audio_out_tokens and self.audio_out_ids is not None and self.audio_out_ids_start is not None:
            if self.audio_out_ids.size(1) > 0:
                # Count tokens for each audio output segment
                for idx in range(len(self.audio_out_ids_start)):
                    code_start = self.audio_out_ids_start[idx]
                    if idx < len(self.audio_out_ids_start) - 1:
                        code_end = self.audio_out_ids_start[idx + 1]
                    else:
                        code_end = self.audio_out_ids.shape[-1]
                    num_tokens += (code_end - code_start) * self.audio_out_ids.shape[0]  # num_codebooks
        
        return int(num_tokens)

    @classmethod
    def merge(
        cls,
        batches: List["ModelBatchInput"],
        pad_token_id: int,
        ignore_index: int = -100,
    ) -> "ModelBatchInput":
        """Merge multiple ModelBatchInput batches into a single batch.
        
        Args:
            batches: List of ModelBatchInput instances to merge.
            pad_token_id: Token ID used for padding.
            ignore_index: Default label for padding (default: -100).
        
        Returns:
            Merged ModelBatchInput instance.
        """
        if not batches:
            raise ValueError("The batches list is empty and cannot be merged.")
        
        # Concatenate input_ids and label_ids along batch dimension
        input_ids_list = []
        attention_mask_list = []
        label_ids_list = []
        
        for batch in batches:
            input_ids_list.append(batch.input_ids)
            attention_mask_list.append(batch.attention_mask)
            if batch.label_ids is not None:
                label_ids_list.append(batch.label_ids)
            else:
                # Create dummy label_ids if missing
                label_ids_list.append(torch.full_like(batch.input_ids, ignore_index))
        
        # Pad to same length
        max_seq_len = max(ids.shape[1] for ids in input_ids_list)
        batch_size = sum(ids.shape[0] for ids in input_ids_list)
        
        padded_input_ids = torch.full((batch_size, max_seq_len), pad_token_id, dtype=torch.long, device=input_ids_list[0].device)
        padded_attention_mask = torch.zeros((batch_size, max_seq_len), dtype=attention_mask_list[0].dtype, device=attention_mask_list[0].device)
        padded_label_ids = torch.full((batch_size, max_seq_len), ignore_index, dtype=torch.long, device=label_ids_list[0].device)
        
        offset = 0
        for batch in batches:
            bsz = batch.input_ids.shape[0]
            seq_len = batch.input_ids.shape[1]
            padded_input_ids[offset:offset+bsz, :seq_len] = batch.input_ids
            padded_attention_mask[offset:offset+bsz, :seq_len] = batch.attention_mask
            padded_label_ids[offset:offset+bsz, :seq_len] = batch.label_ids if batch.label_ids is not None else torch.full_like(batch.input_ids, ignore_index)
            offset += bsz
        
        # Concatenate audio features
        audio_features_list = []
        audio_feature_attention_mask_list = []
        for batch in batches:
            if batch.audio_features is not None:
                audio_features_list.append(batch.audio_features)
                if batch.audio_feature_attention_mask is not None:
                    audio_feature_attention_mask_list.append(batch.audio_feature_attention_mask)
        
        merged_audio_features = torch.cat(audio_features_list, dim=0) if audio_features_list else None
        merged_audio_feature_attention_mask = torch.cat(audio_feature_attention_mask_list, dim=0) if audio_feature_attention_mask_list else None
        
        # Concatenate audio_in_ids
        audio_in_ids_list = []
        audio_in_ids_start_list = []
        audio_in_offset = 0
        
        for batch in batches:
            if batch.audio_in_ids is not None and batch.audio_in_ids.size(1) > 0:
                audio_in_ids_list.append(batch.audio_in_ids)
                if batch.audio_in_ids_start is not None:
                    audio_in_ids_start_list.append(batch.audio_in_ids_start + audio_in_offset)
                    audio_in_offset += batch.audio_in_ids.size(1)
        
        merged_audio_in_ids = torch.cat(audio_in_ids_list, dim=1) if audio_in_ids_list else None
        merged_audio_in_ids_start = torch.cat(audio_in_ids_start_list, dim=0) if audio_in_ids_start_list else None
        
        # Concatenate audio_out_ids
        audio_out_ids_list = []
        audio_out_ids_start_list = []
        audio_out_ids_start_group_loc_list = []
        audio_out_offset = 0
        batch_offset = 0
        
        for batch in batches:
            if batch.audio_out_ids is not None and batch.audio_out_ids.size(1) > 0:
                audio_out_ids_list.append(batch.audio_out_ids)
                if batch.audio_out_ids_start is not None:
                    audio_out_ids_start_list.append(batch.audio_out_ids_start + audio_out_offset)
                    audio_out_offset += batch.audio_out_ids.size(1)
                    
                    # Update group locations
                    if batch.audio_out_ids_start_group_loc is not None:
                        audio_out_ids_start_group_loc_list.append(batch.audio_out_ids_start_group_loc + batch_offset)
                    else:
                        # Create default group locations if missing
                        num_audio_out = len(batch.audio_out_ids_start)
                        audio_out_ids_start_group_loc_list.append(torch.full((num_audio_out,), batch_offset, dtype=torch.long, device=batch.audio_out_ids.device))
                
                batch_offset += batch.input_ids.shape[0]
        
        merged_audio_out_ids = torch.cat(audio_out_ids_list, dim=1) if audio_out_ids_list else None
        merged_audio_out_ids_start = torch.cat(audio_out_ids_start_list, dim=0) if audio_out_ids_start_list else None
        merged_audio_out_ids_start_group_loc = torch.cat(audio_out_ids_start_group_loc_list, dim=0) if audio_out_ids_start_group_loc_list else None
        
        # Concatenate label_audio_ids
        label_audio_ids_list = []
        for batch in batches:
            if batch.label_audio_ids is not None and batch.label_audio_ids.size(1) > 0:
                label_audio_ids_list.append(batch.label_audio_ids)
        
        merged_label_audio_ids = torch.cat(label_audio_ids_list, dim=1) if label_audio_ids_list else None
        
        # Handle reward (use first non-None reward, or None)
        merged_reward = None
        for batch in batches:
            if batch.reward is not None:
                merged_reward = batch.reward
                break
        
        return cls(
            input_ids=padded_input_ids,
            attention_mask=padded_attention_mask,
            audio_features=merged_audio_features,
            audio_feature_attention_mask=merged_audio_feature_attention_mask,
            audio_out_ids=merged_audio_out_ids,
            audio_out_ids_start=merged_audio_out_ids_start,
            audio_out_ids_start_group_loc=merged_audio_out_ids_start_group_loc,
            audio_in_ids=merged_audio_in_ids,
            audio_in_ids_start=merged_audio_in_ids_start,
            label_ids=padded_label_ids,
            label_audio_ids=merged_label_audio_ids,
            reward=merged_reward,
        )