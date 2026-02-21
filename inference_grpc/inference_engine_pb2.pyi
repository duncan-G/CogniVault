# mypy: ignore-errors
from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class SamplingParams(_message.Message):
    __slots__ = ("temperature", "top_p", "top_k", "max_tokens", "seed", "ras_win_len", "ras_win_max_num_repeat", "force_audio_gen")
    TEMPERATURE_FIELD_NUMBER: _ClassVar[int]
    TOP_P_FIELD_NUMBER: _ClassVar[int]
    TOP_K_FIELD_NUMBER: _ClassVar[int]
    MAX_TOKENS_FIELD_NUMBER: _ClassVar[int]
    SEED_FIELD_NUMBER: _ClassVar[int]
    RAS_WIN_LEN_FIELD_NUMBER: _ClassVar[int]
    RAS_WIN_MAX_NUM_REPEAT_FIELD_NUMBER: _ClassVar[int]
    FORCE_AUDIO_GEN_FIELD_NUMBER: _ClassVar[int]
    temperature: float
    top_p: float
    top_k: int
    max_tokens: int
    seed: int
    ras_win_len: int
    ras_win_max_num_repeat: int
    force_audio_gen: bool
    def __init__(self, temperature: _Optional[float] = ..., top_p: _Optional[float] = ..., top_k: _Optional[int] = ..., max_tokens: _Optional[int] = ..., seed: _Optional[int] = ..., ras_win_len: _Optional[int] = ..., ras_win_max_num_repeat: _Optional[int] = ..., force_audio_gen: bool = ...) -> None: ...

class Speaker(_message.Message):
    __slots__ = ("name", "description", "audio_url")
    NAME_FIELD_NUMBER: _ClassVar[int]
    DESCRIPTION_FIELD_NUMBER: _ClassVar[int]
    AUDIO_URL_FIELD_NUMBER: _ClassVar[int]
    name: str
    description: str
    audio_url: str
    def __init__(self, name: _Optional[str] = ..., description: _Optional[str] = ..., audio_url: _Optional[str] = ...) -> None: ...

class TextContent(_message.Message):
    __slots__ = ("text", "type")
    TEXT_FIELD_NUMBER: _ClassVar[int]
    TYPE_FIELD_NUMBER: _ClassVar[int]
    text: str
    type: str
    def __init__(self, text: _Optional[str] = ..., type: _Optional[str] = ...) -> None: ...

class AudioContent(_message.Message):
    __slots__ = ("audio_url", "raw_audio", "type")
    AUDIO_URL_FIELD_NUMBER: _ClassVar[int]
    RAW_AUDIO_FIELD_NUMBER: _ClassVar[int]
    TYPE_FIELD_NUMBER: _ClassVar[int]
    audio_url: str
    raw_audio: str
    type: str
    def __init__(self, audio_url: _Optional[str] = ..., raw_audio: _Optional[str] = ..., type: _Optional[str] = ...) -> None: ...

class MessageContent(_message.Message):
    __slots__ = ("text", "audio")
    TEXT_FIELD_NUMBER: _ClassVar[int]
    AUDIO_FIELD_NUMBER: _ClassVar[int]
    text: TextContent
    audio: AudioContent
    def __init__(self, text: _Optional[_Union[TextContent, _Mapping]] = ..., audio: _Optional[_Union[AudioContent, _Mapping]] = ...) -> None: ...

class Message(_message.Message):
    __slots__ = ("role", "content", "recipient", "speaker")
    ROLE_FIELD_NUMBER: _ClassVar[int]
    CONTENT_FIELD_NUMBER: _ClassVar[int]
    RECIPIENT_FIELD_NUMBER: _ClassVar[int]
    SPEAKER_FIELD_NUMBER: _ClassVar[int]
    role: str
    content: MessageContent
    recipient: str
    speaker: Speaker
    def __init__(self, role: _Optional[str] = ..., content: _Optional[_Union[MessageContent, _Mapping]] = ..., recipient: _Optional[str] = ..., speaker: _Optional[_Union[Speaker, _Mapping]] = ...) -> None: ...

class Chat(_message.Message):
    __slots__ = ("id", "messages", "metadata")
    class MetadataEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    ID_FIELD_NUMBER: _ClassVar[int]
    MESSAGES_FIELD_NUMBER: _ClassVar[int]
    METADATA_FIELD_NUMBER: _ClassVar[int]
    id: str
    messages: _containers.RepeatedCompositeFieldContainer[Message]
    metadata: _containers.ScalarMap[str, str]
    def __init__(self, id: _Optional[str] = ..., messages: _Optional[_Iterable[_Union[Message, _Mapping]]] = ..., metadata: _Optional[_Mapping[str, str]] = ...) -> None: ...

class GenerateRequest(_message.Message):
    __slots__ = ("request_id", "chats", "sampling_params", "stream")
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    CHATS_FIELD_NUMBER: _ClassVar[int]
    SAMPLING_PARAMS_FIELD_NUMBER: _ClassVar[int]
    STREAM_FIELD_NUMBER: _ClassVar[int]
    request_id: str
    chats: _containers.RepeatedCompositeFieldContainer[Chat]
    sampling_params: SamplingParams
    stream: bool
    def __init__(self, request_id: _Optional[str] = ..., chats: _Optional[_Iterable[_Union[Chat, _Mapping]]] = ..., sampling_params: _Optional[_Union[SamplingParams, _Mapping]] = ..., stream: bool = ...) -> None: ...

class GenerateResponse(_message.Message):
    __slots__ = ("chunk", "complete")
    CHUNK_FIELD_NUMBER: _ClassVar[int]
    COMPLETE_FIELD_NUMBER: _ClassVar[int]
    chunk: GenerateStreamChunk
    complete: GenerateComplete
    def __init__(self, chunk: _Optional[_Union[GenerateStreamChunk, _Mapping]] = ..., complete: _Optional[_Union[GenerateComplete, _Mapping]] = ...) -> None: ...

class GenerateStreamChunk(_message.Message):
    __slots__ = ("token_ids", "prompt_tokens", "completion_tokens", "audio_data", "sampling_rate")
    TOKEN_IDS_FIELD_NUMBER: _ClassVar[int]
    PROMPT_TOKENS_FIELD_NUMBER: _ClassVar[int]
    COMPLETION_TOKENS_FIELD_NUMBER: _ClassVar[int]
    AUDIO_DATA_FIELD_NUMBER: _ClassVar[int]
    SAMPLING_RATE_FIELD_NUMBER: _ClassVar[int]
    token_ids: _containers.RepeatedScalarFieldContainer[int]
    prompt_tokens: int
    completion_tokens: int
    audio_data: bytes
    sampling_rate: int
    def __init__(self, token_ids: _Optional[_Iterable[int]] = ..., prompt_tokens: _Optional[int] = ..., completion_tokens: _Optional[int] = ..., audio_data: _Optional[bytes] = ..., sampling_rate: _Optional[int] = ...) -> None: ...

class GenerateComplete(_message.Message):
    __slots__ = ("output_ids", "finish_reason", "prompt_tokens", "completion_tokens", "audio_data", "sampling_rate")
    OUTPUT_IDS_FIELD_NUMBER: _ClassVar[int]
    FINISH_REASON_FIELD_NUMBER: _ClassVar[int]
    PROMPT_TOKENS_FIELD_NUMBER: _ClassVar[int]
    COMPLETION_TOKENS_FIELD_NUMBER: _ClassVar[int]
    AUDIO_DATA_FIELD_NUMBER: _ClassVar[int]
    SAMPLING_RATE_FIELD_NUMBER: _ClassVar[int]
    output_ids: _containers.RepeatedScalarFieldContainer[int]
    finish_reason: str
    prompt_tokens: int
    completion_tokens: int
    audio_data: bytes
    sampling_rate: int
    def __init__(self, output_ids: _Optional[_Iterable[int]] = ..., finish_reason: _Optional[str] = ..., prompt_tokens: _Optional[int] = ..., completion_tokens: _Optional[int] = ..., audio_data: _Optional[bytes] = ..., sampling_rate: _Optional[int] = ...) -> None: ...
