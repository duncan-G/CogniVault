"""
Async gRPC servicer for the InferenceEngine service.

Delegates generation to a synchronous :class:`AudioEngine`, running the
blocking call in a thread pool so the gRPC async event loop stays
responsive.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from collections.abc import AsyncGenerator
from typing import Any, Union

import grpc
import numpy as np

from inference_grpc import inference_engine_pb2 as pb2
from inference_grpc import inference_engine_pb2_grpc

from src.data_models.chat import Chat
from src.data_models.message import Message
from src.data_models.message_content import AudioContent, TextContent
from src.data_models.response import Response
from src.data_models.speaker import Speaker
from src.generation.engine import AudioEngine

logger = logging.getLogger(__name__)


class InferenceEngineServicer(inference_engine_pb2_grpc.InferenceEngineServicer):
    """Async gRPC servicer implementing the ``InferenceEngine`` service.

    RPCs implemented:
        - **Generate** — audio generation via :class:`AudioEngine`
    """

    def __init__(self, engine: AudioEngine) -> None:
        self.engine = engine
        logger.info("InferenceEngineServicer (async) initialized")

    # ------------------------------------------------------------------
    # Generate  (server-streaming RPC)
    # ------------------------------------------------------------------

    async def Generate(
        self,
        request: pb2.GenerateRequest,
        context: grpc.aio.ServicerContext,
    ) -> AsyncGenerator[pb2.GenerateResponse, None]:
        """Handle generation requests.

        Converts the proto ``GenerateRequest`` into internal types, runs
        ``engine.generate`` in a thread pool (the engine is synchronous),
        and yields one ``GenerateComplete`` per chat.
        """
        request_id = request.request_id
        logger.debug("Generate request %s received.", request_id)

        try:
            params = _sampling_params_from_proto(request.sampling_params)
            chats = [_proto_chat_to_internal(c) for c in request.chats]

            for chat in chats:
                response: Response = await asyncio.to_thread(
                    self.engine.generate,
                    chat=chat,
                    **params,
                )
                yield _build_response(response)

        except ValueError as e:
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(e))
        except Exception as e:
            logger.exception("Error in Generate for request %s", request_id)
            await context.abort(grpc.StatusCode.INTERNAL, str(e))


# ======================================================================
# Proto → internal conversion helpers
# ======================================================================

def _proto_chat_to_internal(proto_chat: Any) -> Chat:
    """Convert a proto ``Chat`` message to an internal :class:`Chat`."""
    try:
        chat_id = uuid.UUID(proto_chat.id)
    except (ValueError, AttributeError):
        chat_id = uuid.uuid4()

    messages: list[Message] = []
    for proto_msg in proto_chat.messages:
        content_field = proto_msg.content.WhichOneof("content")
        if content_field == "text":
            content: Union[TextContent, AudioContent] = TextContent(
                text=proto_msg.content.text.text,
                type=proto_msg.content.text.type or "text",
            )
        elif content_field == "audio":
            content = AudioContent(
                audio_url=proto_msg.content.audio.audio_url,
                raw_audio=proto_msg.content.audio.raw_audio or None,
                type=proto_msg.content.audio.type or "audio",
            )
        else:
            content = TextContent(text="")

        speaker = None
        if proto_msg.HasField("speaker"):
            speaker = Speaker(
                name=proto_msg.speaker.name,
                description=proto_msg.speaker.description,
                audio_url=proto_msg.speaker.audio_url or None,
            )

        messages.append(
            Message(
                role=proto_msg.role,
                content=content,
                recipient=proto_msg.recipient or None,
                speaker=speaker,
            ),
        )

    metadata = dict(proto_chat.metadata) if proto_chat.metadata else None
    return Chat(id=chat_id, messages=messages, metadata=metadata)


def _sampling_params_from_proto(params: pb2.SamplingParams) -> dict[str, Any]:
    """Convert a protobuf ``SamplingParams`` into kwargs for
    :meth:`AudioEngine.generate`."""
    return {
        "max_new_tokens": (
            params.max_tokens if params.HasField("max_tokens") else 2048
        ),
        "temperature": (
            params.temperature if params.HasField("temperature") else 0.7
        ),
        "top_k": params.top_k or None,
        "top_p": params.top_p if params.top_p != 0.0 else 0.95,
        "force_audio_gen": params.force_audio_gen,
        "ras_win_len": (
            params.ras_win_len if params.HasField("ras_win_len") else 7
        ),
        "ras_win_max_num_repeat": params.ras_win_max_num_repeat or 2,
        "seed": params.seed if params.HasField("seed") else None,
    }


# ======================================================================
# Internal → proto response helpers
# ======================================================================

def _build_response(response: Response) -> pb2.GenerateResponse:
    """Build a ``GenerateComplete`` from an engine :class:`Response`."""
    usage = response.usage or {}

    audio_bytes = b""
    if response.audio is not None:
        audio_bytes = response.audio.astype(np.float32).tobytes()

    output_ids: list[int] = []
    if response.generated_text_tokens is not None:
        output_ids = response.generated_text_tokens.tolist()

    return pb2.GenerateResponse(
        complete=pb2.GenerateComplete(
            output_ids=output_ids,
            finish_reason="stop",
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            audio_data=audio_bytes,
            sampling_rate=response.sampling_rate or 0,
        ),
    )
