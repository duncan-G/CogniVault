"""gRPC client that sends a Generate request to the inference server and
writes the returned audio to a WAV file.

Usage:
    python scripts/test_server.py
    python scripts/test_server.py --host localhost --port 50051 --output output.wav
"""

import argparse
import os
import sys
import uuid
from pathlib import Path

import grpc
import numpy as np
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from inference_grpc import inference_engine_pb2 as pb2
from inference_grpc import inference_engine_pb2_grpc as pb2_grpc


# ============================================================================
# Speakers
# ============================================================================

ALEX = pb2.Speaker(
    name="SPEAKER0",
    description=(
        "Male, American accent, modern speaking rate, moderate-pitch, "
        "friendly tone, and very clear audio."
    ),
)

# ============================================================================
# Messages  (single-speaker example — swap / add speakers for multi-speaker)
# ============================================================================

MESSAGES = [
    pb2.InputMessage(
        text=(
            "Hey, everyone! Welcome back to Tech Talk Tuesdays. "
            "It's your host, Alex, and today, we're diving into a topic "
            "that's become absolutely crucial in the tech world - deep learning. "
            "<SE>[Laughter]</SE> I know, I know, you've probably heard that phrase "
            "a thousand times by now. But seriously, this stuff is fascinating. "
            "<SE>[Applause]</SE> Thank you, thank you! "
            "Alright, let's get into it."
        ),
        speaker=ALEX,
    ),
]

SCENE_DESCRIPTION = "Audio is recorded from a quiet room."


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="gRPC audio generation client")
    parser.add_argument("--host", default=os.getenv("HOST", "localhost"))
    parser.add_argument("--port", type=int, default=int(os.getenv("PORT", "50051")))
    parser.add_argument("--output", default="grpc_output.wav")
    args = parser.parse_args()

    address = f"{args.host}:{args.port}"
    print(f"Connecting to {address}...")

    channel = grpc.insecure_channel(
        address,
        options=[
            ("grpc.max_send_message_length", -1),
            ("grpc.max_receive_message_length", -1),
        ],
    )
    stub = pb2_grpc.InferenceEngineStub(channel)

    request = pb2.GenerateRequest(
        request_id=str(uuid.uuid4()),
        inputs=[
            pb2.GenerationInput(
                messages=MESSAGES,
                scene_description=SCENE_DESCRIPTION,
            ),
        ],
        sampling_params=pb2.SamplingParams(
            temperature=1.0,
            top_p=0.95,
            top_k=50,
            max_tokens=2048,
            seed=123,
            ras_win_len=7,
            ras_win_max_num_repeat=2,
        ),
        stream=False,
    )

    print("Sending Generate request...")
    audio_chunks = []
    sampling_rate = 24000

    for response in stub.Generate(request):
        resp_type = response.WhichOneof("response")

        if resp_type == "chunk":
            chunk = response.chunk
            print(f"  Stream chunk: {chunk.completion_tokens} tokens")
            if chunk.audio_data:
                pcm = np.frombuffer(chunk.audio_data, dtype=np.float32)
                audio_chunks.append(pcm)
                if chunk.sampling_rate:
                    sampling_rate = chunk.sampling_rate

        elif resp_type == "complete":
            complete = response.complete
            print(f"  Complete: finish_reason={complete.finish_reason}, "
                  f"prompt_tokens={complete.prompt_tokens}, "
                  f"completion_tokens={complete.completion_tokens}")
            if complete.audio_data:
                pcm = np.frombuffer(complete.audio_data, dtype=np.float32)
                audio_chunks.append(pcm)
                if complete.sampling_rate:
                    sampling_rate = complete.sampling_rate

    channel.close()

    if not audio_chunks:
        print("Error: No audio received from server!")
        return

    final_audio = np.concatenate(audio_chunks)
    sf.write(args.output, final_audio, sampling_rate)

    duration = len(final_audio) / sampling_rate
    print(f"Audio saved to {args.output}")
    print(f"  Duration: {duration:.2f}s, Sampling rate: {sampling_rate} Hz")


if __name__ == "__main__":
    main()
