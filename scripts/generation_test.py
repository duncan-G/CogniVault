"""Example script for generating audio using HiggsAudio."""

import uuid
import sys
from pathlib import Path
import numpy as np
import soundfile as sf
from typing import List

# Allow running the script directly without installing the repo as a package
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from src.generation.engine import AudioEngine
from src.data_models.chat import Chat
from src.data_models.message import Message
from src.data_models.message_content import TextContent, AudioContent
from src.data_models.speaker import Speaker
from src.data_models.constants import SCENE_DESC_START, SCENE_DESC_END, AUDIO_PLACEHOLDER_TOKEN


# ============================================================================
# Configuration - All data defined here
# ============================================================================

# User prompt - The text to convert to audio
USER_PROMPT = """Hey, everyone! Welcome back to Tech Talk Tuesdays.
It's your host, Alex, and today, we're diving into a topic that's become absolutely crucial in the tech world — deep learning.
And let's be honest, if you've been even remotely connected to tech, AI, or machine learning lately, you know that deep learning is everywhere.

So here's the big question: Do you want to understand how deep learning works?
How to use it to build powerful models that can predict, automate, and transform industries?
Well, today, I've got some exciting news for you.

We're going to talk about a course that I highly recommend: Dive into Deep Learning.
It's not just another course; it's an entire experience that will take you from a beginner to someone who is well-versed in deep learning techniques."""

# Scene description
SCENE_PROMPT = "Audio is recorded from a quiet room."

# Speaker configuration
# Each speaker can have:
# - description: Text description of the voice (e.g., "Male, American accent, modern speaking rate, moderate-pitch, friendly tone, and very clear audio.")
# - audio_url: Path to reference audio file (optional)
SPEAKERS = [
    Speaker(
        name="SPEAKER0",
        description="Male, American accent, modern speaking rate, moderate-pitch, friendly tone, and very clear audio.",
        audio_url=None,  # Set to a path if you have reference audio
    ),
]

# Generation parameters
MAX_NEW_TOKENS = 2048
TEMPERATURE = 1.0
TOP_K = 50
TOP_P = 0.95
RAS_WIN_LEN = 7
RAS_WIN_MAX_NUM_REPEAT = 2
SEED = 123

# Chunking configuration
# Set to None to process entire prompt at once
# Set to "speaker" to chunk by speaker tags
# Set to "word" to chunk by word count
CHUNK_METHOD = None
CHUNK_MAX_WORD_NUM = 100
CHUNK_MAX_NUM_TURNS = 1

# Output path
OUTPUT_PATH = "generation.wav"

# Model paths
MODEL_PATH = "bosonai/higgs-audio-v2-generation-3B-base"
AUDIO_TOKENIZER_PATH = "bosonai/higgs-audio-v2-tokenizer"


# ============================================================================
# Helper Functions
# ============================================================================

def normalize_prompt(text: str) -> str:
    """Normalize the prompt text according to the specification."""
    # Replace parentheses with spaces
    text = text.replace("(", " ")
    text = text.replace(")", " ")
    
    # Replace temperature symbols
    text = text.replace("°F", " degrees Fahrenheit")
    text = text.replace("°C", " degrees Celsius")
    
    # Replace sound effects with tags
    sound_effect_replacements = [
        ("[laugh]", "<SE>[Laughter]</SE>"),
        ("[humming start]", "<SE_s>[Humming]</SE_s>"),
        ("[humming end]", "<SE_e>[Humming]</SE_e>"),
        ("[music start]", "<SE_s>[Music]</SE_s>"),
        ("[music end]", "<SE_e>[Music]</SE_e>"),
        ("[music]", "<SE>[Music]</SE>"),
        ("[sing start]", "<SE_s>[Singing]</SE_s>"),
        ("[sing end]", "<SE_e>[Singing]</SE_e>"),
        ("[applause]", "<SE>[Applause]</SE>"),
        ("[cheering]", "<SE>[Cheering]</SE>"),
        ("[cough]", "<SE>[Cough]</SE>"),
    ]
    
    for tag, replacement in sound_effect_replacements:
        text = text.replace(tag, replacement)
    
    # Strip extra spaces
    lines = text.split("\n")
    text = "\n".join([" ".join(line.split()) for line in lines if line.strip()])
    text = text.strip()
    
    # Ensure text ends with punctuation
    if not any([text.endswith(c) for c in [".", "!", "?", ",", ";", '"', "'", "</SE_e>", "</SE>"]]):
        text += "."
    
    return text


def chunk_text(text: str, method: str = None, max_word_num: int = 100, max_num_turns: int = 1) -> List[str]:
    """Chunk the text into smaller pieces for generation."""
    if method is None:
        return [text]
    
    elif method == "speaker":
        # Chunk by speaker tags
        lines = text.split("\n")
        speaker_chunks = []
        speaker_utterance = ""
        
        for line in lines:
            line = line.strip()
            if line.startswith("[SPEAKER") or line.startswith("<|speaker_id_start|>"):
                if speaker_utterance:
                    speaker_chunks.append(speaker_utterance.strip())
                speaker_utterance = line
            else:
                if speaker_utterance:
                    speaker_utterance += "\n" + line
                else:
                    speaker_utterance = line
        
        if speaker_utterance:
            speaker_chunks.append(speaker_utterance.strip())
        
        # Merge chunks if max_num_turns > 1
        if max_num_turns > 1:
            merged_chunks = []
            for i in range(0, len(speaker_chunks), max_num_turns):
                merged_chunk = "\n".join(speaker_chunks[i : i + max_num_turns])
                merged_chunks.append(merged_chunk)
            return merged_chunks
        
        return speaker_chunks
    
    elif method == "word":
        # Chunk by word count
        paragraphs = text.split("\n\n")
        chunks = []
        
        for paragraph in paragraphs:
            words = paragraph.split(" ")
            for i in range(0, len(words), max_word_num):
                chunk = " ".join(words[i : i + max_word_num])
                chunks.append(chunk)
            if chunks:
                chunks[-1] += "\n\n"
        
        return chunks
    
    else:
        raise ValueError(f"Unknown chunk method: {method}")


def build_system_message(scene_prompt: str, speakers: List[Speaker]) -> Message:
    """Build the system message with scene description and speaker information."""
    # Build speaker descriptions with placeholders for audio
    speaker_descriptions = []
    for speaker in speakers:
        if speaker.audio_url:
            # Use audio placeholder if audio is provided
            speaker_descriptions.append(f"{speaker.name}: {AUDIO_PLACEHOLDER_TOKEN}")
        else:
            # Use text description
            speaker_descriptions.append(f"{speaker.name}: {speaker.description}")
    
    # Build system message text
    system_parts = [
        "Generate audio following instruction.",
        "",
        f"{SCENE_DESC_START}",
        scene_prompt,
        "",
        "\n".join(speaker_descriptions),
        f"{SCENE_DESC_END}",
    ]
    
    system_text = "\n".join(system_parts)
    
    # Build content list with audio placeholders interleaved
    # The chat processor will handle lists of content items
    content_list = []
    remaining_text = system_text
    
    while AUDIO_PLACEHOLDER_TOKEN in remaining_text:
        loc = remaining_text.find(AUDIO_PLACEHOLDER_TOKEN)
        # Add text before placeholder
        if loc > 0:
            content_list.append(TextContent(text=remaining_text[:loc]))
        # Add audio placeholder (empty audio_url means placeholder)
        content_list.append(AudioContent(audio_url=""))
        # Continue with remaining text
        remaining_text = remaining_text[loc + len(AUDIO_PLACEHOLDER_TOKEN):]
    
    # Add remaining text
    if remaining_text:
        content_list.append(TextContent(text=remaining_text))
    
    # If no placeholders were found, just use text content
    if not content_list:
        content_list = [TextContent(text=system_text)]
    
    # Return message with list of content (processor handles this)
    return Message(role="system", content=content_list)


def build_chat_with_speaker_audio(
    system_message: Message,
    speakers: List[Speaker],
    normalized_prompt: str,
    chunked_prompts: List[str],
) -> List[Chat]:
    """Build chat objects for generation, including speaker audio if provided."""
    chats = []
    
    # Build initial messages with system message
    # Note: If speakers have audio_url, they should be included in the system message
    # as placeholders, and then provided as separate user/assistant pairs
    messages = [system_message]
    
    # Add speaker audio as user/assistant pairs if audio is provided
    # This follows the pattern: user describes speaker, assistant provides audio
    for speaker in speakers:
        if speaker.audio_url:
            # User message with speaker description
            messages.append(
                Message(
                    role="user",
                    content=TextContent(text=f"{speaker.name}: {speaker.description}"),
                )
            )
            # Assistant message with audio
            messages.append(
                Message(
                    role="assistant",
                    content=AudioContent(audio_url=speaker.audio_url),
                )
            )
    
    # Create a chat for each chunk
    # Each chunk will be processed separately, maintaining context from previous chunks
    for chunk_text in chunked_prompts:
        chunk_messages = messages.copy()
        
        # Add user message with chunk text
        chunk_messages.append(
            Message(
                role="user",
                content=TextContent(text=chunk_text),
            )
        )
        
        chat = Chat(
            id=uuid.uuid4(),
            messages=chunk_messages,
        )
        chats.append(chat)
    
    return chats


# ============================================================================
# Main Generation Function
# ============================================================================

def main():
    """Main generation function."""
    print("Initializing AudioEngine...")
    engine = AudioEngine(
        model_name_or_path=MODEL_PATH,
        tokenizer_name_or_path=MODEL_PATH,
        audio_tokenizer_name_or_path=AUDIO_TOKENIZER_PATH,
    )
    
    print("Normalizing prompt...")
    normalized_prompt = normalize_prompt(USER_PROMPT)
    
    print("Chunking prompt...")
    chunked_prompts = chunk_text(
        normalized_prompt,
        method=CHUNK_METHOD,
        max_word_num=CHUNK_MAX_WORD_NUM,
        max_num_turns=CHUNK_MAX_NUM_TURNS,
    )
    
    print(f"Prompt split into {len(chunked_prompts)} chunk(s)")
    for idx, chunk in enumerate(chunked_prompts):
        print(f"Chunk {idx + 1}: {chunk[:100]}...")
    
    print("Building system message...")
    system_message = build_system_message(SCENE_PROMPT, SPEAKERS)
    
    print("Building chat objects...")
    chats = build_chat_with_speaker_audio(
        system_message,
        SPEAKERS,
        normalized_prompt,
        chunked_prompts,
    )
    
    print("Generating audio...")
    audio_chunks = []
    
    for idx, chat in enumerate(chats):
        print(f"Generating chunk {idx + 1}/{len(chats)}...")
        response = engine.generate(
            chat=chat,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            top_k=TOP_K,
            top_p=TOP_P,
            ras_win_len=RAS_WIN_LEN,
            ras_win_max_num_repeat=RAS_WIN_MAX_NUM_REPEAT,
            seed=SEED,
        )
        
        if response.audio is not None:
            audio_chunks.append(response.audio)
            print(f"  Generated {len(response.audio) / response.sampling_rate:.2f} seconds of audio")
        else:
            print(f"  Warning: No audio generated for chunk {idx + 1}")
    
    if not audio_chunks:
        print("Error: No audio was generated!")
        return
    
    print("Concatenating audio chunks...")
    final_audio = np.concatenate(audio_chunks)
    
    print(f"Saving audio to {OUTPUT_PATH}...")
    sampling_rate = response.sampling_rate if response.sampling_rate else 24000
    sf.write(OUTPUT_PATH, final_audio, sampling_rate)
    
    total_duration = len(final_audio) / sampling_rate
    print(f"✓ Audio saved successfully!")
    print(f"  Total duration: {total_duration:.2f} seconds")
    print(f"  Sampling rate: {sampling_rate} Hz")


if __name__ == "__main__":
    main()
