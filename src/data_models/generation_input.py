from dataclasses import dataclass, field
from typing import List
import uuid

from .speaker import Speaker


@dataclass
class GenerationChatInput:
    """High-level input for audio generation.

    Clients pass an array of these objects. The InputProcessor normalizes
    the prompt, builds system messages with scene/speaker info, and produces
    Chat objects ready for tokenization and generation.
    """
    prompt: str
    scene_description: str
    speakers: List[Speaker]
    id: uuid.UUID = field(default_factory=uuid.uuid4)
