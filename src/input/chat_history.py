from dataclasses import dataclass
from typing import List, Optional
from src.input.message import Message
import uuid

@dataclass
class ChatHistory:
    id: uuid.UUID
    messages: List[Message]
    metadata: Optional[dict] = None