from dataclasses import dataclass
from typing import List, Optional
from .message import Message
import uuid

@dataclass
class Chat:
    id: uuid.UUID
    messages: List[Message]
    metadata: Optional[dict] = None