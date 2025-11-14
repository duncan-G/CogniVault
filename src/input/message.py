from dataclasses import dataclass
from typing import Optional, Union

from src.input.message_content import AudioContent, TextContent

@dataclass
class Message:
    role: str
    content: Union[AudioContent, TextContent]
    recipient: Optional[str] = None