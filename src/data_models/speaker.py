from dataclasses import dataclass
from typing import Optional

@dataclass
class Speaker:
    name: str
    description: str
    audio_url: Optional[str] = None