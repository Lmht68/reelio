"""Provider-neutral types for Movie Mention interpretation."""

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True, slots=True)
class LLMMessage:
    """Contain one trusted-role message sent through an LLM provider.

    Attributes:
        role: Chat role controlling how the provider interprets the content.
        content: Complete message text.
    """

    role: Literal["system", "user", "assistant"]
    content: str
