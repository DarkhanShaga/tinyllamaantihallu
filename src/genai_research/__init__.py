"""Research helpers: TinyLlama loading, generation, and prompt strategies."""

__version__ = "0.1.0"

from genai_research.model import TinyLlamaEngine
from genai_research.prompting import PromptStrategy, build_messages, build_user_content

__all__ = [
    "TinyLlamaEngine",
    "PromptStrategy",
    "build_messages",
    "build_user_content",
    "__version__",
]
