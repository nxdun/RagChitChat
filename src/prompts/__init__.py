"""
Prompt engineering module for RagChitChat
"""

from .prompt_templates import (
    get_rag_prompt,
    get_reflection_prompt,
    get_structured_prompt
)

__all__ = ["get_rag_prompt", "get_reflection_prompt", "get_structured_prompt"]
