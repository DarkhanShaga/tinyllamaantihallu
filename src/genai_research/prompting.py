from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class PromptStrategy(str, Enum):
    """How to format the user turn for experiments (e.g. double prompting)."""

    # With context: Context + blank line + query. No context: query only.
    SINGLE = "single"
    # Repeat the natural-language query twice within one user message (query–context–query).
    DOUBLE_QUERY = "double_query"
    # With context: Context + query block repeated twice.
    SANDWICH = "sandwich"
    # Duplicate the full user block (after construction) for ablations.
    REPEAT_USER_BLOCK = "repeat_user_block"


@dataclass(frozen=True)
class UserTurn:
    """One chat user message worth of text (plus optional system string)."""

    user_content: str
    system_content: str | None = None


def _default_instruction() -> str:
    return (
        "Use ONLY the provided context to answer. If the answer is not contained in the context, "
        "you must output 'I don't know'"
    )


def resolve_prompt_context(
    context: str | None,
    reference: str | None,
    *,
    source: str = "auto",
) -> str:
    """
    Decide which string is passed to build_messages() as the RAG "context" turn.

    Some datasets (e.g. libreeval on HF) put the source passage in `reference` and omit
    a separate `context` column. In that case ``source=auto`` uses `reference` when
    `context` is empty.
    """
    c = (context or "").strip()
    r = (reference or "").strip()
    if source == "context":
        return c
    if source == "reference":
        return r
    if source == "concat":
        return "\n\n".join(x for x in (c, r) if x)
    # auto
    return c if c else r


def build_user_content(
    query: str,
    context: str | None = None,
    *,
    instruction: str | None = None,
    strategy: PromptStrategy = PromptStrategy.SINGLE,
) -> str:
    """
    Build the user message body for a single turn, following common double-prompt patterns.

    * SINGLE: with context, ``<context>\n\n<query>``; with no context, query only.
    * DOUBLE_QUERY: with context, query + context + query again; with no context, exactly two lines: `Question: <query>` and `Question: <query>`.
    * SANDWICH: with context, ``<context>\n\n<query>\n\n<context>\n\n<query>``; with no context, instruction + query.
    * REPEAT_USER_BLOCK: like SINGLE, then the same string duplicated (coarse ablation)
    """
    inst = instruction if instruction is not None else _default_instruction()
    q = query.strip()
    ctx = (context or "").strip()

    if strategy == PromptStrategy.SINGLE:
        if ctx:
            return f"{ctx}\n\n{q}\n"
        return q

    if strategy == PromptStrategy.DOUBLE_QUERY:
        if ctx:
            return f"Question: {q}\n\nContext:\n{ctx}\n\nQuestion (repeated): {q}\n"
        return f"Question: {q}\nQuestion: {q}\n"

    if strategy == PromptStrategy.SANDWICH:
        if ctx:
            return f"{ctx}\n\n{q}\n\n{ctx}\n\n{q}\n"
        return f"{inst}\n\n{q}\n"

    if strategy == PromptStrategy.REPEAT_USER_BLOCK:
        base = build_user_content(
            q, context=ctx, instruction=instruction, strategy=PromptStrategy.SINGLE
        )
        return f"{base.rstrip()}\n\n-----\n{base.rstrip()}\n"

    raise NotImplementedError(strategy)


def build_messages(
    query: str,
    context: str | None = None,
    *,
    instruction: str | None = None,
    strategy: PromptStrategy = PromptStrategy.SINGLE,
    system: str | None = None,
) -> list[dict[str, str]]:
    """Build HuggingFace-style chat `messages` (system + user)."""
    user = build_user_content(
        query, context=context, instruction=instruction, strategy=strategy
    )
    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})
    return messages
