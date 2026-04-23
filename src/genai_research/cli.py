from __future__ import annotations

import argparse
import os
import sys

from genai_research.config import load_config, with_overrides
from genai_research.model import TinyLlamaEngine
from genai_research.paths import default_config_path
from genai_research.prompting import PromptStrategy, build_messages


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="TinyLlama chat generation for research (double-prompt strategies)."
    )
    p.add_argument(
        "prompt",
        nargs="*",
        help="User question text (if empty, reads one line from stdin).",
    )
    p.add_argument(
        "--context",
        type=str,
        default=None,
        help="Optional grounding context (RAG payload).",
    )
    p.add_argument(
        "--strategy",
        type=str,
        default=PromptStrategy.SINGLE.value,
        choices=[s.value for s in PromptStrategy],
        help="Prompt formatting strategy.",
    )
    p.add_argument(
        "--config",
        type=str,
        default=None,
        help=f"Optional YAML file (defaults: GENAI_CONFIG_PATH, then {default_config_path()}).",
    )
    p.add_argument(
        "--model-id",
        type=str,
        default=None,
        help="Override model id (or set GENAI_MODEL_ID).",
    )
    p.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
    )
    p.add_argument(
        "--temperature",
        type=float,
        default=None,
    )
    p.add_argument(
        "--no-download",
        action="store_true",
        help="Require local cache only (sets HF_HUB_OFFLINE=1 for this run).",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.no_download:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")

    from pathlib import Path

    cfg = load_config(Path(args.config) if args.config else None)
    mid = args.model_id or os.environ.get("GENAI_MODEL_ID")
    cfg = with_overrides(
        cfg,
        model_id=mid,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )

    strategy = PromptStrategy(args.strategy)
    if args.prompt:
        user_q = " ".join(args.prompt).strip()
    else:
        user_q = sys.stdin.readline().strip()
    if not user_q:
        print("Error: empty prompt", file=sys.stderr)
        return 2

    print("Loading model (first run may download weights)…", file=sys.stderr)
    engine = TinyLlamaEngine(cfg)
    messages = build_messages(user_q, context=args.context, strategy=strategy)
    text = engine.generate_text(messages)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
