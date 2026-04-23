from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from genai_research.config import load_config, with_overrides
from genai_research.gemini_api import GeminiApiError, generate_json_text
from genai_research.model import TinyLlamaEngine
from genai_research.prompting import PromptStrategy, build_messages, resolve_prompt_context

try:
    from datasets import load_dataset  # pyright: ignore[reportMissingImports]
except Exception:
    load_dataset = None  # type: ignore[assignment]


FIELD_ALIASES = {
    "id": ["id", "sample_id", "uuid", "qid"],
    "question": ["question", "query", "prompt", "user_query", "input"],
    "context": ["context", "contexts", "documents", "retrieved_contexts", "evidence"],
    "reference": [
        "answer",
        "answers",
        "ground_truth",
        "reference",
        "references",
        "gold_answer",
    ],
}


@dataclass
class Sample:
    sample_id: str
    question: str
    context: str
    reference: str
    raw: dict[str, Any]


@dataclass
class JudgeDecision:
    winner: str
    hallucination_a: bool
    hallucination_b: bool
    reason: str
    raw_output: str
    parse_ok: bool


def _first_present(record: dict[str, Any], keys: list[str]) -> Any:
    for key in keys:
        if key in record and record[key] not in (None, ""):
            return record[key]
    return None


def _flatten(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        for k in ("text", "content", "passage", "document", "answer"):
            if isinstance(value.get(k), str):
                return value[k].strip()
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            txt = _flatten(item)
            if txt:
                parts.append(txt)
        return "\n\n".join(parts).strip()
    return str(value).strip()


def normalize_record(idx: int, record: dict[str, Any]) -> Sample | None:
    q = _first_present(record, FIELD_ALIASES["question"])
    if not isinstance(q, str) or not q.strip():
        return None
    rid = _first_present(record, FIELD_ALIASES["id"])
    sid = str(rid) if rid is not None else f"row-{idx}"
    return Sample(
        sample_id=sid,
        question=q.strip(),
        context=_flatten(_first_present(record, FIELD_ALIASES["context"])),
        reference=_flatten(_first_present(record, FIELD_ALIASES["reference"])),
        raw=record,
    )


def load_samples_from_local(path: Path) -> list[Sample]:
    rows: list[dict[str, Any]] = []
    if path.suffix.lower() == ".jsonl":
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    elif path.suffix.lower() == ".json":
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            rows = [r for r in data if isinstance(r, dict)]
        elif isinstance(data, dict) and isinstance(data.get("data"), list):
            rows = [r for r in data["data"] if isinstance(r, dict)]
        else:
            raise ValueError("Unsupported JSON shape for --dataset-path")
    else:
        raise ValueError("--dataset-path must point to .json or .jsonl")
    out: list[Sample] = []
    for i, row in enumerate(rows):
        s = normalize_record(i, row)
        if s is not None:
            out.append(s)
    return out


def load_samples_from_hf(dataset_id: str, split: str) -> list[Sample]:
    if load_dataset is None:
        raise RuntimeError("datasets package unavailable.")
    ds = load_dataset(dataset_id, split=split)
    out: list[Sample] = []
    for i, row in enumerate(ds):
        if not isinstance(row, dict):
            continue
        s = normalize_record(i, row)
        if s is not None:
            out.append(s)
    return out


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_dotenv(path: Path) -> None:
    """
    Minimal .env loader (no external dependency required).

    Keys in the file set os.environ[KEY] and override any existing value so a
    filled-in .env is not masked by a stale GEMINI_API_KEY from the shell/IDE.
    """
    if not path.exists():
        return
    with open(path, encoding="utf-8-sig") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[len("export ") :].strip()
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()
            if not key:
                continue
            if value and value[0] == value[-1] and value[0] in {"'", '"'}:
                value = value[1:-1]
            os.environ[key] = value


def load_existing_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                print(f"Warning: malformed JSON line {line_no} in {path}", file=sys.stderr)
                continue
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def _parse_bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    if isinstance(v, str):
        return v.strip().lower() in {"1", "true", "yes", "y"}
    return False


def _extract_json_obj(text: str) -> dict[str, Any] | None:
    text = text.strip()
    if not text:
        return None
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        return None


def judge_pair(
    judge_engine: TinyLlamaEngine | None,
    *,
    judge_backend: str,
    gemini_model: str,
    gemini_api_key: str,
    gemini_timeout_sec: int,
    question: str,
    evidence: str,
    answer_a: str,
    answer_b: str,
) -> JudgeDecision:
    prompt = (
        "Evaluate factuality only.\n"
        "Using QUESTION and EVIDENCE, compare ANSWER_A and ANSWER_B.\n"
        "Return JSON only with keys: winner, hallucination_a, hallucination_b, reason.\n"
        "winner must be one of: A, B, tie.\n\n"
        f"QUESTION:\n{question}\n\n"
        f"EVIDENCE:\n{evidence}\n\n"
        f"ANSWER_A:\n{answer_a}\n\n"
        f"ANSWER_B:\n{answer_b}\n"
    )
    if judge_backend == "gemini":
        try:
            raw = generate_json_text(
                prompt=prompt,
                model=gemini_model,
                api_key=gemini_api_key,
                timeout_sec=gemini_timeout_sec,
            )
        except GeminiApiError as e:
            return JudgeDecision(
                winner="tie",
                hallucination_a=False,
                hallucination_b=False,
                reason=f"gemini_error: {e}",
                raw_output="",
                parse_ok=False,
            )
    else:
        if judge_engine is None:
            return JudgeDecision(
                winner="tie",
                hallucination_a=False,
                hallucination_b=False,
                reason="missing_local_judge_engine",
                raw_output="",
                parse_ok=False,
            )
        messages = build_messages(
            prompt,
            strategy=PromptStrategy.SINGLE,
            system="You are a strict factuality judge. Output valid JSON only.",
        )
        raw = judge_engine.generate_text(
            messages,
            do_sample=False,
            temperature=0.0,
            top_p=1.0,
            max_new_tokens=180,
        )
    obj = _extract_json_obj(raw)
    if obj is None:
        return JudgeDecision(
            winner="tie",
            hallucination_a=False,
            hallucination_b=False,
            reason="parse_failed",
            raw_output=raw,
            parse_ok=False,
        )
    winner = str(obj.get("winner", "tie")).strip().lower()
    if winner in {"a", "answer_a"}:
        w = "A"
    elif winner in {"b", "answer_b"}:
        w = "B"
    else:
        w = "tie"
    return JudgeDecision(
        winner=w,
        hallucination_a=_parse_bool(obj.get("hallucination_a")),
        hallucination_b=_parse_bool(obj.get("hallucination_b")),
        reason=str(obj.get("reason", "")),
        raw_output=raw,
        parse_ok=True,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="With-RAG pipeline using LLM-as-judge metrics only.")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--dataset-path", type=Path, help="Path to local .json/.jsonl dataset.")
    src.add_argument("--hf-dataset", type=str, help="HF dataset id.")
    p.add_argument("--hf-split", type=str, default="test")
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--model-id", type=str, default=None)
    p.add_argument("--judge-model-id", type=str, default=None)
    p.add_argument(
        "--judge-backend",
        choices=["local", "gemini"],
        default="local",
        help="Judge backend: local model or Gemini API.",
    )
    p.add_argument("--gemini-model", type=str, default="gemini-2.0-flash")
    p.add_argument(
        "--gemini-api-key",
        type=str,
        default=None,
        help="Gemini API key; defaults to GEMINI_API_KEY env var.",
    )
    p.add_argument("--gemini-timeout-sec", type=int, default=60)
    p.add_argument("--max-new-tokens", type=int, default=None)
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs" / "llm_judge_with_rag",
    )
    p.add_argument("--resume", action="store_true")
    p.add_argument(
        "--evidence-source",
        choices=["auto", "context", "reference", "concat"],
        default="auto",
        help="Evidence used by judge. auto=context else reference.",
    )
    p.add_argument(
        "--generator-context",
        choices=["auto", "context", "reference", "concat"],
        default="auto",
        help="Which field supplies the RAG passage in the *generator* prompt. "
        "Use 'auto' when the dataset has no 'context' column (passage only in 'reference', e.g. LibreEval).",
    )
    p.add_argument(
        "--print-model-prompts",
        choices=["off", "first", "all"],
        default="first",
        help="Print the exact chat-templated string sent to the generator (stderr). "
        "Default 'first': one sample's standard_rag + sandwich prompts. Use 'all' for every sample.",
    )
    return p.parse_args()


def _print_generator_prompt(
    gen_engine: TinyLlamaEngine, *, sample_id: str, label: str, messages: list[dict[str, str]]
) -> None:
    """Log the same prompt `format_chat` / `generate_text` uses (chat template applied)."""
    text = gen_engine.render_chat_prompt(messages)
    sep = "=" * 72
    print(f"\n{sep}\n[{label}] sample_id={sample_id}\n{sep}\n{text}\n", file=sys.stderr)


def _evidence_for_judge(sample: Sample, source: str) -> str:
    if source == "context":
        return sample.context
    if source == "reference":
        return sample.reference
    if source == "concat":
        return "\n\n".join([x for x in [sample.context, sample.reference] if x.strip()])
    # auto
    return sample.context if sample.context.strip() else sample.reference


def run() -> int:
    load_dotenv(ROOT / ".env")
    args = parse_args()
    if args.dataset_path:
        samples = load_samples_from_local(args.dataset_path)
    else:
        samples = load_samples_from_hf(args.hf_dataset, args.hf_split)
    if args.max_samples is not None:
        samples = samples[: args.max_samples]
    if not samples:
        raise RuntimeError("No valid samples loaded.")

    base_cfg = with_overrides(
        load_config(args.config),
        model_id=args.model_id,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )
    gen_engine = TinyLlamaEngine(base_cfg)
    judge_engine: TinyLlamaEngine | None = gen_engine
    if args.judge_backend == "gemini":
        judge_engine = None
    elif args.judge_model_id and args.judge_model_id != base_cfg.model.model_id:
        judge_cfg = with_overrides(load_config(args.config), model_id=args.judge_model_id)
        judge_engine = TinyLlamaEngine(judge_cfg)

    gemini_api_key = (args.gemini_api_key or os.environ.get("GEMINI_API_KEY", "")).strip()
    if args.judge_backend == "gemini" and not gemini_api_key:
        raise RuntimeError("Gemini judge backend selected but GEMINI_API_KEY is missing.")

    ensure_dir(args.output_dir)
    pred_path = args.output_dir / "predictions.jsonl"
    summary_path = args.output_dir / "summary.json"

    existing_rows: list[dict[str, Any]] = []
    completed_ids: set[str] = set()
    if args.resume:
        existing_rows = load_existing_rows(pred_path)
        for r in existing_rows:
            sid = r.get("sample_id")
            if sid is not None:
                completed_ids.add(str(sid))
        if completed_ids:
            samples = [s for s in samples if s.sample_id not in completed_ids]
            print(
                f"Resume: loaded {len(existing_rows)} rows, skipping {len(completed_ids)} samples.",
                file=sys.stderr,
            )
    elif pred_path.exists():
        pred_path.unlink()

    rows: list[dict[str, Any]] = list(existing_rows)
    from tqdm import tqdm

    for i, s in enumerate(tqdm(samples, desc="Evaluating", unit="sample")):
        ctx_in = resolve_prompt_context(
            s.context, s.reference, source=args.generator_context
        )
        msg_std = build_messages(s.question, context=ctx_in, strategy=PromptStrategy.SINGLE)
        msg_sand = build_messages(s.question, context=ctx_in, strategy=PromptStrategy.SANDWICH)
        pp = args.print_model_prompts
        if pp == "all" or (pp == "first" and i == 0):
            _print_generator_prompt(
                gen_engine, sample_id=s.sample_id, label="standard_rag (SINGLE)", messages=msg_std
            )
            _print_generator_prompt(
                gen_engine, sample_id=s.sample_id, label="sandwich (SANDWICH)", messages=msg_sand
            )
        pred_std = gen_engine.generate_text(msg_std)
        pred_sand = gen_engine.generate_text(msg_sand)

        # blind order per sample
        swap = int(hashlib.md5(s.sample_id.encode("utf-8")).hexdigest(), 16) % 2 == 1
        a = pred_sand if swap else pred_std
        b = pred_std if swap else pred_sand
        decision = judge_pair(
            judge_engine,
            judge_backend=args.judge_backend,
            gemini_model=args.gemini_model,
            gemini_api_key=gemini_api_key,
            gemini_timeout_sec=args.gemini_timeout_sec,
            question=s.question,
            evidence=_evidence_for_judge(s, args.evidence_source),
            answer_a=a,
            answer_b=b,
        )

        winner = decision.winner
        hall_a = decision.hallucination_a
        hall_b = decision.hallucination_b
        if swap:
            # map back to standard/sandwich
            if winner == "A":
                winner = "B"
            elif winner == "B":
                winner = "A"
            hall_a, hall_b = hall_b, hall_a

        winner_label = "tie"
        if winner == "A":
            winner_label = "standard_rag"
        elif winner == "B":
            winner_label = "sandwich"

        row = {
            "sample_id": s.sample_id,
            "question": s.question,
            "context": s.context,
            "reference": s.reference,
            "prompt_context": ctx_in,
            "prediction_standard_rag": pred_std,
            "prediction_sandwich": pred_sand,
            "judge": {
                "winner": winner_label,
                "hallucination_standard_rag": hall_a,
                "hallucination_sandwich": hall_b,
                "reason": decision.reason,
                "parse_ok": decision.parse_ok,
                "raw_output": decision.raw_output,
            },
        }
        rows.append(row)
        with open(pred_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    total = len(rows) or 1
    wins_std = sum(1 for r in rows if r["judge"]["winner"] == "standard_rag")
    wins_sand = sum(1 for r in rows if r["judge"]["winner"] == "sandwich")
    ties = sum(1 for r in rows if r["judge"]["winner"] == "tie")
    hall_std = sum(1 for r in rows if r["judge"]["hallucination_standard_rag"])
    hall_sand = sum(1 for r in rows if r["judge"]["hallucination_sandwich"])
    parse_ok_rate = mean(float(r["judge"]["parse_ok"]) for r in rows) if rows else 0.0

    summary = {
        "num_samples": len(rows),
        "model_id": base_cfg.model.model_id,
        "judge_backend": args.judge_backend,
        "judge_model_id": (
            args.gemini_model
            if args.judge_backend == "gemini"
            else (args.judge_model_id or base_cfg.model.model_id)
        ),
        "generation": asdict(base_cfg.generation),
        "duplication_strategy": "sandwich",
        "evidence_source": args.evidence_source,
        "generator_context": args.generator_context,
        "aggregate": {
            "standard_rag_win_rate": wins_std / total,
            "sandwich_win_rate": wins_sand / total,
            "tie_rate": ties / total,
            "hallucination_rate_standard_rag": hall_std / total,
            "hallucination_rate_sandwich": hall_sand / total,
            "hallucination_delta_sandwich_minus_standard": (hall_sand - hall_std) / total,
            "win_rate_delta_sandwich_minus_standard": (wins_sand - wins_std) / total,
            "judge_parse_ok_rate": parse_ok_rate,
        },
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"Wrote {pred_path}")
    print(f"Wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())

