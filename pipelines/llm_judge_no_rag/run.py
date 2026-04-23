from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import sys
from dataclasses import asdict
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
from genai_research.prompting import PromptStrategy, build_messages

try:
    from datasets import load_dataset  # pyright: ignore[reportMissingImports]
except Exception:
    load_dataset = None  # type: ignore[assignment]


DATASET_SPECS: dict[str, dict[str, str]] = {
    "nq_open": {
        "hf_id": "nq_open",
        "hf_config": "nq_open",
        "split": "validation",
    },
    "truthfulqa": {
        "hf_id": "truthfulqa/truthful_qa",
        "hf_config": "generation",
        "split": "validation",
    },
    "popqa": {
        "hf_id": "akariasai/PopQA",
        "hf_config": "",
        "split": "test",
    },
}


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
        for ln, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                print(f"Warning: malformed JSON line {ln} in {path}", file=sys.stderr)
                continue
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        txt = value.strip()
        if txt.startswith("[") and txt.endswith("]"):
            try:
                arr = json.loads(txt)
                if isinstance(arr, list):
                    return [str(x).strip() for x in arr if str(x).strip()]
            except json.JSONDecodeError:
                try:
                    arr = ast.literal_eval(txt)
                    if isinstance(arr, list):
                        return [str(x).strip() for x in arr if str(x).strip()]
                except Exception:
                    pass
        return [txt] if txt else []
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    return [str(value).strip()]


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for v in values:
        key = v.strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(v.strip())
    return out


def load_samples(dataset_key: str, split_override: str | None = None) -> list[dict[str, Any]]:
    if load_dataset is None:
        raise RuntimeError("datasets package unavailable.")
    spec = DATASET_SPECS[dataset_key]
    cfg = spec["hf_config"] or None
    split = split_override or spec["split"]
    ds = load_dataset(spec["hf_id"], cfg, split=split)
    out: list[dict[str, Any]] = []
    for i, row in enumerate(ds):
        if dataset_key == "truthfulqa":
            q = str(row.get("question", "")).strip()
            refs = _dedupe(_as_list(row.get("best_answer")) + _as_list(row.get("correct_answers")))
        elif dataset_key == "popqa":
            q = str(row.get("question", "")).strip()
            refs = _dedupe(_as_list(row.get("possible_answers")))
        else:
            q = str(row.get("question", "")).strip()
            refs = _dedupe(_as_list(row.get("answer")))
        if not q or not refs:
            continue
        out.append({"sample_id": f"{dataset_key}-{i}", "question": q, "references": refs})
    return out


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
) -> dict[str, Any]:
    prompt = (
        "Evaluate factuality only.\n"
        "Given QUESTION and REFERENCES, compare ANSWER_A and ANSWER_B.\n"
        "Return JSON only with keys: winner, hallucination_a, hallucination_b, reason.\n"
        "winner must be one of: A, B, tie.\n\n"
        f"QUESTION:\n{question}\n\n"
        f"REFERENCES:\n{evidence}\n\n"
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
            return {
                "winner": "tie",
                "hallucination_a": False,
                "hallucination_b": False,
                "reason": f"gemini_error: {e}",
                "raw_output": "",
                "parse_ok": False,
            }
    else:
        if judge_engine is None:
            return {
                "winner": "tie",
                "hallucination_a": False,
                "hallucination_b": False,
                "reason": "missing_local_judge_engine",
                "raw_output": "",
                "parse_ok": False,
            }
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
        return {
            "winner": "tie",
            "hallucination_a": False,
            "hallucination_b": False,
            "reason": "parse_failed",
            "raw_output": raw,
            "parse_ok": False,
        }
    w = str(obj.get("winner", "tie")).strip().lower()
    winner = "A" if w in {"a", "answer_a"} else ("B" if w in {"b", "answer_b"} else "tie")
    return {
        "winner": winner,
        "hallucination_a": _parse_bool(obj.get("hallucination_a")),
        "hallucination_b": _parse_bool(obj.get("hallucination_b")),
        "reason": str(obj.get("reason", "")),
        "raw_output": raw,
        "parse_ok": True,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="No-RAG pipeline with LLM-as-judge metrics only.")
    p.add_argument(
        "--dataset",
        choices=["nq_open", "truthfulqa", "popqa", "all"],
        default="truthfulqa",
    )
    p.add_argument("--split", type=str, default=None)
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
    p.add_argument(
        "--duplication-strategy",
        choices=[PromptStrategy.DOUBLE_QUERY.value, PromptStrategy.REPEAT_USER_BLOCK.value],
        default=PromptStrategy.DOUBLE_QUERY.value,
    )
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "llm_judge_no_rag")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--reference-policy", choices=["first", "concat"], default="first")
    p.add_argument(
        "--print-model-prompts",
        choices=["off", "first", "all"],
        default="first",
        help="Print the exact chat-templated generator input to stderr. "
        "'first': one sample per dataset (single + duplicated). 'all': every sample.",
    )
    return p.parse_args()


def _print_generator_prompt(
    gen_engine: TinyLlamaEngine, *, sample_id: str, label: str, messages: list[dict[str, str]]
) -> None:
    """Log the same prompt `format_chat` / `generate_text` uses (chat template applied)."""
    text = gen_engine.render_chat_prompt(messages)
    sep = "=" * 72
    print(f"\n{sep}\n[{label}] sample_id={sample_id}\n{sep}\n{text}\n", file=sys.stderr)


def evaluate_dataset(
    dataset_key: str,
    args: argparse.Namespace,
    gen_engine: TinyLlamaEngine,
    judge_engine: TinyLlamaEngine | None,
    gemini_api_key: str,
) -> dict[str, Any]:
    samples = load_samples(dataset_key, args.split)
    if args.max_samples is not None:
        samples = samples[: args.max_samples]
    if not samples:
        raise RuntimeError(f"No valid rows loaded for {dataset_key}.")

    out_dir = args.output_dir / dataset_key
    ensure_dir(out_dir)
    pred_path = out_dir / "predictions.jsonl"
    summary_path = out_dir / "summary.json"

    existing_rows: list[dict[str, Any]] = []
    completed_ids: set[str] = set()
    if args.resume:
        existing_rows = load_existing_rows(pred_path)
        for r in existing_rows:
            sid = r.get("sample_id")
            if sid is not None:
                completed_ids.add(str(sid))
        if completed_ids:
            samples = [s for s in samples if s["sample_id"] not in completed_ids]
            print(
                f"[{dataset_key}] Resume: loaded {len(existing_rows)}, skipping {len(completed_ids)}.",
                file=sys.stderr,
            )
    elif pred_path.exists():
        pred_path.unlink()

    rows: list[dict[str, Any]] = list(existing_rows)
    from tqdm import tqdm

    dup_strategy = PromptStrategy(args.duplication_strategy)
    dup_label = f"duplicated ({dup_strategy.value})"
    for i, s in enumerate(tqdm(samples, desc=f"Evaluating {dataset_key}", unit="sample")):
        q = s["question"]
        refs = s["references"]
        ref_text = refs[0] if args.reference_policy == "first" else "\n".join(refs)

        msg_single = build_messages(q, strategy=PromptStrategy.SINGLE)
        msg_dup = build_messages(q, strategy=dup_strategy)
        pp = args.print_model_prompts
        if pp == "all" or (pp == "first" and i == 0):
            _print_generator_prompt(
                gen_engine, sample_id=s["sample_id"], label="single (SINGLE)", messages=msg_single
            )
            _print_generator_prompt(
                gen_engine, sample_id=s["sample_id"], label=dup_label, messages=msg_dup
            )
        pred_single = gen_engine.generate_text(msg_single)
        pred_dup = gen_engine.generate_text(msg_dup)

        swap = int(hashlib.md5(s["sample_id"].encode("utf-8")).hexdigest(), 16) % 2 == 1
        a = pred_dup if swap else pred_single
        b = pred_single if swap else pred_dup
        decision = judge_pair(
            judge_engine,
            judge_backend=args.judge_backend,
            gemini_model=args.gemini_model,
            gemini_api_key=gemini_api_key,
            gemini_timeout_sec=args.gemini_timeout_sec,
            question=q,
            evidence=ref_text,
            answer_a=a,
            answer_b=b,
        )

        winner = decision["winner"]
        hall_a = decision["hallucination_a"]
        hall_b = decision["hallucination_b"]
        if swap:
            if winner == "A":
                winner = "B"
            elif winner == "B":
                winner = "A"
            hall_a, hall_b = hall_b, hall_a

        winner_label = "tie"
        if winner == "A":
            winner_label = "single"
        elif winner == "B":
            winner_label = "duplicated"

        row = {
            "sample_id": s["sample_id"],
            "dataset": dataset_key,
            "question": q,
            "references": refs,
            "prediction_single": pred_single,
            "prediction_duplicated": pred_dup,
            "duplication_strategy": dup_strategy.value,
            "judge": {
                "winner": winner_label,
                "hallucination_single": hall_a,
                "hallucination_duplicated": hall_b,
                "reason": decision["reason"],
                "parse_ok": decision["parse_ok"],
                "raw_output": decision["raw_output"],
            },
        }
        rows.append(row)
        with open(pred_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    total = len(rows) or 1
    wins_single = sum(1 for r in rows if r["judge"]["winner"] == "single")
    wins_dup = sum(1 for r in rows if r["judge"]["winner"] == "duplicated")
    ties = sum(1 for r in rows if r["judge"]["winner"] == "tie")
    hall_single = sum(1 for r in rows if r["judge"]["hallucination_single"])
    hall_dup = sum(1 for r in rows if r["judge"]["hallucination_duplicated"])
    parse_ok_rate = mean(float(r["judge"]["parse_ok"]) for r in rows) if rows else 0.0

    summary = {
        "dataset": dataset_key,
        "num_samples": len(rows),
        "model_id": gen_engine.config.model.model_id,
        "judge_backend": args.judge_backend,
        "judge_model_id": (
            args.gemini_model
            if args.judge_backend == "gemini"
            else (args.judge_model_id or gen_engine.config.model.model_id)
        ),
        "generation": asdict(gen_engine.config.generation),
        "duplication_strategy": dup_strategy.value,
        "reference_policy": args.reference_policy,
        "aggregate": {
            "single_win_rate": wins_single / total,
            "duplicated_win_rate": wins_dup / total,
            "tie_rate": ties / total,
            "hallucination_rate_single": hall_single / total,
            "hallucination_rate_duplicated": hall_dup / total,
            "hallucination_delta_duplicated_minus_single": (hall_dup - hall_single) / total,
            "win_rate_delta_duplicated_minus_single": (wins_dup - wins_single) / total,
            "judge_parse_ok_rate": parse_ok_rate,
        },
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[{dataset_key}] wrote {pred_path}")
    print(f"[{dataset_key}] wrote {summary_path}")
    return summary


def run() -> int:
    load_dotenv(ROOT / ".env")
    args = parse_args()
    cfg = with_overrides(
        load_config(args.config),
        model_id=args.model_id,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )
    gen_engine = TinyLlamaEngine(cfg)
    judge_engine: TinyLlamaEngine | None = gen_engine
    if args.judge_backend == "gemini":
        judge_engine = None
    elif args.judge_model_id and args.judge_model_id != cfg.model.model_id:
        judge_cfg = with_overrides(load_config(args.config), model_id=args.judge_model_id)
        judge_engine = TinyLlamaEngine(judge_cfg)

    gemini_api_key = (args.gemini_api_key or os.environ.get("GEMINI_API_KEY", "")).strip()
    if args.judge_backend == "gemini" and not gemini_api_key:
        raise RuntimeError("Gemini judge backend selected but GEMINI_API_KEY is missing.")

    keys = list(DATASET_SPECS.keys()) if args.dataset == "all" else [args.dataset]
    combined: dict[str, Any] = {}
    for key in keys:
        combined[key] = evaluate_dataset(key, args, gen_engine, judge_engine, gemini_api_key)

    if args.dataset == "all":
        ensure_dir(args.output_dir)
        path = args.output_dir / "summary_all.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(combined, f, ensure_ascii=False, indent=2)
        print(f"[all] wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())

