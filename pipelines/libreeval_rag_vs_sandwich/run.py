from __future__ import annotations

import argparse
import json
import re
import string
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Any

import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from genai_research.config import load_config, with_overrides
from genai_research.model import TinyLlamaEngine
from genai_research.prompting import PromptStrategy, build_messages, resolve_prompt_context
from transformers import AutoModelForSequenceClassification, AutoTokenizer

try:
    from datasets import Dataset, load_dataset  # pyright: ignore[reportMissingImports]
except Exception:  # pragma: no cover - optional runtime import
    Dataset = Any  # type: ignore[assignment]
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

DEFAULT_NLI_MODEL_ID = "valhalla/distilbart-mnli-12-3"


@dataclass
class Sample:
    sample_id: str
    question: str
    context: str
    reference: str
    raw: dict[str, Any]


@dataclass
class NLIConfig:
    enabled: bool = True
    model_id: str = DEFAULT_NLI_MODEL_ID
    max_claims: int = 8
    max_length: int = 384
    batch_size: int = 8
    premise_source: str = "auto"


class NLIHallucinationChecker:
    """
    Lightweight claim-support checker using MNLI-style entailment labels.

    Premise: retrieved context
    Hypothesis: answer claim sentence
    """

    def __init__(self, cfg: NLIConfig) -> None:
        self.cfg = cfg
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_id)
        self.model = AutoModelForSequenceClassification.from_pretrained(cfg.model_id)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        self._label_map = {
            int(k): str(v).lower()
            for k, v in getattr(self.model.config, "id2label", {}).items()
        }
        self._idx_entailment = self._find_label_index("entail")
        self._idx_contradiction = self._find_label_index("contra")

    def _find_label_index(self, needle: str) -> int | None:
        for idx, name in self._label_map.items():
            if needle in name:
                return idx
        # Common MNLI fallback ordering: contradiction, neutral, entailment
        if len(self._label_map) == 3 and needle == "entail":
            return 2
        if len(self._label_map) == 3 and needle == "contra":
            return 0
        return None

    @staticmethod
    def _split_claims(text: str, max_claims: int) -> list[str]:
        parts = re.split(r"(?<=[.!?])\s+|\n+", text.strip())
        claims = [p.strip() for p in parts if p.strip()]
        return claims[:max_claims]

    def _score_claim_batch(self, context: str, claims: list[str]) -> list[dict[str, float]]:
        if not claims:
            return []
        enc = self.tokenizer(
            [context] * len(claims),
            claims,
            padding=True,
            truncation=True,
            max_length=self.cfg.max_length,
            return_tensors="pt",
        ).to(self.device)
        with torch.inference_mode():
            logits = self.model(**enc).logits
            probs = torch.softmax(logits, dim=-1)
        rows: list[dict[str, float]] = []
        for row in probs:
            entail = float(row[self._idx_entailment].item()) if self._idx_entailment is not None else 0.0
            contra = (
                float(row[self._idx_contradiction].item())
                if self._idx_contradiction is not None
                else 0.0
            )
            neutral = max(0.0, 1.0 - entail - contra)
            rows.append(
                {
                    "entailment_prob": entail,
                    "contradiction_prob": contra,
                    "neutral_prob": neutral,
                }
            )
        return rows

    def score_answer(self, context: str, answer: str) -> dict[str, float]:
        if not context.strip() or not answer.strip():
            return {
                "nli_num_claims": 0.0,
                "nli_supported_claim_rate": 0.0,
                "nli_contradiction_claim_rate": 0.0,
                "nli_neutral_claim_rate": 0.0,
                "nli_hallucination_claim_rate": 0.0,
                "nli_mean_entailment_prob": 0.0,
            }

        claims = self._split_claims(answer, self.cfg.max_claims)
        if not claims:
            return {
                "nli_num_claims": 0.0,
                "nli_supported_claim_rate": 0.0,
                "nli_contradiction_claim_rate": 0.0,
                "nli_neutral_claim_rate": 0.0,
                "nli_hallucination_claim_rate": 0.0,
                "nli_mean_entailment_prob": 0.0,
            }

        all_scores: list[dict[str, float]] = []
        for i in range(0, len(claims), self.cfg.batch_size):
            chunk = claims[i : i + self.cfg.batch_size]
            all_scores.extend(self._score_claim_batch(context, chunk))

        supported = sum(float(s["entailment_prob"] >= 0.5) for s in all_scores)
        contrad = sum(float(s["contradiction_prob"] >= 0.5) for s in all_scores)
        neutral = max(0.0, float(len(all_scores)) - supported - contrad)
        denom = float(len(all_scores))
        mean_entail = mean(s["entailment_prob"] for s in all_scores)
        return {
            "nli_num_claims": denom,
            "nli_supported_claim_rate": supported / denom,
            "nli_contradiction_claim_rate": contrad / denom,
            "nli_neutral_claim_rate": neutral / denom,
            "nli_hallucination_claim_rate": 1.0 - (supported / denom),
            "nli_mean_entailment_prob": mean_entail,
        }


def _first_present(record: dict[str, Any], keys: list[str]) -> Any:
    for key in keys:
        if key in record and record[key] not in (None, ""):
            return record[key]
    return None


def _flatten_context(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        for k in ("text", "content", "passage", "document"):
            if isinstance(value.get(k), str):
                return value[k].strip()
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            text = _flatten_context(item)
            if text:
                parts.append(text)
        return "\n\n".join(parts).strip()
    return str(value).strip()


def _flatten_reference(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        # keep one canonical target for simple lexical metrics
        first = next((v for v in value if v not in (None, "")), "")
        return _flatten_reference(first)
    if isinstance(value, dict):
        for k in ("text", "answer", "content"):
            if isinstance(value.get(k), str):
                return value[k].strip()
        return json.dumps(value, ensure_ascii=False)
    return str(value).strip()


def normalize_record(idx: int, record: dict[str, Any]) -> Sample | None:
    question = _first_present(record, FIELD_ALIASES["question"])
    if not isinstance(question, str) or not question.strip():
        return None

    rid = _first_present(record, FIELD_ALIASES["id"])
    sample_id = str(rid) if rid is not None else f"row-{idx}"
    context_raw = _first_present(record, FIELD_ALIASES["context"])
    reference_raw = _first_present(record, FIELD_ALIASES["reference"])

    return Sample(
        sample_id=sample_id,
        question=question.strip(),
        context=_flatten_context(context_raw),
        reference=_flatten_reference(reference_raw),
        raw=record,
    )


def load_samples_from_local(path: Path) -> list[Sample]:
    rows: list[dict[str, Any]] = []
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    elif suffix == ".json":
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            rows = [r for r in data if isinstance(r, dict)]
        elif isinstance(data, dict):
            if "data" in data and isinstance(data["data"], list):
                rows = [r for r in data["data"] if isinstance(r, dict)]
            else:
                raise ValueError("JSON must be a list of records or contain a `data` list")
        else:
            raise ValueError("Unsupported JSON structure")
    else:
        raise ValueError("Only .json and .jsonl are supported for --dataset-path")

    out: list[Sample] = []
    for idx, row in enumerate(rows):
        sample = normalize_record(idx, row)
        if sample is not None:
            out.append(sample)
    return out


def load_samples_from_hf(dataset_name: str, split: str) -> list[Sample]:
    if load_dataset is None:
        raise RuntimeError(
            "Hugging Face datasets is unavailable. Install dependencies with pip install -e ."
        )
    ds: Dataset = load_dataset(dataset_name, split=split)  # type: ignore[assignment]
    out: list[Sample] = []
    for idx, row in enumerate(ds):
        if not isinstance(row, dict):
            continue
        sample = normalize_record(idx, row)
        if sample is not None:
            out.append(sample)
    return out


_WS = re.compile(r"\s+")
_PUNC_TABLE = str.maketrans("", "", string.punctuation)


def _norm_text(s: str) -> str:
    s = s.lower().translate(_PUNC_TABLE)
    return _WS.sub(" ", s).strip()


def exact_match(pred: str, ref: str) -> float:
    if not ref.strip():
        return 0.0
    return float(_norm_text(pred) == _norm_text(ref))


def token_f1(pred: str, ref: str) -> float:
    p = _norm_text(pred).split()
    r = _norm_text(ref).split()
    if not p or not r:
        return 0.0
    cp = Counter(p)
    cr = Counter(r)
    common = sum((cp & cr).values())
    if common == 0:
        return 0.0
    precision = common / len(p)
    recall = common / len(r)
    return 2 * precision * recall / (precision + recall)


def context_overlap(pred: str, context: str) -> float:
    pt = set(_norm_text(pred).split())
    ct = set(_norm_text(context).split())
    if not pt:
        return 0.0
    if not ct:
        return 0.0
    return len(pt.intersection(ct)) / len(pt)


def refusal_flag(pred: str) -> float:
    value = _norm_text(pred)
    phrases = [
        "i do not know",
        "i dont know",
        "cannot determine",
        "not enough information",
        "insufficient information",
    ]
    return float(any(p in value for p in phrases))


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


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
                # Keep resume resilient: skip corrupted tail lines from interrupted writes.
                print(
                    f"Warning: skipping invalid JSON line {line_no} in {path}",
                    file=sys.stderr,
                )
                continue
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def select_nli_premise(sample: Sample, source: str) -> str:
    """
    Select premise text for NLI claim checking.

    - auto: context, else reference
    - context: context only
    - reference: reference only
    - output: dataset output field from raw record (if present)
    """
    key = source.lower().strip()
    if key == "auto":
        return sample.context if sample.context.strip() else sample.reference
    if key == "context":
        return sample.context
    if key == "reference":
        return sample.reference
    if key == "output":
        value = sample.raw.get("output")
        return _flatten_reference(value)
    raise ValueError(f"Unsupported nli premise source: {source!r}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compare standard RAG vs Sandwich prompting on LibreEval-style data."
    )
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--dataset-path", type=Path, help="Path to local .json or .jsonl file.")
    source.add_argument(
        "--hf-dataset",
        type=str,
        help="HF dataset id (e.g. org/libreeval1.0). Use with --hf-split.",
    )
    p.add_argument("--hf-split", type=str, default="test", help="HF split name.")
    p.add_argument("--config", type=Path, default=None, help="Optional model YAML config.")
    p.add_argument("--model-id", type=str, default=None, help="Override model id.")
    p.add_argument("--max-new-tokens", type=int, default=None)
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--system-prompt", type=str, default=None)
    p.add_argument(
        "--disable-nli-checker",
        action="store_true",
        help="Disable second-stage NLI hallucination checker.",
    )
    p.add_argument(
        "--nli-model-id",
        type=str,
        default=DEFAULT_NLI_MODEL_ID,
        help="NLI model id used for claim support checks.",
    )
    p.add_argument(
        "--nli-max-claims",
        type=int,
        default=8,
        help="Maximum number of answer claims (sentences) to score per answer.",
    )
    p.add_argument(
        "--nli-max-length",
        type=int,
        default=384,
        help="Max sequence length for NLI pair tokenization.",
    )
    p.add_argument(
        "--nli-batch-size",
        type=int,
        default=8,
        help="Batch size for NLI inference over claims.",
    )
    p.add_argument(
        "--nli-premise-source",
        type=str,
        default="auto",
        choices=["auto", "context", "reference", "output"],
        help="Premise source for NLI checks (auto uses context, else reference).",
    )
    p.add_argument(
        "--generator-context",
        choices=["auto", "context", "reference", "concat"],
        default="auto",
        help="RAG passage for the generator. 'auto' uses context if set, else reference (HF libreeval: passage in 'reference' only).",
    )
    p.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit rows for quick smoke tests.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs" / "libreeval_rag_vs_sandwich",
        help="Directory for predictions and summary.",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing predictions.jsonl in output dir.",
    )
    return p.parse_args()


def run() -> int:
    args = parse_args()

    if args.dataset_path:
        samples = load_samples_from_local(args.dataset_path)
    else:
        samples = load_samples_from_hf(args.hf_dataset, args.hf_split)

    if args.max_samples is not None:
        samples = samples[: args.max_samples]
    if not samples:
        raise RuntimeError("No valid samples found after schema normalization.")

    cfg = load_config(args.config)
    cfg = with_overrides(
        cfg,
        model_id=args.model_id,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )
    engine = TinyLlamaEngine(cfg)
    nli_cfg = NLIConfig(
        enabled=not args.disable_nli_checker,
        model_id=args.nli_model_id,
        max_claims=args.nli_max_claims,
        max_length=args.nli_max_length,
        batch_size=args.nli_batch_size,
        premise_source=args.nli_premise_source,
    )
    nli_checker = NLIHallucinationChecker(nli_cfg) if nli_cfg.enabled else None

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
                f"Resume mode: loaded {len(existing_rows)} rows, skipping {len(completed_ids)} completed samples.",
                file=sys.stderr,
            )
    elif pred_path.exists():
        pred_path.unlink()

    rows: list[dict[str, Any]] = list(existing_rows)
    for sample in tqdm(samples, desc="Evaluating", unit="sample"):
        ctx_in = resolve_prompt_context(
            sample.context, sample.reference, source=args.generator_context
        )
        std_messages = build_messages(
            sample.question,
            context=ctx_in,
            strategy=PromptStrategy.SINGLE,
            system=args.system_prompt,
        )
        sandwich_messages = build_messages(
            sample.question,
            context=ctx_in,
            strategy=PromptStrategy.SANDWICH,
            system=args.system_prompt,
        )

        pred_standard = engine.generate_text(std_messages)
        pred_sandwich = engine.generate_text(sandwich_messages)

        standard_metrics = {
            "exact_match": exact_match(pred_standard, sample.reference),
            "token_f1": token_f1(pred_standard, sample.reference),
            "context_overlap": context_overlap(pred_standard, ctx_in),
            "refusal_rate": refusal_flag(pred_standard),
        }
        sandwich_metrics = {
            "exact_match": exact_match(pred_sandwich, sample.reference),
            "token_f1": token_f1(pred_sandwich, sample.reference),
            "context_overlap": context_overlap(pred_sandwich, ctx_in),
            "refusal_rate": refusal_flag(pred_sandwich),
        }
        if nli_checker is not None:
            nli_premise = select_nli_premise(sample, nli_cfg.premise_source)
            standard_metrics.update(nli_checker.score_answer(nli_premise, pred_standard))
            sandwich_metrics.update(nli_checker.score_answer(nli_premise, pred_sandwich))
        row = {
            "sample_id": sample.sample_id,
            "question": sample.question,
            "reference": sample.reference,
            "context": sample.context,
            "prompt_context": ctx_in,
            "nli_premise_source": nli_cfg.premise_source,
            "prediction_standard_rag": pred_standard,
            "prediction_sandwich": pred_sandwich,
            "metrics": {
                "standard_rag": standard_metrics,
                "sandwich": sandwich_metrics,
                "delta_sandwich_minus_standard": {
                    k: sandwich_metrics[k] - standard_metrics[k]
                    for k in standard_metrics.keys()
                },
            },
        }
        rows.append(row)
        with open(pred_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def avg(path: str, variant: str) -> float:
        parts = path.split(".")
        vals: list[float] = []
        for r in rows:
            obj: Any = r["metrics"][variant]
            for p in parts:
                obj = obj[p]
            vals.append(float(obj))
        return mean(vals) if vals else 0.0

    summary = {
        "num_samples": len(rows),
        "model_id": cfg.model.model_id,
        "generation": asdict(cfg.generation),
        "generator_context": args.generator_context,
        "nli_checker": asdict(nli_cfg),
        "aggregate": {
            "standard_rag": {
                "exact_match": avg("exact_match", "standard_rag"),
                "token_f1": avg("token_f1", "standard_rag"),
                "context_overlap": avg("context_overlap", "standard_rag"),
                "refusal_rate": avg("refusal_rate", "standard_rag"),
            },
            "sandwich": {
                "exact_match": avg("exact_match", "sandwich"),
                "token_f1": avg("token_f1", "sandwich"),
                "context_overlap": avg("context_overlap", "sandwich"),
                "refusal_rate": avg("refusal_rate", "sandwich"),
            },
        },
    }
    if nli_checker is not None:
        for variant in ("standard_rag", "sandwich"):
            summary["aggregate"][variant]["nli_supported_claim_rate"] = avg(
                "nli_supported_claim_rate", variant
            )
            summary["aggregate"][variant]["nli_contradiction_claim_rate"] = avg(
                "nli_contradiction_claim_rate", variant
            )
            summary["aggregate"][variant]["nli_neutral_claim_rate"] = avg(
                "nli_neutral_claim_rate", variant
            )
            summary["aggregate"][variant]["nli_hallucination_claim_rate"] = avg(
                "nli_hallucination_claim_rate", variant
            )
            summary["aggregate"][variant]["nli_mean_entailment_prob"] = avg(
                "nli_mean_entailment_prob", variant
            )
            summary["aggregate"][variant]["nli_num_claims"] = avg("nli_num_claims", variant)
    summary["aggregate"]["delta_sandwich_minus_standard"] = {
        key: summary["aggregate"]["sandwich"][key] - summary["aggregate"]["standard_rag"][key]
        for key in summary["aggregate"]["standard_rag"].keys()
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"Wrote {pred_path}")
    print(f"Wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())

