from __future__ import annotations

import argparse
import ast
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
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from genai_research.config import load_config, with_overrides
from genai_research.model import TinyLlamaEngine
from genai_research.prompting import PromptStrategy, build_messages

try:
    from datasets import load_dataset  # pyright: ignore[reportMissingImports]
except Exception:
    load_dataset = None  # type: ignore[assignment]

DEFAULT_NLI_MODEL_ID = "valhalla/distilbart-mnli-12-3"

DATASET_SPECS: dict[str, dict[str, str]] = {
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
    "nq_open": {
        "hf_id": "nq_open",
        "hf_config": "nq_open",
        "split": "validation",
    },
}


@dataclass
class EvalSample:
    sample_id: str
    question: str
    references: list[str]
    metadata: dict[str, Any]


@dataclass
class NLIConfig:
    enabled: bool = True
    model_id: str = DEFAULT_NLI_MODEL_ID
    max_claims: int = 6
    max_length: int = 256
    batch_size: int = 16


class NLIHallucinationChecker:
    def __init__(self, cfg: NLIConfig) -> None:
        self.cfg = cfg
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_id)
        self.model = AutoModelForSequenceClassification.from_pretrained(cfg.model_id)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        label_map = {int(k): str(v).lower() for k, v in self.model.config.id2label.items()}
        self._idx_entailment = self._pick_label_idx(label_map, "entail")
        self._idx_contradiction = self._pick_label_idx(label_map, "contra")

    @staticmethod
    def _pick_label_idx(label_map: dict[int, str], needle: str) -> int | None:
        for idx, name in label_map.items():
            if needle in name:
                return idx
        if len(label_map) == 3 and needle == "entail":
            return 2
        if len(label_map) == 3 and needle == "contra":
            return 0
        return None

    @staticmethod
    def _split_claims(text: str, max_claims: int) -> list[str]:
        chunks = re.split(r"(?<=[.!?])\s+|\n+", text.strip())
        claims = [c.strip() for c in chunks if c.strip()]
        return claims[:max_claims]

    def _score_batch(self, premise: str, claims: list[str]) -> list[dict[str, float]]:
        enc = self.tokenizer(
            [premise] * len(claims),
            claims,
            truncation=True,
            padding=True,
            max_length=self.cfg.max_length,
            return_tensors="pt",
        ).to(self.device)
        with torch.inference_mode():
            probs = torch.softmax(self.model(**enc).logits, dim=-1)
        rows: list[dict[str, float]] = []
        for row in probs:
            ent = float(row[self._idx_entailment].item()) if self._idx_entailment is not None else 0.0
            ctr = (
                float(row[self._idx_contradiction].item())
                if self._idx_contradiction is not None
                else 0.0
            )
            neu = max(0.0, 1.0 - ent - ctr)
            rows.append(
                {
                    "entailment_prob": ent,
                    "contradiction_prob": ctr,
                    "neutral_prob": neu,
                }
            )
        return rows

    def score(self, premise: str, answer: str) -> dict[str, float]:
        if not premise.strip() or not answer.strip():
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
        scored: list[dict[str, float]] = []
        for i in range(0, len(claims), self.cfg.batch_size):
            scored.extend(self._score_batch(premise, claims[i : i + self.cfg.batch_size]))
        supported = sum(float(s["entailment_prob"] >= 0.5) for s in scored)
        contrad = sum(float(s["contradiction_prob"] >= 0.5) for s in scored)
        neutral = max(0.0, float(len(scored)) - supported - contrad)
        n = float(len(scored))
        return {
            "nli_num_claims": n,
            "nli_supported_claim_rate": supported / n,
            "nli_contradiction_claim_rate": contrad / n,
            "nli_neutral_claim_rate": neutral / n,
            "nli_hallucination_claim_rate": 1.0 - (supported / n),
            "nli_mean_entailment_prob": mean(s["entailment_prob"] for s in scored),
        }


_WS = re.compile(r"\s+")
_PUNC_TABLE = str.maketrans("", "", string.punctuation)


def _norm_text(s: str) -> str:
    return _WS.sub(" ", s.lower().translate(_PUNC_TABLE)).strip()


def _dedupe_refs(refs: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for r in refs:
        t = r.strip()
        if not t:
            continue
        k = _norm_text(t)
        if k in seen:
            continue
        seen.add(k)
        out.append(t)
    return out


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        txt = value.strip()
        if txt.startswith("[") and txt.endswith("]"):
            try:
                parsed = json.loads(txt)
                if isinstance(parsed, list):
                    return [str(x) for x in parsed if str(x).strip()]
            except json.JSONDecodeError:
                try:
                    parsed = ast.literal_eval(txt)
                    if isinstance(parsed, list):
                        return [str(x) for x in parsed if str(x).strip()]
                except Exception:
                    pass
        return [txt] if txt else []
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    return [str(value).strip()]


def load_eval_samples(dataset_key: str, split_override: str | None = None) -> list[EvalSample]:
    if load_dataset is None:
        raise RuntimeError("datasets package unavailable. Install dependencies first.")
    if dataset_key not in DATASET_SPECS:
        raise ValueError(f"Unsupported dataset key: {dataset_key}")
    spec = DATASET_SPECS[dataset_key]
    split = split_override or spec["split"]
    cfg = spec["hf_config"] or None
    ds = load_dataset(spec["hf_id"], cfg, split=split)

    out: list[EvalSample] = []
    for i, row in enumerate(ds):
        if dataset_key == "truthfulqa":
            refs = _as_list(row.get("correct_answers"))
            refs = _dedupe_refs(_as_list(row.get("best_answer")) + refs)
            q = str(row.get("question", "")).strip()
        elif dataset_key == "popqa":
            refs = _dedupe_refs(_as_list(row.get("possible_answers")))
            q = str(row.get("question", "")).strip()
        else:  # nq_open
            refs = _dedupe_refs(_as_list(row.get("answer")))
            q = str(row.get("question", "")).strip()
        if not q or not refs:
            continue
        out.append(
            EvalSample(
                sample_id=f"{dataset_key}-{i}",
                question=q,
                references=refs,
                metadata={"dataset": dataset_key},
            )
        )
    return out


def exact_match_any(pred: str, refs: list[str]) -> float:
    p = _norm_text(pred)
    return float(any(_norm_text(r) == p for r in refs))


def token_f1_best(pred: str, refs: list[str]) -> float:
    p = _norm_text(pred).split()
    if not p:
        return 0.0
    cp = Counter(p)
    best = 0.0
    for ref in refs:
        r = _norm_text(ref).split()
        if not r:
            continue
        cr = Counter(r)
        common = sum((cp & cr).values())
        if common == 0:
            continue
        precision = common / len(p)
        recall = common / len(r)
        score = 2 * precision * recall / (precision + recall)
        best = max(best, score)
    return best


def refusal_flag(pred: str) -> float:
    text = _norm_text(pred)
    patterns = [
        "i do not know",
        "i dont know",
        "cannot determine",
        "not enough information",
        "insufficient information",
    ]
    return float(any(p in text for p in patterns))


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


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
                print(f"Warning: skipping malformed JSON line {ln} in {path}", file=sys.stderr)
                continue
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="No-RAG prompt duplication benchmark on TruthfulQA/PopQA/NQ-Open."
    )
    p.add_argument(
        "--dataset",
        choices=["truthfulqa", "popqa", "nq_open", "all"],
        default="truthfulqa",
        help="Dataset preset to run.",
    )
    p.add_argument("--split", type=str, default=None, help="Override HF split for selected dataset(s).")
    p.add_argument("--config", type=Path, default=None, help="Optional generation config YAML.")
    p.add_argument("--model-id", type=str, default=None)
    p.add_argument("--max-new-tokens", type=int, default=None)
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument(
        "--duplication-strategy",
        choices=[PromptStrategy.DOUBLE_QUERY.value, PromptStrategy.REPEAT_USER_BLOCK.value],
        default=PromptStrategy.DOUBLE_QUERY.value,
        help="Prompt duplication variant to compare against single prompt.",
    )
    p.add_argument("--system-prompt", type=str, default=None)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs" / "prompt_duplication_no_rag",
        help="Root output directory.",
    )
    p.add_argument("--resume", action="store_true")
    p.add_argument("--disable-nli-checker", action="store_true")
    p.add_argument("--nli-model-id", type=str, default=DEFAULT_NLI_MODEL_ID)
    p.add_argument("--nli-max-claims", type=int, default=6)
    p.add_argument("--nli-max-length", type=int, default=256)
    p.add_argument("--nli-batch-size", type=int, default=16)
    p.add_argument(
        "--nli-reference-policy",
        choices=["first", "concat"],
        default="first",
        help="How to construct premise from references for NLI checking.",
    )
    return p.parse_args()


def evaluate_dataset(
    dataset_key: str,
    args: argparse.Namespace,
    engine: TinyLlamaEngine,
    nli_checker: NLIHallucinationChecker | None,
) -> dict[str, Any]:
    samples = load_eval_samples(dataset_key, args.split)
    if args.max_samples is not None:
        samples = samples[: args.max_samples]
    if not samples:
        raise RuntimeError(f"No valid rows found for {dataset_key}.")

    out_dir = args.output_dir / dataset_key
    ensure_dir(out_dir)
    pred_path = out_dir / "predictions.jsonl"
    summary_path = out_dir / "summary.json"

    existing_rows: list[dict[str, Any]] = []
    completed_ids: set[str] = set()
    if args.resume:
        existing_rows = load_existing_rows(pred_path)
        for row in existing_rows:
            sid = row.get("sample_id")
            if sid is not None:
                completed_ids.add(str(sid))
        if completed_ids:
            samples = [s for s in samples if s.sample_id not in completed_ids]
            print(
                f"[{dataset_key}] Resume: loaded {len(existing_rows)} rows; skipping {len(completed_ids)}.",
                file=sys.stderr,
            )
    elif pred_path.exists():
        pred_path.unlink()

    dup_strategy = PromptStrategy(args.duplication_strategy)
    rows: list[dict[str, Any]] = list(existing_rows)
    for sample in tqdm(samples, desc=f"Evaluating {dataset_key}", unit="sample"):
        baseline_msg = build_messages(
            sample.question,
            strategy=PromptStrategy.SINGLE,
            system=args.system_prompt,
        )
        dup_msg = build_messages(
            sample.question,
            strategy=dup_strategy,
            system=args.system_prompt,
        )
        pred_baseline = engine.generate_text(baseline_msg)
        pred_dup = engine.generate_text(dup_msg)

        baseline_metrics = {
            "exact_match": exact_match_any(pred_baseline, sample.references),
            "token_f1": token_f1_best(pred_baseline, sample.references),
            "refusal_rate": refusal_flag(pred_baseline),
        }
        dup_metrics = {
            "exact_match": exact_match_any(pred_dup, sample.references),
            "token_f1": token_f1_best(pred_dup, sample.references),
            "refusal_rate": refusal_flag(pred_dup),
        }
        if nli_checker is not None:
            premise = sample.references[0] if args.nli_reference_policy == "first" else "\n".join(sample.references)
            baseline_metrics.update(nli_checker.score(premise, pred_baseline))
            dup_metrics.update(nli_checker.score(premise, pred_dup))

        row = {
            "sample_id": sample.sample_id,
            "dataset": dataset_key,
            "question": sample.question,
            "references": sample.references,
            "prediction_single": pred_baseline,
            "prediction_duplicated": pred_dup,
            "duplication_strategy": dup_strategy.value,
            "metrics": {
                "single": baseline_metrics,
                "duplicated": dup_metrics,
                "delta_duplicated_minus_single": {
                    k: dup_metrics[k] - baseline_metrics[k] for k in baseline_metrics.keys()
                },
            },
        }
        rows.append(row)
        with open(pred_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def avg(metric: str, variant: str) -> float:
        vals = [float(r["metrics"][variant][metric]) for r in rows]
        return mean(vals) if vals else 0.0

    summary: dict[str, Any] = {
        "dataset": dataset_key,
        "num_samples": len(rows),
        "model_id": engine.config.model.model_id,
        "generation": asdict(engine.config.generation),
        "duplication_strategy": dup_strategy.value,
        "nli_checker": {
            "enabled": nli_checker is not None,
            "model_id": args.nli_model_id,
            "max_claims": args.nli_max_claims,
            "max_length": args.nli_max_length,
            "batch_size": args.nli_batch_size,
            "reference_policy": args.nli_reference_policy,
        },
        "aggregate": {
            "single": {
                "exact_match": avg("exact_match", "single"),
                "token_f1": avg("token_f1", "single"),
                "refusal_rate": avg("refusal_rate", "single"),
            },
            "duplicated": {
                "exact_match": avg("exact_match", "duplicated"),
                "token_f1": avg("token_f1", "duplicated"),
                "refusal_rate": avg("refusal_rate", "duplicated"),
            },
        },
    }
    if nli_checker is not None:
        for variant in ("single", "duplicated"):
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
    summary["aggregate"]["delta_duplicated_minus_single"] = {
        k: summary["aggregate"]["duplicated"][k] - summary["aggregate"]["single"][k]
        for k in summary["aggregate"]["single"].keys()
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"[{dataset_key}] wrote {pred_path}")
    print(f"[{dataset_key}] wrote {summary_path}")
    return summary


def run() -> int:
    args = parse_args()
    cfg = with_overrides(
        load_config(args.config),
        model_id=args.model_id,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )
    engine = TinyLlamaEngine(cfg)
    nli_checker = None
    if not args.disable_nli_checker:
        nli_checker = NLIHallucinationChecker(
            NLIConfig(
                enabled=True,
                model_id=args.nli_model_id,
                max_claims=args.nli_max_claims,
                max_length=args.nli_max_length,
                batch_size=args.nli_batch_size,
            )
        )

    keys = list(DATASET_SPECS.keys()) if args.dataset == "all" else [args.dataset]
    combined: dict[str, Any] = {}
    for key in keys:
        combined[key] = evaluate_dataset(key, args, engine, nli_checker)

    if args.dataset == "all":
        ensure_dir(args.output_dir)
        combined_path = args.output_dir / "summary_all.json"
        with open(combined_path, "w", encoding="utf-8") as f:
            json.dump(combined, f, ensure_ascii=False, indent=2)
        print(f"[all] wrote {combined_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())

