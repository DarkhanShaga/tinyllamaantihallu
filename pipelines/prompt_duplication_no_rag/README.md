# Prompt Duplication (No RAG) Pipeline

This pipeline compares **single prompt** vs **prompt duplication** for factual QA tasks without retrieval context.

Supported dataset presets:

- `truthfulqa` -> `truthfulqa/truthful_qa` (`generation`, `validation`)
- `popqa` -> `akariasai/PopQA` (`test`)
- `nq_open` -> `nq_open` (`nq_open`, `validation`)

## What it evaluates

For each sample, it generates:

- baseline: `PromptStrategy.SINGLE`
- duplicated: `PromptStrategy.DOUBLE_QUERY` (default) or `REPEAT_USER_BLOCK`

Metrics:

- `exact_match` (against any reference)
- `token_f1` (best across references)
- `refusal_rate`
- optional NLI metrics via `valhalla/distilbart-mnli-12-3` using references as premise

## Install

```bash
cd /home/darkhanshaga/CursorProjects/GenAIProject
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Run one dataset

```bash
python pipelines/prompt_duplication_no_rag/run.py \
  --dataset truthfulqa \
  --output-dir outputs/prompt_dup_no_rag \
  --max-samples 200
```

## Run all three datasets

```bash
python pipelines/prompt_duplication_no_rag/run.py \
  --dataset all \
  --output-dir outputs/prompt_dup_no_rag_all \
  --max-samples 500
```

## Useful options

- `--duplication-strategy double_query|repeat_user_block`
- `--model-id TinyLlama/TinyLlama-1.1B-Chat-v1.0`
- `--max-new-tokens 96 --temperature 0.2`
- `--resume` (skip completed `sample_id`s)
- `--disable-nli-checker`
- `--nli-reference-policy first|concat`

## Outputs

Per dataset, in `--output-dir/<dataset>/`:

- `predictions.jsonl` (per-sample outputs + metrics)
- `summary.json` (aggregate comparison + deltas)

If `--dataset all`, also:

- `--output-dir/summary_all.json`
