# LLM Judge With RAG Pipeline

This pipeline compares `standard_rag` vs `sandwich` and uses **only LLM-as-judge metrics**.

## What it does

1. Loads a RAG-style dataset from local JSON/JSONL or Hugging Face.
2. Generates two answers per sample:
   - `standard_rag` (`PromptStrategy.SINGLE`)
   - `sandwich` (`PromptStrategy.SANDWICH`)
3. Runs a blinded pairwise judge prompt (A/B randomization by sample id hash).
4. Writes:
   - `predictions.jsonl` with per-sample judge decision
   - `summary.json` with aggregate judge metrics

## Metrics

- `standard_rag_win_rate`
- `sandwich_win_rate`
- `tie_rate`
- `hallucination_rate_standard_rag`
- `hallucination_rate_sandwich`
- `hallucination_delta_sandwich_minus_standard`
- `win_rate_delta_sandwich_minus_standard`
- `judge_parse_ok_rate`

## Run

```bash
cd /home/darkhanshaga/CursorProjects/GenAIProject
source .venv/bin/activate
python pipelines/llm_judge_with_rag/run.py \
  --hf-dataset minko186/libreeval-non-synthetic-hallucinations-en \
  --hf-split train \
  --output-dir outputs/llm_judge_with_rag \
  --max-samples 200 \
  --max-new-tokens 96 \
  --temperature 0.2 \
  --resume
```

## Useful options

- `--judge-model-id <model>`: use a separate model for judging
- `--judge-backend local|gemini`: select judge backend
- `--gemini-model gemini-2.0-flash`
- `--gemini-api-key <key>` or set `GEMINI_API_KEY`
- `--evidence-source auto|context|reference|concat` (default `auto`)
- `--dataset-path /path/to/data.jsonl` for local datasets

The pipeline automatically loads `GEMINI_API_KEY` from project `.env` if present.
