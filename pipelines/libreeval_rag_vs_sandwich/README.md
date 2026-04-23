## LibreEval RAG vs Sandwich Pipeline

This folder contains a standalone A/B evaluation pipeline to compare:

- `standard_rag` -> `PromptStrategy.SINGLE`
- `sandwich` -> `PromptStrategy.SANDWICH`

against LibreEval1.0-style records using TinyLlama.

### What it does

1. Loads records from:
   - local JSON/JSONL via `--dataset-path`, or
   - Hugging Face datasets via `--hf-dataset` and `--hf-split`.
2. Maps each record to:
   - question
   - context
   - reference answer
3. Runs both strategies on the same sample.
4. Runs a second-stage NLI hallucination check (default enabled):
   - model: `valhalla/distilbart-mnli-12-3`
   - premise: selectable via `--nli-premise-source` (default `auto`: context, else reference)
   - hypothesis: each sentence-level claim from the generated answer
   - output metrics: supported/contradiction/neutral/hallucination claim rates
5. Writes:
   - per-sample outputs JSONL
   - aggregate summary JSON with deltas

### Expected fields

The mapper supports common schema variants and picks from several aliases:

- question: `question`, `query`, `prompt`, `user_query`, `input`
- context: `context`, `contexts`, `documents`, `retrieved_contexts`, `evidence`
- reference: `answer`, `answers`, `ground_truth`, `reference`, `references`, `gold_answer`

If your LibreEval export uses different keys, update `FIELD_ALIASES` in `run.py`.

### Install

```bash
cd /home/darkhanshaga/CursorProjects/GenAIProject
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### Run (local file)

```bash
python pipelines/libreeval_rag_vs_sandwich/run.py \
  --dataset-path /path/to/libreeval.jsonl \
  --output-dir outputs/libreeval_compare \
  --max-samples 200
```

### Run (HF dataset)

```bash
python pipelines/libreeval_rag_vs_sandwich/run.py \
  --hf-dataset your-org/LibreEval1.0 \
  --hf-split test \
  --output-dir outputs/libreeval_compare
```

### NLI checker options

- `--disable-nli-checker`: turn off second-stage checker
- `--nli-model-id`: override model id (default `valhalla/distilbart-mnli-12-3`)
- `--nli-max-claims`: max claims per answer (default `8`)
- `--nli-max-length`: tokenizer max length (default `384`)
- `--nli-batch-size`: claim batch size (default `8`)
- `--nli-premise-source`: one of `auto|context|reference|output` (default `auto`)

### Resume mode

- `--resume`: continue from existing `predictions.jsonl` in `--output-dir`
- the pipeline loads existing rows, skips completed `sample_id`s, and appends only missing ones
- invalid/truncated JSON lines (e.g. after interruption) are skipped with a warning

### Outputs

- `outputs/libreeval_compare/predictions.jsonl`
- `outputs/libreeval_compare/summary.json`

