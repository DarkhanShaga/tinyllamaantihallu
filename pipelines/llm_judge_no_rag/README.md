# LLM Judge No-RAG Pipeline

This pipeline compares `single` vs duplicated prompts and uses **only LLM-as-judge metrics**.

Supported dataset presets:

- `truthfulqa`
- `popqa`
- `nq_open`
- `all` (runs all three sequentially)

## What it does

1. Loads QA data from Hugging Face preset.
2. Generates:
   - `single` (`PromptStrategy.SINGLE`)
   - `duplicated` (`double_query` or `repeat_user_block`)
3. Performs blinded A/B pairwise judging with LLM.
4. Writes:
   - `<output-dir>/<dataset>/predictions.jsonl`
   - `<output-dir>/<dataset>/summary.json`
   - and `summary_all.json` when `--dataset all`

## Metrics

- `single_win_rate`
- `duplicated_win_rate`
- `tie_rate`
- `hallucination_rate_single`
- `hallucination_rate_duplicated`
- `hallucination_delta_duplicated_minus_single`
- `win_rate_delta_duplicated_minus_single`
- `judge_parse_ok_rate`

## Run

```bash
cd /home/darkhanshaga/CursorProjects/GenAIProject
source .venv/bin/activate
python pipelines/llm_judge_no_rag/run.py \
  --dataset all \
  --output-dir outputs/llm_judge_no_rag \
  --max-samples 200 \
  --duplication-strategy double_query \
  --max-new-tokens 96 \
  --temperature 0.2 \
  --resume
```

## Useful options

- `--judge-model-id <model>`: separate model for judge stage
- `--judge-backend local|gemini`: select judge backend
- `--gemini-model gemini-2.0-flash`
- `--gemini-api-key <key>` or set `GEMINI_API_KEY`
- `--reference-policy first|concat`: how references are passed to the judge

The pipeline automatically loads `GEMINI_API_KEY` from project `.env` if present.
