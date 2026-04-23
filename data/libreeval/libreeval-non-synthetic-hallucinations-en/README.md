---
dataset_info:
  features:
  - name: reference
    dtype: string
  - name: input
    dtype: string
  - name: output
    dtype: string
  - name: label
    dtype: string
  - name: label_gpt-4o
    dtype: string
  - name: explanation_gpt-4o
    dtype: string
  - name: label_claude-3-5-sonnet-latest
    dtype: string
  - name: explanation_claude-3-5-sonnet-latest
    dtype: string
  - name: label_litellm/together_ai/Qwen/Qwen2.5-7B-Instruct-Turbo
    dtype: string
  - name: explanation_litellm/together_ai/Qwen/Qwen2.5-7B-Instruct-Turbo
    dtype: string
  - name: rag_model
    dtype: string
  - name: force_even_split
    dtype: bool
  - name: website
    dtype: string
  - name: synthetic
    dtype: bool
  - name: language
    dtype: string
  - name: hallucination_type_realized
    dtype: string
  - name: question_type
    dtype: string
  - name: hallucination_type_encouraged
    dtype: string
  - name: hallucination_type_realized_ensemble
    dtype: string
  - name: human_label
    dtype: string
  splits:
  - name: train
    num_bytes: 34558712
    num_examples: 10871
  download_size: 11323164
  dataset_size: 34558712
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
---
