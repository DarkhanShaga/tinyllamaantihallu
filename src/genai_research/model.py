from __future__ import annotations

from dataclasses import replace
from typing import Any, Iterable, Mapping

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from genai_research.config import GenerationSettings, ModelSettings, RuntimeConfig, load_config


def _select_dtype(name: str) -> torch.dtype:
    if name == "auto":
        if not torch.cuda.is_available():
            return torch.float32
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    key = name.lower()
    if key in ("fp32", "float32"):
        return torch.float32
    if key in ("fp16", "float16"):
        return torch.float16
    if key in ("bf16", "bfloat16"):
        return torch.bfloat16
    raise ValueError(f"Unknown torch_dtype: {name!r}")


def _load_kwargs(model_cfg: ModelSettings) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "trust_remote_code": model_cfg.trust_remote_code,
        "torch_dtype": _select_dtype(model_cfg.torch_dtype),
    }
    if torch.cuda.is_available() and model_cfg.device_map == "auto":
        kwargs["device_map"] = "auto"
    return kwargs


class TinyLlamaEngine:
    """
    Load TinyLlama (or any compatible causal LM) once and run chat-templated generation.

    Intended for controlled experiments: pair with `build_messages` / `PromptStrategy`.
    """

    def __init__(
        self,
        config: RuntimeConfig | None = None,
        *,
        model_settings: ModelSettings | None = None,
        generation: GenerationSettings | None = None,
    ) -> None:
        self.config = config or load_config()
        if model_settings is not None:
            self.config = replace(self.config, model=model_settings)
        if generation is not None:
            self.config = replace(self.config, generation=generation)

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.config.model.model_id,
            trust_remote_code=self.config.model.trust_remote_code,
        )
        if self._tokenizer.pad_token is None and self._tokenizer.eos_token is not None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        load_kw = _load_kwargs(self.config.model)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.config.model.model_id, **load_kw
        )
        if not torch.cuda.is_available() or self.config.model.device_map != "auto":
            self._model.to(self._device())
        self._model.eval()

    def _device(self) -> torch.device:
        p = next(self._model.parameters(), None)
        if p is None:
            return torch.device("cpu")
        return p.device

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
        return self._model

    def format_chat(self, messages: list[dict[str, str]]) -> dict[str, torch.Tensor]:
        """Apply the model's chat template and return tensors on the model device."""
        raw = self._tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            tokenize=True,
        )
        if isinstance(raw, torch.Tensor):
            input_ids = raw
            attention_mask = None
        else:
            # Some tokenizer versions return BatchEncoding / dict
            input_ids = raw["input_ids"]  # type: ignore[index]
            attention_mask = raw.get("attention_mask")  # type: ignore[assignment]
        input_ids = input_ids.to(self._device())
        if attention_mask is not None:
            attention_mask = attention_mask.to(self._device())
        out: dict[str, torch.Tensor] = {"input_ids": input_ids}
        if attention_mask is not None:
            out["attention_mask"] = attention_mask
        return out

    def render_chat_prompt(self, messages: list[dict[str, str]]) -> str:
        """
        Exact prompt string passed to the model (chat template + generation prompt).
        Matches what `format_chat` tokenizes before `generate`.
        """
        raw = self._tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )
        if isinstance(raw, str):
            return raw
        # Some tokenizer versions may return an unexpected type; coerce for debug output.
        return str(raw)

    def generate_text(
        self,
        messages: list[dict[str, str]],
        *,
        gen_overrides: GenerationSettings | None = None,
        **generate_kwargs: Any,
    ) -> str:
        """Generate assistant text (decoded new tokens only)."""
        g = gen_overrides or self.config.generation
        enc = self.format_chat(messages)
        input_len = int(enc["input_ids"].shape[1])
        extra: dict[str, Any] = {
            "max_new_tokens": g.max_new_tokens,
            "do_sample": g.do_sample,
            "temperature": g.temperature,
            "top_p": g.top_p,
        }
        if g.repetition_penalty and g.repetition_penalty != 1.0:
            extra["repetition_penalty"] = g.repetition_penalty
        extra.update(generate_kwargs)
        with torch.inference_mode():
            outputs = self._model.generate(**enc, **extra)
        new_tokens = outputs[0, input_len:]
        return self._tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    def generate_from_records(
        self,
        items: Iterable[Mapping[str, Any]],
        message_key: str = "messages",
    ) -> list[str]:
        """
        Run batch-like generation for an iterable of dicts, each with `message_key` -> messages list.

        Kept for future eval loops (JSONL loaders).
        """
        return [self.generate_text(r[message_key]) for r in items]
