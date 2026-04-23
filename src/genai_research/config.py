from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from genai_research.paths import default_config_path


@dataclass
class ModelSettings:
    model_id: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    trust_remote_code: bool = False
    device_map: str = "auto"
    torch_dtype: str = "auto"


@dataclass
class GenerationSettings:
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    do_sample: bool = True
    repetition_penalty: float = 1.0


@dataclass
class RuntimeConfig:
    model: ModelSettings = field(default_factory=ModelSettings)
    generation: GenerationSettings = field(default_factory=GenerationSettings)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> RuntimeConfig:
        m = raw.get("model", {}) or {}
        g = raw.get("generation", {}) or {}
        return cls(
            model=ModelSettings(
                model_id=m.get("model_id", ModelSettings.model_id),
                trust_remote_code=bool(m.get("trust_remote_code", False)),
                device_map=str(m.get("device_map", "auto")),
                torch_dtype=str(m.get("torch_dtype", "auto")),
            ),
            generation=GenerationSettings(
                max_new_tokens=int(g.get("max_new_tokens", 256)),
                temperature=float(g.get("temperature", 0.7)),
                top_p=float(g.get("top_p", 0.9)),
                do_sample=bool(g.get("do_sample", True)),
                repetition_penalty=float(g.get("repetition_penalty", 1.0)),
            ),
        )


def _read_yaml(path: Path) -> RuntimeConfig:
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return RuntimeConfig.from_dict(data)


def load_config(path: Path | None = None) -> RuntimeConfig:
    if path and path.is_file():
        return _read_yaml(path)
    env = os.environ.get("GENAI_CONFIG_PATH")
    if env:
        ep = Path(env)
        if ep.is_file():
            return _read_yaml(ep)
    default = default_config_path()
    if default.is_file():
        return _read_yaml(default)
    return RuntimeConfig()


def with_overrides(
    base: RuntimeConfig,
    *,
    model_id: str | None = None,
    max_new_tokens: int | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    do_sample: bool | None = None,
) -> RuntimeConfig:
    """Return a copy of `base` with selected fields replaced."""
    from dataclasses import replace

    cfg = base
    if model_id is not None:
        cfg = replace(cfg, model=replace(cfg.model, model_id=model_id))
    g = cfg.generation
    if any(
        x is not None for x in (max_new_tokens, temperature, top_p, do_sample)
    ):
        g = replace(
            g,
            max_new_tokens=g.max_new_tokens
            if max_new_tokens is None
            else int(max_new_tokens),
            temperature=g.temperature if temperature is None else float(temperature),
            top_p=g.top_p if top_p is None else float(top_p),
            do_sample=g.do_sample if do_sample is None else bool(do_sample),
        )
        cfg = replace(cfg, generation=g)
    return cfg


def config_from_env(overrides: dict[str, Any] | None = None) -> RuntimeConfig:
    base = load_config()
    o = dict(overrides or {})
    if mid := os.environ.get("GENAI_MODEL_ID"):
        o.setdefault("model_id", mid)
    return RuntimeConfig.from_dict(
        {
            "model": {
                "model_id": o.get("model_id", base.model.model_id),
                "trust_remote_code": o.get("trust_remote_code", base.model.trust_remote_code),
                "device_map": o.get("device_map", base.model.device_map),
                "torch_dtype": o.get("torch_dtype", base.model.torch_dtype),
            },
            "generation": {
                "max_new_tokens": o.get("max_new_tokens", base.generation.max_new_tokens),
                "temperature": o.get("temperature", base.generation.temperature),
                "top_p": o.get("top_p", base.generation.top_p),
                "do_sample": o.get("do_sample", base.generation.do_sample),
                "repetition_penalty": o.get(
                    "repetition_penalty", base.generation.repetition_penalty
                ),
            },
        }
    )
