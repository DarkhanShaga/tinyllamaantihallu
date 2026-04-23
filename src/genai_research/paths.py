from __future__ import annotations

from pathlib import Path


def project_root() -> Path:
    """Return repository root (directory containing `config/` and `src/`)."""
    return Path(__file__).resolve().parents[2]


def default_config_path() -> Path:
    return project_root() / "config" / "default.yaml"
