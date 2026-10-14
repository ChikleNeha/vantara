"""Config loader: YAML file plus environment overrides, with repo-relative paths."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "config" / "config.yaml"


class Config(dict):
    """Dict with attribute-style access and helpers for resolved paths."""

    def __getattr__(self, item: str) -> Any:
        try:
            value = self[item]
        except KeyError as exc:
            raise AttributeError(item) from exc
        return Config(value) if isinstance(value, dict) else value

    def path(self, key: str) -> Path:
        """Return an absolute path for a key under ``paths``."""
        p = Path(self["paths"][key])
        return p if p.is_absolute() else ROOT / p


def _apply_env(cfg: dict) -> dict:
    env_map = {
        "ARTIFACTS_DIR": ("paths", "artifacts_dir"),
        "DB_PATH": ("paths", "db_path"),
        "OLLAMA_BASE_URL": ("ollama", "base_url"),
        "OLLAMA_MODEL": ("ollama", "model"),
        "DATA_SOURCE": ("data", "source"),
    }
    for env, (section, key) in env_map.items():
        if os.environ.get(env):
            cfg[section][key] = os.environ[env]
    return cfg


def load_config(path: str | Path | None = None, overrides: dict | None = None) -> Config:
    """Load config from ``path`` (or $CBP_CONFIG, or the default) and apply overrides."""
    cfg_path = Path(path or os.environ.get("CBP_CONFIG") or DEFAULT_CONFIG)
    with open(cfg_path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    cfg = _apply_env(cfg)
    for dotted, value in (overrides or {}).items():
        node = cfg
        parts = dotted.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value
    return Config(cfg)


@lru_cache(maxsize=1)
def get_config() -> Config:
    """Cached config for the API process."""
    return load_config()
