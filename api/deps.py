"""Application state, database helpers and shared dependencies."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import HTTPException, Request
from sqlalchemy import text
from sqlalchemy.engine import Engine

from src.serving.ollama_client import OllamaClient
from src.serving.scoring import ModelBundle
from src.utils.config import Config
from src.utils.logging import get_logger

log = get_logger("api")


class AppState:
    """Holds the model bundle, DB engine and Ollama client; created once at startup."""

    def __init__(self, cfg: Config, engine: Engine, bundle: ModelBundle | None, metrics: dict[str, Any], ollama: OllamaClient):
        self.cfg, self.engine, self.bundle, self.metrics, self.ollama = cfg, engine, bundle, metrics, ollama
        self.started = time.time()


def state(request: Request) -> AppState:
    """Return app state or a clear 503 if the pipeline has not been run yet."""
    st: AppState | None = getattr(request.app.state, "cbp", None)
    if st is None or st.bundle is None:
        raise HTTPException(503, "Model artifacts not found. Run `python -m src.pipeline` first (see README).")
    return st


def query(engine: Engine, sql: str, params: dict[str, Any] | None = None) -> pd.DataFrame:
    """Parameterised read-only query returning a DataFrame."""
    with engine.connect() as conn:
        return pd.read_sql(text(sql), conn, params=params or {})


def load_metrics(art_dir: Path) -> dict[str, Any]:
    """metrics.json written by the pipeline (empty dict if missing)."""
    p = art_dir / "metrics.json"
    return json.loads(p.read_text()) if p.exists() else {}


def clean_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    """DataFrame to JSON-safe records (NaN -> None, numpy bool/int -> python)."""
    return json.loads(df.to_json(orient="records", date_format="iso"))
