"""Shared fixtures: a tiny end-to-end pipeline run in a temp directory, reused by pipeline and API tests."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import pytest
import yaml

from src.utils.config import ROOT


@pytest.fixture(scope="session")
def tiny_config_path(tmp_path_factory) -> Path:
    """A config that points everything at a temp dir and shrinks the models so the whole pipeline runs quickly."""
    base = yaml.safe_load((ROOT / "config" / "config.yaml").read_text())
    tmp = tmp_path_factory.mktemp("cbp")
    base["paths"] = {k: str(tmp / k) for k in ("raw_dir", "interim_dir", "processed_dir", "artifacts_dir", "docs_dir")} | {
        "db_path": str(tmp / "cbp.sqlite")
    }
    base["data"]["source"] = "synthetic"
    base["data"]["synthetic"]["n_customers"] = 700
    base["models"].update({"cv_folds": 3, "search_iter": 2})
    base["deep_learning"].update({"epochs": 4, "patience": 2})
    base["segmentation"].update({"k_min": 3, "k_max": 4})
    path = tmp / "config.yaml"
    path.write_text(yaml.safe_dump(base))
    return path


@pytest.fixture(scope="session")
def pipeline_run(tiny_config_path):
    """Run the full pipeline once; returns (cfg, metrics)."""
    from src.pipeline import run_pipeline
    from src.utils.config import load_config

    cfg = load_config(tiny_config_path)
    metrics = run_pipeline(cfg)
    return cfg, metrics


@pytest.fixture(scope="session")
def client(pipeline_run, tiny_config_path):
    """FastAPI TestClient backed by the tiny pipeline's artifacts and database."""
    os.environ["CBP_CONFIG"] = str(tiny_config_path)
    from fastapi.testclient import TestClient

    from src.utils.config import get_config

    get_config.cache_clear()
    from api.main import app

    with TestClient(app) as c:
        yield c
    get_config.cache_clear()


def tx(rows: list[tuple]) -> pd.DataFrame:
    """Build a small cleaned-transactions frame: (customer, invoice, date, qty, price, category, is_return)."""
    df = pd.DataFrame(rows, columns=["customer_id", "invoice", "invoice_date", "quantity", "price", "category", "is_return"])
    df["invoice_date"] = pd.to_datetime(df["invoice_date"])
    df["line_total"] = df["quantity"] * df["price"]
    df["stock_code"] = "A1"
    df["country"] = "United Kingdom"
    df["description"] = "ITEM"
    return df
