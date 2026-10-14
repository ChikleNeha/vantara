"""Data collection (PRD 6.1): load both sheets, standardise names/types, sort chronologically."""
from __future__ import annotations

import pandas as pd

from src.data.synthetic import generate_transactions
from src.utils.config import Config
from src.utils.logging import get_logger

log = get_logger(__name__)

RENAME = {
    "Invoice": "invoice",
    "StockCode": "stock_code",
    "Description": "description",
    "Quantity": "quantity",
    "InvoiceDate": "invoice_date",
    "Price": "price",
    "Customer ID": "customer_id",
    "CustomerID": "customer_id",
    "Country": "country",
}


def standardize(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns, fix dtypes and sort chronologically."""
    df = df.rename(columns=RENAME).copy()
    df["invoice"] = df["invoice"].astype(str)
    df["stock_code"] = df["stock_code"].astype(str).str.strip().str.upper()
    df["invoice_date"] = pd.to_datetime(df["invoice_date"])
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["customer_id"] = pd.to_numeric(df["customer_id"], errors="coerce")
    df["country"] = df["country"].astype(str).str.strip()
    return df.sort_values("invoice_date", kind="stable").reset_index(drop=True)


def _from_file(cfg: Config) -> pd.DataFrame:
    path = cfg.path("raw_dir") / cfg.data["raw_file"]
    sheets = pd.read_excel(path, sheet_name=None)
    log.info("Loaded %d sheets from %s", len(sheets), path.name)
    return pd.concat(sheets.values(), ignore_index=True)


def _from_ucimlrepo() -> pd.DataFrame:
    from ucimlrepo import fetch_ucirepo  # imported lazily: optional dependency

    ds = fetch_ucirepo(id=502)
    return ds.data.features


def load_raw(cfg: Config) -> tuple[pd.DataFrame, str]:
    """Load raw transactions. Returns (frame, source_name)."""
    source = cfg.data["source"]
    raw_file = cfg.path("raw_dir") / cfg.data["raw_file"]
    attempts: list[str] = {
        "file": ["file"],
        "ucimlrepo": ["ucimlrepo"],
        "synthetic": ["synthetic"],
        "auto": ["file", "ucimlrepo", "synthetic"],
    }[source]
    for attempt in attempts:
        try:
            if attempt == "file":
                if not raw_file.exists():
                    raise FileNotFoundError(raw_file)
                return standardize(_from_file(cfg)), "file"
            if attempt == "ucimlrepo":
                return standardize(_from_ucimlrepo()), "ucimlrepo"
            syn = cfg.data["synthetic"]
            raw = generate_transactions(syn["n_customers"], syn["start"], syn["end"], cfg["seed"])
            log.warning("Using SYNTHETIC demo data. Place %s in data/raw/ to use the real dataset.", raw_file.name)
            return standardize(raw), "synthetic"
        except Exception as exc:  # noqa: BLE001 - fall through to the next source
            log.warning("Source '%s' unavailable: %s", attempt, exc)
    raise RuntimeError(f"No data source available (tried {attempts})")
