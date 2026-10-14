"""Data cleaning and wrangling (PRD 6.2, 6.3). Pure, unit-testable functions."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.categories import categorize
from src.utils.config import Config
from src.utils.logging import get_logger

log = get_logger(__name__)


def flag_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Mark cancellations/returns (invoice starts with 'C' or negative quantity). Kept, not dropped."""
    out = df.copy()
    out["is_return"] = out["invoice"].str.upper().str.startswith("C") | (out["quantity"] < 0)
    return out


def drop_exact_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove rows identical on every field; repeated SKUs on one order with different quantities survive."""
    cols = ["invoice", "stock_code", "quantity", "invoice_date", "price", "customer_id"]
    return df.drop_duplicates(subset=cols, keep="first")


def standardize_descriptions(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise case/whitespace and fill missing descriptions from a StockCode lookup (modal value)."""
    out = df.copy()
    out["description"] = out["description"].astype("string").str.strip().str.replace(r"\s+", " ", regex=True).str.upper()
    lookup = (
        out.dropna(subset=["description"]).groupby("stock_code")["description"].agg(lambda s: s.mode().iat[0])
    )
    out["description"] = out["description"].fillna(out["stock_code"].map(lookup)).fillna("UNKNOWN ITEM")
    out["description"] = out["stock_code"].map(lookup).fillna(out["description"])
    return out.astype({"description": "object"})


def iqr_bounds(s: pd.Series, k: float = 3.0) -> tuple[float, float]:
    """Tukey fences with multiplier ``k`` (3.0 = 'far out' to leave legitimate bulk orders alone)."""
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    return float(q1 - k * iqr), float(q3 + k * iqr)


def handle_prices_and_outliers(df: pd.DataFrame, cfg: Config) -> tuple[pd.DataFrame, dict]:
    """Drop invalid prices and data-entry errors; keep (but flag) legitimate bulk purchases.

    Rule: price <= 0 is an adjustment and removed. |quantity| above ``max_valid_quantity`` is a
    data-entry error and removed. Anything beyond the IQR fence but below that cap is a bulk order,
    flagged ``is_bulk`` and kept.
    """
    d = cfg.data
    before = len(df)
    out = df[(df["price"] > 0) & (df["price"] <= d["max_valid_price"])].copy()
    n_bad_price = before - len(out)
    n1 = len(out)
    out = out[out["quantity"].abs() <= d["max_valid_quantity"]].copy()
    n_entry = n1 - len(out)
    _, hi = iqr_bounds(out.loc[~out["is_return"], "quantity"])
    out["is_bulk"] = (out["quantity"] > hi) & ~out["is_return"]
    stats = {"invalid_price_rows": int(n_bad_price), "data_entry_errors": int(n_entry), "bulk_rows_flagged": int(out["is_bulk"].sum())}
    return out, stats


def split_admin_codes(df: pd.DataFrame, cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (product_lines, admin_lines). Admin = postage, bank charges, manual entries."""
    admin_codes = {c.upper() for c in cfg.data["admin_stock_codes"]}
    is_admin = df["stock_code"].isin(admin_codes) | df["stock_code"].str.match(r"^[A-Z]+$", na=False) & ~df["stock_code"].str.match(r"^\d", na=False)
    return df[~is_admin].copy(), df[is_admin].copy()


def clean_transactions(raw: pd.DataFrame, cfg: Config) -> tuple[pd.DataFrame, dict]:
    """Full cleaning chain. Returns product-level lines (all customers) and a cleaning report."""
    report: dict = {"raw_rows": int(len(raw))}
    df = flag_returns(raw)
    df = drop_exact_duplicates(df)
    report["after_dedup"] = int(len(df))
    df = standardize_descriptions(df)
    df, stats = handle_prices_and_outliers(df, cfg)
    report.update(stats)
    df, admin = split_admin_codes(df, cfg)
    report["admin_rows_removed"] = int(len(admin))
    report["missing_customer_id_rows"] = int(df["customer_id"].isna().sum())
    df["line_total"] = df["quantity"] * df["price"]
    df["category"] = categorize(df["description"])
    df["customer_id"] = df["customer_id"].astype("Int64")
    report["clean_rows"] = int(len(df))
    log.info("Cleaning report: %s", report)
    return df.reset_index(drop=True), report


def customer_lines(clean: pd.DataFrame) -> pd.DataFrame:
    """Rows usable for customer-level modelling (valid Customer ID)."""
    return clean[clean["customer_id"].notna()].copy()


def encode_country(series: pd.Series, levels: list[str]) -> pd.DataFrame:
    """One-hot encode country using a fixed level list (fitted at training time, reused at inference)."""
    cleaned = series.where(series.isin(levels), "Other")
    cols = {f"country_{lvl}": (cleaned == lvl).astype(int) for lvl in [*levels, "Other"]}
    return pd.DataFrame(cols, index=series.index)


def top_countries(series: pd.Series, n: int) -> list[str]:
    """Most frequent countries, used as the fixed one-hot vocabulary."""
    return series.value_counts().head(n).index.tolist()


def safe_div(a, b):
    """Elementwise division returning 0 where the denominator is 0."""
    return np.where(np.asarray(b) == 0, 0.0, np.asarray(a) / np.where(np.asarray(b) == 0, 1, np.asarray(b)))
