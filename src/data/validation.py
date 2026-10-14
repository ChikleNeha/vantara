"""Automated data validation: fails loudly before feature engineering (PRD 6.2)."""
from __future__ import annotations

import pandas as pd

from src.utils.config import Config
from src.utils.logging import get_logger

log = get_logger(__name__)

REQUIRED_COLUMNS = ["invoice", "stock_code", "description", "quantity", "invoice_date", "price", "customer_id", "country"]


class DataValidationError(ValueError):
    """Raised when raw data violates schema, null-rate or date-range expectations."""


def validate_raw(df: pd.DataFrame, cfg: Config) -> dict:
    """Run schema, null-rate, volume and date-range checks. Raises ``DataValidationError`` on failure."""
    v = cfg.data["validation"]
    problems: list[str] = []

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise DataValidationError(f"Missing required columns: {missing}")
    if len(df) < v["min_rows"]:
        problems.append(f"Only {len(df)} rows (< {v['min_rows']})")
    if df["invoice_date"].isna().any():
        problems.append("Null InvoiceDate values present")
    if df["price"].isna().mean() > v["max_null_rate_price"]:
        problems.append(f"Price null rate {df['price'].isna().mean():.3%} too high")
    if df["description"].isna().mean() > v["max_null_rate_description"]:
        problems.append(f"Description null rate {df['description'].isna().mean():.3%} too high")
    if not pd.api.types.is_datetime64_any_dtype(df["invoice_date"]):
        problems.append("InvoiceDate is not a datetime")
    else:
        span_days = (df["invoice_date"].max() - df["invoice_date"].min()).days
        if span_days < 180:
            problems.append(f"Date range only {span_days} days; need at least 180 for a churn label")
    if problems:
        for p in problems:
            log.error("Validation failure: %s", p)
        raise DataValidationError("; ".join(problems))
    report = {
        "rows": int(len(df)),
        "date_min": str(df["invoice_date"].min()),
        "date_max": str(df["invoice_date"].max()),
        "customer_id_null_rate": round(float(df["customer_id"].isna().mean()), 4),
    }
    log.info("Raw data validation passed: %s", report)
    return report
