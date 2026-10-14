"""Customer-level feature engineering (PRD section 7).

Point-in-time rule: every feature is computed from rows with ``invoice_date < cutoff``.
Labels are computed only from rows in ``[cutoff, cutoff + horizon)``. The leakage test in
``tests/test_features.py`` checks that features are unchanged when all post-cutoff rows are removed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.categories import CATEGORIES
from src.data.cleaning import encode_country
from src.utils.config import Config
from src.utils.logging import get_logger

log = get_logger(__name__)

AFF_COLS = [f"aff_{c}" for c in CATEGORIES]
BASE_FEATURES = [
    "recency_days",
    "frequency",
    "monetary_total",
    "monetary_avg",
    "clv_historical",
    "avg_basket_size",
    "purchase_trend",
    "gap_mean",
    "gap_std",
    "seasonal_concentration",
    "return_rate",
    "discount_sensitivity",
    "tenure_days",
    "engagement_score",
]
TARGET_COLS = ["churn", "clv_target", "next_category"]
EXPLAIN_NAMES = {
    "recency_days": "days since last purchase",
    "frequency": "number of orders",
    "monetary_total": "total spend",
    "monetary_avg": "average order value",
    "clv_historical": "net lifetime spend",
    "avg_basket_size": "average basket size",
    "purchase_trend": "order-frequency trend",
    "gap_mean": "average days between orders",
    "gap_std": "irregularity of purchase timing",
    "seasonal_concentration": "share of orders in Oct to Dec",
    "return_rate": "return rate",
    "discount_sensitivity": "reliance on discounted items",
    "tenure_days": "customer age in days",
    "engagement_score": "engagement score",
}


def feature_columns(country_levels: list[str]) -> list[str]:
    """Ordered list of model input columns for a given country vocabulary."""
    return BASE_FEATURES + AFF_COLS + [f"country_{lvl}" for lvl in [*country_levels, "Other"]]


def _monthly_slope(dates: pd.DataFrame, cutoff: pd.Timestamp, window: int) -> pd.Series:
    """Least-squares slope of monthly order counts over the last ``window`` months (per customer)."""
    offs = ((cutoff - dates["date"]).dt.days // 30.4375).astype(int)
    recent = dates[offs < window].assign(bucket=window - 1 - offs[offs < window])  # oldest bucket = 0
    counts = recent.groupby(["customer_id", "bucket"]).size().unstack(fill_value=0).reindex(columns=range(window), fill_value=0)
    x = np.arange(window, dtype=float)
    xc = x - x.mean()
    slope = (counts.to_numpy() - counts.to_numpy().mean(axis=1, keepdims=True)) @ xc / (xc**2).sum()
    return pd.Series(slope, index=counts.index)


def fit_reference(feats: pd.DataFrame) -> dict[str, np.ndarray]:
    """Sorted training distributions used to turn recency/frequency/monetary into percentile ranks."""
    return {
        "recency_days": np.sort(feats["recency_days"].to_numpy()),
        "frequency": np.sort(feats["frequency"].to_numpy()),
        "monetary_total": np.sort(feats["monetary_total"].to_numpy()),
    }


def engagement_score(feats: pd.DataFrame, reference: dict[str, np.ndarray]) -> pd.Series:
    """Composite 0-100 score: mean percentile of (inverted) recency, frequency and monetary value."""

    def pct(col: str) -> np.ndarray:
        ref = reference[col]
        return np.searchsorted(ref, feats[col].to_numpy(), side="right") / len(ref)

    score = ((1 - pct("recency_days")) + pct("frequency") + pct("monetary_total")) / 3 * 100
    return pd.Series(np.round(score, 2), index=feats.index)


def build_features(
    lines: pd.DataFrame,
    cutoff: pd.Timestamp,
    cfg: Config,
    country_levels: list[str],
    reference: dict[str, np.ndarray] | None = None,
    with_labels: bool = True,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Build the customer feature table as of ``cutoff``.

    Returns (features, reference). If ``reference`` is None it is fitted on this population.
    """
    cutoff = pd.Timestamp(cutoff)
    f_cfg = cfg.features
    hist = lines[(lines["invoice_date"] < cutoff) & lines["customer_id"].notna()].copy()
    hist["customer_id"] = hist["customer_id"].astype("int64")
    purchases = hist[~hist["is_return"]]
    returns = hist[hist["is_return"]]
    if purchases.empty:
        raise ValueError("No purchases before the cutoff date")

    orders = (
        purchases.groupby(["customer_id", "invoice"])
        .agg(date=("invoice_date", "min"), total=("line_total", "sum"), qty=("quantity", "sum"))
        .reset_index()
        .sort_values(["customer_id", "date"])
    )
    g = orders.groupby("customer_id")
    feats = pd.DataFrame(
        {
            "recency_days": (cutoff - g["date"].max()).dt.total_seconds() / 86400,
            "frequency": g.size(),
            "monetary_total": g["total"].sum(),
            "monetary_avg": g["total"].mean(),
            "avg_basket_size": g["qty"].mean(),
            "tenure_days": (cutoff - g["date"].min()).dt.total_seconds() / 86400,
        }
    )
    feats["recency_days"] = feats["recency_days"].round(2)
    feats["tenure_days"] = feats["tenure_days"].round(2)

    gaps = orders.groupby("customer_id")["date"].diff().dt.total_seconds().div(86400)
    gap_g = gaps.groupby(orders["customer_id"])
    feats["gap_mean"] = gap_g.mean()
    feats["gap_std"] = gap_g.std()
    feats["purchase_trend"] = _monthly_slope(orders, cutoff, f_cfg["trend_window_months"])
    in_q4 = orders["date"].dt.month >= 10
    feats["seasonal_concentration"] = in_q4.groupby(orders["customer_id"]).mean()

    ret_qty = returns.groupby("customer_id")["quantity"].apply(lambda s: s.abs().sum())
    buy_qty = purchases.groupby("customer_id")["quantity"].sum()
    feats["return_rate"] = (ret_qty.reindex(feats.index).fillna(0) / buy_qty.reindex(feats.index)).clip(0, 1)
    ret_val = returns.groupby("customer_id")["line_total"].sum().abs().reindex(feats.index).fillna(0)
    feats["clv_historical"] = (feats["monetary_total"] - ret_val).round(2)

    sku_median = purchases.groupby("stock_code")["price"].median()
    disc = purchases["price"] < f_cfg["discount_threshold"] * purchases["stock_code"].map(sku_median)
    feats["discount_sensitivity"] = disc.groupby(purchases["customer_id"]).mean()

    spend_by_cat = purchases.pivot_table(index="customer_id", columns="category", values="line_total", aggfunc="sum", fill_value=0.0)
    spend_by_cat = spend_by_cat.reindex(columns=CATEGORIES, fill_value=0.0)
    aff = spend_by_cat.div(spend_by_cat.sum(axis=1).replace(0, 1), axis=0)
    aff.columns = [f"aff_{c}" for c in aff.columns]
    feats = feats.join(aff)

    country = purchases.sort_values("invoice_date").groupby("customer_id")["country"].last()
    feats = feats.join(encode_country(country.reindex(feats.index), country_levels))

    feats[["gap_mean", "gap_std"]] = feats[["gap_mean", "gap_std"]].fillna(0.0)
    feats["purchase_trend"] = feats["purchase_trend"].fillna(0.0)

    if reference is None:
        reference = fit_reference(feats)
    feats["engagement_score"] = engagement_score(feats, reference)
    feats["country"] = country.reindex(feats.index)

    if with_labels:
        horizon_end = cutoff + pd.Timedelta(days=f_cfg["churn_horizon_days"])
        fut = lines[(lines["invoice_date"] >= cutoff) & (lines["invoice_date"] < horizon_end) & lines["customer_id"].notna()].copy()
        fut["customer_id"] = fut["customer_id"].astype("int64")
        fut_buy = fut[~fut["is_return"]]
        bought = set(fut_buy["customer_id"])
        feats["churn"] = (~feats.index.isin(bought)).astype(int)
        net = fut.groupby("customer_id")["line_total"].sum()
        feats["clv_target"] = net.reindex(feats.index).fillna(0.0).clip(lower=0.0)
        first_inv = fut_buy.sort_values("invoice_date").groupby("customer_id")["invoice"].first()
        first_lines = fut_buy.merge(
            first_inv.rename("first_invoice"), left_on=["customer_id", "invoice"], right_on=[first_inv.index.name, "first_invoice"]
        )
        nxt = first_lines.groupby("customer_id").apply(lambda d: d.groupby("category")["line_total"].sum().idxmax(), include_groups=False)
        feats["next_category"] = nxt.reindex(feats.index)

    feats.index.name = "customer_id"
    return feats, reference


def build_sequences(lines: pd.DataFrame, cutoff: pd.Timestamp, customer_ids: pd.Index, max_len: int) -> np.ndarray:
    """Time-ordered purchase sequences, shape (n_customers, max_len, 3 + n_categories), left-padded.

    Per order: log1p(order value), log1p(gap in days since previous order), log1p(days before the
    cutoff), one-hot of the order's top category. Uses only rows strictly before ``cutoff``.
    """
    cutoff = pd.Timestamp(cutoff)
    d = lines[(lines["invoice_date"] < cutoff) & ~lines["is_return"] & lines["customer_id"].notna()].copy()
    d["customer_id"] = d["customer_id"].astype("int64")
    d = d[d["customer_id"].isin(customer_ids)]
    cat_idx = {c: i for i, c in enumerate(CATEGORIES)}
    per_cat = d.groupby(["customer_id", "invoice", "category"])["line_total"].sum().reset_index()
    top = per_cat.sort_values("line_total", ascending=False).drop_duplicates(["customer_id", "invoice"])[["customer_id", "invoice", "category"]]
    orders = (
        d.groupby(["customer_id", "invoice"])
        .agg(date=("invoice_date", "min"), total=("line_total", "sum"))
        .reset_index()
        .merge(top, on=["customer_id", "invoice"])
    )
    orders = orders.sort_values(["customer_id", "date"])
    orders["gap"] = orders.groupby("customer_id")["date"].diff().dt.total_seconds().div(86400).fillna(0)
    n_feat = 3 + len(CATEGORIES)
    out = np.zeros((len(customer_ids), max_len, n_feat), dtype="float32")
    pos = {cid: i for i, cid in enumerate(customer_ids)}
    for cid, grp in orders.groupby("customer_id"):
        grp = grp.tail(max_len)
        arr = np.zeros((len(grp), n_feat), dtype="float32")
        arr[:, 0] = np.log1p(np.clip(grp["total"].to_numpy(), 0, None))
        arr[:, 1] = np.log1p(grp["gap"].to_numpy())
        arr[:, 2] = np.log1p((cutoff - grp["date"]).dt.total_seconds().to_numpy() / 86400)
        arr[np.arange(len(grp)), 3 + grp["category"].map(cat_idx).to_numpy()] = 1.0
        out[pos[cid], max_len - len(grp) :, :] = arr
    return out
