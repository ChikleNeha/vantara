"""Monthly revenue history and a simple seasonal forecast (PRD 12.2 forecast overlay)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def monthly_revenue(clean: pd.DataFrame) -> pd.DataFrame:
    """Net revenue (purchases minus returns), order and customer counts per month; flags a partial last month."""
    d = clean.copy()
    d["month"] = d["invoice_date"].dt.to_period("M")
    rev = d.groupby("month")["line_total"].sum()
    orders = d[~d["is_return"]].groupby("month")["invoice"].nunique()
    cust = d[~d["is_return"] & d["customer_id"].notna()].groupby("month")["customer_id"].nunique()
    out = pd.DataFrame({"revenue": rev, "orders": orders, "customers": cust}).fillna(0)
    out["is_partial"] = False
    last = out.index.max()
    if d["invoice_date"].max() < last.end_time.normalize() - pd.Timedelta(days=3):
        out.loc[last, "is_partial"] = True
    out["is_forecast"] = False
    return out


def seasonal_forecast(hist: pd.DataFrame, months_ahead: int) -> pd.DataFrame:
    """Seasonal-naive forecast scaled by recent year-over-year growth.

    forecast(m) = revenue(m - 12) x g, where g = (last 6 full months) / (same 6 months a year earlier),
    clipped to [0.5, 1.5]. Needs at least 18 full months; otherwise falls back to the trailing mean.
    The forecast starts the month after the last actual row (a partial last month stays an actual).
    """
    full = hist[~hist["is_partial"]]["revenue"]
    idx = pd.period_range(hist.index.max() + 1, periods=months_ahead, freq="M")
    if len(full) >= 18:
        g = float(np.clip(full.iloc[-6:].sum() / max(full.iloc[-18:-12].sum(), 1e-9), 0.5, 1.5))
        vals = [float(full.get(p - 12, full.iloc[-12:].mean()) * g) for p in idx]
    else:
        vals = [float(full.tail(6).mean())] * months_ahead
    out = pd.DataFrame({"revenue": vals, "orders": np.nan, "customers": np.nan, "is_partial": False, "is_forecast": True}, index=idx)
    return out
