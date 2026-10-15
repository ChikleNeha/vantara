"""Customer lookup, leaderboard and per-customer explanations."""

from __future__ import annotations

import json
from typing import Any

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query

from api.deps import AppState, clean_records, query, state
from api.routers.predict import llm_narrative

router = APIRouter(prefix="/api/customers", tags=["customers"])

FILTER_COLS = {"segment": "s.segment_label", "country": "c.country", "value_tier": "c.value_tier", "risk_band": "p.risk_band"}
BASE = "FROM customers c JOIN predictions p ON p.customer_id = c.customer_id JOIN segments s ON s.customer_id = c.customer_id"


def where_clause(filters: dict[str, str | None]) -> tuple[str, dict[str, Any]]:
    """Build a parameterised WHERE from whitelisted columns (never interpolates user text)."""
    parts, params = [], {}
    for key, val in filters.items():
        if val:
            parts.append(f"{FILTER_COLS[key]} = :{key}")
            params[key] = val
    return (" WHERE " + " AND ".join(parts)) if parts else "", params


@router.get("/leaderboard")
def leaderboard(
    limit: int = Query(25, ge=1, le=200),
    offset: int = Query(0, ge=0),
    segment: str | None = None,
    country: str | None = None,
    value_tier: str | None = None,
    risk_band: str | None = None,
    st: AppState = Depends(state),
) -> dict:
    """Customers ranked by expected spend at risk (churn probability x typical 90-day spend), high-value high-risk first."""
    where, params = where_clause({"segment": segment, "country": country, "value_tier": value_tier, "risk_band": risk_band})
    total = int(query(st.engine, f"SELECT COUNT(*) AS n {BASE}{where}", params)["n"].iloc[0])
    df = query(
        st.engine,
        f"""SELECT c.customer_id, c.country, c.value_tier, c.frequency, c.monetary_total, c.avg_spend_90d, p.churn_prob, p.risk_band, p.priority_score,
                   p.clv_pred_90d, p.is_anomaly, p.next_category, s.segment_label {BASE}{where}
            ORDER BY p.priority_score DESC LIMIT :limit OFFSET :offset""",
        {**params, "limit": limit, "offset": offset},
    )
    return {"total": total, "rows": clean_records(df)}


def _get_customer(st: AppState, cid: int) -> pd.Series:
    df = query(
        st.engine,
        f"SELECT c.*, p.churn_prob, p.churn_flag, p.risk_band, p.clv_pred_90d, p.priority_score, p.ann_prob, p.lstm_prob, p.anomaly_score, p.is_anomaly, p.next_category, s.segment_label {BASE} WHERE c.customer_id = :id",
        {"id": cid},
    )
    if df.empty:
        raise HTTPException(404, f"Customer {cid} not found. Use the leaderboard to find valid IDs.")
    return df.iloc[0]


@router.get("/{customer_id}")
def customer_detail(customer_id: int, st: AppState = Depends(state)) -> dict:
    """Profile, prediction, segment and recent orders for one customer."""
    row = _get_customer(st, customer_id)
    feats = json.loads(row["features_json"])
    tx = query(
        st.engine,
        "SELECT invoice, invoice_date, description, category, quantity, price, line_total, is_return FROM transactions WHERE customer_id = :id ORDER BY invoice_date DESC LIMIT 15",
        {"id": customer_id},
    )
    aff = {k[4:]: v for k, v in feats.items() if k.startswith("aff_") and v > 0}
    prof = row.drop(labels=["features_json"]).to_dict()
    return {
        "profile": json.loads(pd.Series(prof).to_json(date_format="iso")),
        "category_affinity": dict(sorted(aff.items(), key=lambda kv: -kv[1])),
        "recent_lines": clean_records(tx),
        "threshold": st.bundle.threshold,
    }


@router.get("/{customer_id}/explain")
def customer_explain(
    customer_id: int,
    method: str = Query("shap", pattern="^(shap|lime)$"),
    narrative: str = Query("template", pattern="^(template|llm)$"),
    st: AppState = Depends(state),
) -> dict:
    """SHAP drivers (fast) or LIME rules (slower cross-check), with a plain-language narrative."""
    row = _get_customer(st, customer_id)
    feats = pd.DataFrame([json.loads(row["features_json"])], index=[customer_id])
    scored = pd.Series({"churn_prob": row["churn_prob"], "risk_band": row["risk_band"]})
    exp = st.bundle.explain(feats, customer_id, scored)
    out: dict[str, Any] = {
        "customer_id": customer_id,
        "churn_probability": float(row["churn_prob"]),
        "risk_band": row["risk_band"],
        "base_value": exp["base_value"],
        "drivers": exp["drivers"],
        "narrative": exp["narrative"],
        "narrative_source": "template",
        "method": method,
    }
    if method == "lime":
        try:
            out["lime"] = st.bundle.explainer.lime_drivers(feats.loc[customer_id])
        except RuntimeError as exc:
            raise HTTPException(503, str(exc)) from exc
    if narrative == "llm":
        out["narrative"], out["narrative_source"] = llm_narrative(st, customer_id, scored, exp)
    return out
