"""Portfolio overview, segments, revenue trend, filters and global explanations."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from api.deps import AppState, clean_records, query, state
from api.routers.customers import BASE, where_clause

router = APIRouter(prefix="/api", tags=["analytics"])


@router.get("/overview")
def overview(st: AppState = Depends(state)) -> dict:
    """Headline KPIs for the dashboard home."""
    k = query(
        st.engine,
        f"""SELECT COUNT(*) AS customers, SUM(p.risk_band = 'high') AS high_risk, SUM(p.churn_flag) AS flagged, SUM(p.is_anomaly) AS anomalies,
                   AVG(p.churn_prob) AS avg_churn_prob, SUM(p.priority_score) AS spend_at_risk, SUM(c.clv_historical) AS lifetime_revenue {BASE}""",
    ).iloc[0]
    bands = query(st.engine, "SELECT risk_band, COUNT(*) AS n FROM predictions GROUP BY risk_band")
    m = st.metrics
    return {
        "kpis": {kk: float(vv) for kk, vv in k.items()},
        "risk_bands": {r.risk_band: int(r.n) for r in bands.itertuples()},
        "data_source": st.bundle.meta["data_source"],
        "production_model": st.bundle.meta["production_model"],
        "test_auc": m.get("classical", {}).get(st.bundle.meta["production_model"], {}).get("test", {}).get("roc_auc"),
        "test_recall": m.get("classical", {}).get(st.bundle.meta["production_model"], {}).get("test", {}).get("recall"),
    }


@router.get("/filters")
def filters(st: AppState = Depends(state)) -> dict:
    """Distinct values for the dashboard filter controls."""
    seg = query(st.engine, "SELECT DISTINCT segment_label FROM segments ORDER BY 1")["segment_label"].tolist()
    ctry = query(st.engine, "SELECT country, COUNT(*) AS n FROM customers GROUP BY country ORDER BY n DESC")["country"].tolist()
    return {"segments": seg, "countries": ctry, "value_tiers": ["high", "mid", "low"], "risk_bands": ["high", "medium", "low"]}


@router.get("/segments")
def segments(country: str | None = None, value_tier: str | None = None, st: AppState = Depends(state)) -> dict:
    """Segment profiles with live counts and risk for the chosen country / value-tier filters."""
    where, params = where_clause({"country": country, "value_tier": value_tier, "segment": None, "risk_band": None})
    live = query(
        st.engine,
        f"""SELECT s.segment_label AS label, COUNT(*) AS customers, AVG(p.churn_prob) AS avg_churn_prob, SUM(p.priority_score) AS spend_at_risk,
                   AVG(c.monetary_total) AS avg_spend, AVG(c.frequency) AS avg_orders, AVG(c.engagement_score) AS avg_engagement {BASE}{where} GROUP BY s.segment_label ORDER BY customers DESC""",
        params,
    )
    prof = query(st.engine, "SELECT * FROM segment_profiles")
    return {
        "live": clean_records(live),
        "profiles": clean_records(prof),
        "quality": st.metrics.get("segmentation", {}).get("quality", {}),
        "sweep": st.metrics.get("segmentation", {}).get("sweep", []),
    }


@router.get("/segments/points")
def segment_points(
    country: str | None = None,
    value_tier: str | None = None,
    segment: str | None = None,
    limit: int = Query(1500, ge=100, le=5000),
    st: AppState = Depends(state),
) -> dict:
    """PCA scatter coordinates (sampled) for the segment map."""
    where, params = where_clause({"country": country, "value_tier": value_tier, "segment": segment, "risk_band": None})
    df = query(
        st.engine,
        f"SELECT s.pca_x AS x, s.pca_y AS y, s.segment_label AS segment, c.customer_id {BASE}{where} ORDER BY RANDOM() LIMIT :limit",
        {**params, "limit": limit},
    )
    return {"points": clean_records(df)}


@router.get("/revenue")
def revenue(st: AppState = Depends(state)) -> dict:
    """Monthly net revenue with the forecast overlay."""
    df = query(st.engine, "SELECT * FROM monthly_revenue ORDER BY month")
    return {"rows": clean_records(df), "method": "Seasonal naive scaled by trailing 6-month year-over-year growth"}


@router.get("/explain/global")
def explain_global(st: AppState = Depends(state)) -> dict:
    """Global feature importance (mean |SHAP|), partial-dependence curves and SHAP/LIME agreement."""
    e = st.metrics.get("explain", {})
    return {"importance": e.get("global_importance", []), "pdp": e.get("pdp", {}), "shap_lime_top5_agreement": e.get("shap_lime_top5_agreement")}
