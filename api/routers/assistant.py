"""Ollama-backed Q&A over the scored portfolio. The LLM only reads a compact context built from the database."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from api.deps import AppState, clean_records, query, state
from api.schemas.models import AskRequest
from src.serving.ollama_client import OllamaUnavailable

router = APIRouter(prefix="/api/assistant", tags=["assistant"])


def build_context(st: AppState) -> dict:
    """Small, factual snapshot of the portfolio (keeps prompts short and answers grounded)."""
    seg = query(
        st.engine,
        """SELECT s.segment_label AS segment, COUNT(*) AS customers, ROUND(AVG(p.churn_prob), 3) AS avg_churn_prob, ROUND(SUM(p.priority_score), 0) AS spend_at_risk
                              FROM segments s JOIN predictions p ON p.customer_id = s.customer_id GROUP BY s.segment_label""",
    )
    top = query(
        st.engine,
        """SELECT p.customer_id, ROUND(p.churn_prob, 3) AS churn_prob, p.risk_band, ROUND(p.priority_score, 0) AS spend_at_risk, s.segment_label AS segment, c.country
                              FROM predictions p JOIN segments s ON s.customer_id = p.customer_id JOIN customers c ON c.customer_id = p.customer_id ORDER BY p.priority_score DESC LIMIT 10""",
    )
    kpi = query(
        st.engine,
        "SELECT COUNT(*) AS customers, SUM(risk_band = 'high') AS high_risk, ROUND(AVG(churn_prob), 3) AS avg_churn_prob, ROUND(SUM(priority_score), 0) AS spend_at_risk FROM predictions",
    ).iloc[0]
    rev = query(st.engine, "SELECT month, ROUND(revenue, 0) AS revenue, is_forecast FROM monthly_revenue ORDER BY month DESC LIMIT 8")
    m = st.metrics
    return {
        "portfolio": {k: float(v) for k, v in kpi.items()},
        "segments": clean_records(seg),
        "top_priority_customers": clean_records(top),
        "recent_revenue": clean_records(rev),
        "model": {
            "name": st.bundle.meta["production_model"],
            "test_auc": m.get("classical", {}).get(st.bundle.meta["production_model"], {}).get("test", {}).get("roc_auc"),
            "threshold": st.bundle.threshold,
        },
        "top_churn_drivers": [d["label"] for d in m.get("explain", {}).get("global_importance", [])[:5]],
        "data_source": st.bundle.meta["data_source"],
    }


@router.post("/ask")
def ask(req: AskRequest, st: AppState = Depends(state)) -> dict:
    """Answer a free-text question using Ollama; returns 503 with guidance if Ollama is not running."""
    try:
        answer = st.ollama.ask(req.question, build_context(st))
    except OllamaUnavailable as exc:
        raise HTTPException(
            503,
            f"The local language model is not reachable ({str(exc)[:80]}). Start Ollama and run `ollama pull {st.ollama.model}`; all other dashboard features work without it.",
        ) from exc
    return {"answer": answer, "model": st.ollama.model}
