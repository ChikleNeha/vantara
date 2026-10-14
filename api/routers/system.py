"""Health, model metadata and Ollama status endpoints."""

from __future__ import annotations

import time

from fastapi import APIRouter, Depends, Request
from sqlalchemy import text

from api.deps import AppState, state

router = APIRouter(prefix="/api", tags=["system"])


@router.get("/health")
def health(request: Request) -> dict:
    """Liveness plus readiness of the model bundle and database."""
    st: AppState | None = getattr(request.app.state, "cbp", None)
    db_ok = False
    n = 0
    if st is not None:
        try:
            with st.engine.connect() as c:
                n = c.execute(text("SELECT COUNT(*) FROM predictions")).scalar() or 0
            db_ok = True
        except Exception:  # noqa: BLE001 - health must never raise
            db_ok = False
    return {
        "status": "ok" if st and st.bundle and db_ok else "degraded",
        "model_loaded": bool(st and st.bundle),
        "database_ok": db_ok,
        "scored_customers": int(n),
        "uptime_seconds": round(time.time() - st.started, 1) if st else 0,
    }


@router.get("/model/metadata")
def model_metadata(st: AppState = Depends(state)) -> dict:
    """What is deployed: production model, threshold, data source, comparison table and success metrics."""
    m = st.metrics
    return {
        "production_model": st.bundle.meta["production_model"],
        "threshold": st.bundle.threshold,
        "trained_at": st.bundle.meta["trained_at"],
        "data_source": st.bundle.meta["data_source"],
        "cutoffs": st.bundle.meta["cutoffs"],
        "model_columns": st.bundle.model_cols,
        "comparison_table": m.get("comparison_table", []),
        "clv": m.get("clv", {}),
        "deep": m.get("deep", {}),
        "segmentation_quality": m.get("segmentation", {}).get("quality", {}),
        "latency_ms": m.get("latency_ms", {}),
        "vif_dropped": m.get("features", {}).get("vif_dropped", []),
    }


@router.get("/ollama/status")
def ollama_status(st: AppState = Depends(state)) -> dict:
    """Is Ollama reachable and is the configured model pulled?"""
    return st.ollama.status()
