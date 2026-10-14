"""Single-customer and batch prediction endpoints."""

from __future__ import annotations

import io
from typing import Any

import pandas as pd
from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile

from api.deps import AppState, clean_records, log, state
from api.schemas.models import CustomerRecord, PredictionResponse
from src.serving.ollama_client import OllamaUnavailable

router = APIRouter(prefix="/api/predict", tags=["prediction"])
MAX_UPLOAD_BYTES = 30 * 1024 * 1024
TX_COLS = {"invoice", "invoicedate", "quantity", "price"}


def llm_narrative(st: AppState, customer_id: Any, scored: pd.Series, exp: dict) -> tuple[str, str]:
    """Ask Ollama to rewrite the facts; fall back to the deterministic template if it is unavailable."""
    facts = {
        "customer_id": customer_id,
        "churn_probability_pct": round(float(scored["churn_prob"]) * 100),
        "risk_band": scored["risk_band"],
        "drivers": [{"factor": d["label"], "value": d["value_text"], "effect": d["direction"]} for d in exp["drivers"]],
    }
    try:
        return st.ollama.narrate(facts), "ollama"
    except OllamaUnavailable as exc:
        log.warning("Ollama unavailable, using template narrative: %s", str(exc)[:100])
        return exp["narrative"], "template"


@router.post("", response_model=PredictionResponse)
def predict_one(
    rec: CustomerRecord, narrative: str = Query("template", pattern="^(template|llm)$"), st: AppState = Depends(state)
) -> PredictionResponse:
    """Score one customer record and explain the result."""
    feats = st.bundle.features_from_records([rec.to_row()])
    scored = st.bundle.score(feats).iloc[0]
    exp = st.bundle.explain(feats, feats.index[0], scored)
    text_, src = (exp["narrative"], "template") if narrative == "template" else llm_narrative(st, rec.customer_id or "this customer", scored, exp)
    return PredictionResponse(
        customer_id=rec.customer_id,
        churn_probability=float(scored["churn_prob"]),
        churn_flag=bool(scored["churn_flag"]),
        risk_band=scored["risk_band"],
        threshold=st.bundle.threshold,
        clv_pred_90d=float(scored["clv_pred_90d"]),
        segment=str(scored["segment_label"]),
        next_category=None if pd.isna(scored["next_category"]) else str(scored["next_category"]),
        anomaly=bool(scored["is_anomaly"]),
        ann_probability=float(scored["ann_prob"]),
        model=st.bundle.meta["production_model"],
        drivers=exp["drivers"],
        narrative=text_,
        narrative_source=src,
    )


@router.post("/batch")
async def predict_batch(file: UploadFile = File(...), st: AppState = Depends(state)) -> dict:
    """Score a CSV upload. Accepts either raw transactions (Online Retail II columns) or customer feature records."""
    raw = await file.read()
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"File too large ({len(raw) / 1e6:.0f} MB); limit is {MAX_UPLOAD_BYTES // 1_000_000} MB")
    if not raw.strip():
        raise HTTPException(400, "The uploaded file is empty")
    try:
        df = pd.read_csv(io.BytesIO(raw))
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(400, f"Could not parse CSV: {str(exc)[:150]}") from exc
    norm = {c.lower().replace(" ", "").replace("_", ""): c for c in df.columns}
    if "invoiceno" in norm:
        norm.setdefault("invoice", norm["invoiceno"])

    report: dict[str, Any] = {}
    if TX_COLS <= set(norm):
        mode = "transactions"
        need = {"stockcode", "customerid", "country", "description"}
        missing = sorted(need - set(norm))
        if missing:
            raise HTTPException(
                422,
                f"Transaction CSV is missing columns: {missing}. Expected Online Retail II columns (Invoice, StockCode, Description, Quantity, InvoiceDate, Price, Customer ID, Country).",
            )
        try:
            canon = {
                "invoice": "Invoice",
                "invoicedate": "InvoiceDate",
                "customerid": "Customer ID",
                "stockcode": "StockCode",
                "quantity": "Quantity",
                "price": "Price",
                "country": "Country",
                "description": "Description",
            }
            feats, report = st.bundle.features_from_transactions(df.rename(columns={norm[k]: v for k, v in canon.items() if k in norm}))
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(422, f"Could not build customer features from the transactions: {str(exc)[:200]}") from exc
    else:
        mode = "customer_records"
        try:
            recs = [CustomerRecord(**{k: v for k, v in r.items() if pd.notna(v)}) for r in df.to_dict("records")]
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(422, f"Invalid customer rows: {str(exc)[:300]}") from exc
        ids = [r.customer_id if r.customer_id is not None else i + 1 for i, r in enumerate(recs)]
        feats = st.bundle.features_from_records([r.to_row() for r in recs])
        feats.index = pd.Index(ids, name="customer_id")
    if feats.empty:
        raise HTTPException(422, "No scorable customers found in the file")
    scored = st.bundle.score(feats)
    out = (
        scored[
            [
                "churn_prob",
                "churn_flag",
                "risk_band",
                "clv_pred_90d",
                "avg_spend_90d",
                "priority_score",
                "value_tier",
                "segment_label",
                "next_category",
                "ann_prob",
                "is_anomaly",
            ]
        ]
        .reset_index()
        .sort_values("priority_score", ascending=False)
    )
    summary = {
        "mode": mode,
        "customers_scored": int(len(out)),
        "high_risk": int((out["risk_band"] == "high").sum()),
        "flagged_churn": int(out["churn_flag"].sum()),
        "avg_churn_prob": round(float(out["churn_prob"].mean()), 4),
        "spend_at_risk_90d": round(float(out["priority_score"].sum()), 2),
    }
    log.info("Batch scored %d customers (%s)", len(out), mode)
    return {"summary": summary, "cleaning_report": report, "rows": clean_records(out.head(20000))}
