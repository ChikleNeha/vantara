"""API integration tests against the artifacts and database produced by the tiny pipeline run."""

from __future__ import annotations

import io
import time

import pandas as pd
import pytest

from src.data.synthetic import generate_transactions

GOOD = {"recency_days": 200, "frequency": 3, "monetary_total": 450, "tenure_days": 400}


def test_health_and_metadata(client):
    h = client.get("/api/health").json()
    assert h["status"] == "ok" and h["model_loaded"] and h["database_ok"] and h["scored_customers"] > 100
    m = client.get("/api/model/metadata").json()
    assert m["production_model"] in {"random_forest", "xgboost", "lightgbm"} and len(m["comparison_table"]) == 8
    assert 0 < m["threshold"] < 1 and m["data_source"] == "synthetic"


def test_openapi_docs_enabled(client):
    spec = client.get("/openapi.json").json()
    assert "/api/predict" in spec["paths"] and "/api/predict/batch" in spec["paths"]
    assert client.get("/docs").status_code == 200


def test_predict_single_returns_explained_prediction(client):
    r = client.post("/api/predict", json={**GOOD, "customer_id": 42, "country": "France", "category_affinity": {"kitchen": 0.6}})
    assert r.status_code == 200 and r.headers["X-Request-ID"]
    b = r.json()
    assert 0 <= b["churn_probability"] <= 1 and b["risk_band"] in {"low", "medium", "high"} and b["customer_id"] == 42
    assert len(b["drivers"]) == 5 and b["narrative"] and b["narrative_source"] == "template" and b["clv_pred_90d"] >= 0


def test_predict_orders_obvious_cases(client):
    lapsed = client.post("/api/predict", json={"recency_days": 500, "frequency": 1, "monetary_total": 15, "tenure_days": 500}).json()
    loyal = client.post("/api/predict", json={"recency_days": 5, "frequency": 40, "monetary_total": 8000, "tenure_days": 600}).json()
    assert lapsed["churn_probability"] > loyal["churn_probability"]


@pytest.mark.parametrize(
    "bad,field",
    [
        ({**GOOD, "frequency": 0}, "frequency"),
        ({**GOOD, "monetary_total": -1}, "monetary_total"),
        ({**GOOD, "return_rate": 1.5}, "return_rate"),
        ({**GOOD, "typo_field": 1}, "typo_field"),
        ({"recency_days": 10}, "frequency"),
        ({**GOOD, "recency_days": 900, "tenure_days": 100}, ""),
        ({**GOOD, "category_affinity": {"nonsense": 0.5}}, ""),
    ],
)
def test_predict_validation_messages(client, bad, field):
    r = client.post("/api/predict", json=bad)
    assert r.status_code == 422
    body = r.json()
    assert body["detail"] == "Invalid request" and body["errors"]
    if field:
        assert any(e["field"] == field for e in body["errors"])


def test_predict_p95_latency_under_prd_target(client):
    times = []
    for _ in range(40):
        t0 = time.perf_counter()
        assert client.post("/api/predict", json=GOOD).status_code == 200
        times.append((time.perf_counter() - t0) * 1000)
    assert sorted(times)[int(0.95 * len(times)) - 1] < 400


def test_llm_narrative_falls_back_when_ollama_is_down(client):
    b = client.post("/api/predict?narrative=llm", json=GOOD).json()
    assert b["narrative_source"] == "template" and b["narrative"]
    assert client.get("/api/ollama/status").json()["available"] is False
    r = client.post("/api/assistant/ask", json={"question": "Which segment is riskiest?"})
    assert r.status_code == 503 and "Ollama" in r.json()["detail"]
    assert client.post("/api/assistant/ask", json={"question": "x"}).status_code == 422


def test_batch_with_transaction_csv(client):
    raw = generate_transactions(150, seed=7)
    buf = io.StringIO()
    raw.to_csv(buf, index=False)
    r = client.post("/api/predict/batch", files={"file": ("tx.csv", buf.getvalue(), "text/csv")})
    assert r.status_code == 200
    b = r.json()
    assert b["summary"]["mode"] == "transactions" and b["summary"]["customers_scored"] > 20 and b["cleaning_report"]["raw_rows"] == len(raw)
    scores = [x["priority_score"] for x in b["rows"]]
    assert scores == sorted(scores, reverse=True) and {"churn_prob", "risk_band", "segment_label"} <= set(b["rows"][0])


def test_batch_with_customer_records_csv(client):
    csv = pd.DataFrame(
        [{"customer_id": 1, **GOOD}, {"customer_id": 2, "recency_days": 3, "frequency": 25, "monetary_total": 5000, "tenure_days": 500}]
    ).to_csv(index=False)
    r = client.post("/api/predict/batch", files={"file": ("c.csv", csv, "text/csv")})
    assert r.status_code == 200 and r.json()["summary"]["mode"] == "customer_records" and r.json()["summary"]["customers_scored"] == 2


@pytest.mark.parametrize(
    "content,status,needle",
    [
        ("", 400, "empty"),
        ("a,b\n1,2\n", 422, "Invalid customer rows"),
        ("Invoice,InvoiceDate,Quantity,Price\n1,2020-01-01,1,1\n", 422, "missing columns"),
        ("recency_days,frequency,monetary_total,tenure_days\n10,0,5,20\n", 422, "Invalid customer rows"),
    ],
)
def test_batch_error_handling(client, content, status, needle):
    r = client.post("/api/predict/batch", files={"file": ("f.csv", content, "text/csv")})
    assert r.status_code == status and needle in r.json()["detail"]


def test_overview_filters_segments_revenue(client):
    ov = client.get("/api/overview").json()
    k = ov["kpis"]
    assert k["customers"] > 100 and sum(ov["risk_bands"].values()) == k["customers"] and ov["data_source"] == "synthetic"
    f = client.get("/api/filters").json()
    assert "United Kingdom" in f["countries"] and f["value_tiers"] == ["high", "mid", "low"] and f["segments"]
    seg = client.get("/api/segments").json()
    assert sum(r["customers"] for r in seg["live"]) == k["customers"] and seg["profiles"]
    uk = client.get("/api/segments", params={"country": "United Kingdom", "value_tier": "high"}).json()
    assert sum(r["customers"] for r in uk["live"]) < k["customers"]
    pts = client.get("/api/segments/points", params={"limit": 100}).json()["points"]
    assert len(pts) == 100 and {"x", "y", "segment"} <= set(pts[0])
    rev = client.get("/api/revenue").json()["rows"]
    assert any(r["is_forecast"] for r in rev) and len({r["month"] for r in rev}) == len(rev)


def test_leaderboard_sorting_filters_pagination_and_injection_safety(client):
    lb = client.get("/api/customers/leaderboard", params={"limit": 10}).json()
    s = [r["priority_score"] for r in lb["rows"]]
    assert s == sorted(s, reverse=True) and lb["total"] > 100
    hi = client.get("/api/customers/leaderboard", params={"risk_band": "high", "limit": 50}).json()
    assert hi["rows"] and {r["risk_band"] for r in hi["rows"]} == {"high"}
    p2 = client.get("/api/customers/leaderboard", params={"limit": 10, "offset": 10}).json()
    assert p2["rows"][0]["customer_id"] not in {r["customer_id"] for r in lb["rows"]}
    inj = client.get("/api/customers/leaderboard", params={"country": "x' OR '1'='1"}).json()
    assert inj["total"] == 0
    assert client.get("/api/customers/leaderboard", params={"limit": 5000}).status_code == 422


def test_customer_detail_and_explanations(client):
    cid = client.get("/api/customers/leaderboard", params={"limit": 1}).json()["rows"][0]["customer_id"]
    d = client.get(f"/api/customers/{cid}").json()
    assert d["profile"]["customer_id"] == cid and "features_json" not in d["profile"] and d["recent_lines"]
    e = client.get(f"/api/customers/{cid}/explain").json()
    assert e["method"] == "shap" and len(e["drivers"]) == 5 and abs(e["churn_probability"] - d["profile"]["churn_prob"]) < 1e-9
    lime = client.get(f"/api/customers/{cid}/explain", params={"method": "lime"}).json()
    assert len(lime["lime"]) == 5 and all("rule" in x and "weight" in x for x in lime["lime"])
    assert client.get(f"/api/customers/{cid}/explain", params={"method": "bogus"}).status_code == 422


def test_unknown_customer_is_a_clear_404(client):
    r = client.get("/api/customers/999999999")
    assert r.status_code == 404 and "not found" in r.json()["detail"]
    assert client.get("/api/customers/abc").status_code == 422


def test_global_explanations(client):
    g = client.get("/api/explain/global").json()
    assert g["importance"] and g["importance"][0]["importance"] >= g["importance"][-1]["importance"] and len(g["pdp"]) == 3
    assert 0 <= g["shap_lime_top5_agreement"] <= 1


def test_degraded_mode_returns_actionable_503(client):
    st = client.app.state.cbp
    saved, st.bundle = st.bundle, None
    try:
        r = client.post("/api/predict", json=GOOD)
        assert r.status_code == 503 and "python -m src.pipeline" in r.json()["detail"]
        assert client.get("/api/health").json()["status"] == "degraded"
    finally:
        st.bundle = saved
