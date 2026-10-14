"""Modelling utilities, forecast, segmentation naming, explanations and the Ollama client."""

from __future__ import annotations

import httpx
import numpy as np
import pandas as pd
import pytest

from src.explainability.explain import fmt_value, pretty, template_narrative
from src.forecast import monthly_revenue, seasonal_forecast
from src.models.common import ExperimentLog, classification_metrics, pick_threshold, regression_metrics, stratified_split
from src.segmentation.cluster import SEG_COLS, name_segments
from src.serving.ollama_client import OllamaClient, OllamaUnavailable
from src.serving.scoring import risk_band
from src.utils.config import load_config
from tests.conftest import tx

CFG = load_config()


def test_split_is_stratified_disjoint_and_reproducible():
    y = pd.Series(np.r_[np.ones(300), np.zeros(700)].astype(int), index=range(1000))
    tr, va, te = stratified_split(y, CFG)
    assert (len(tr), len(va), len(te)) == (700, 150, 150)
    assert not (set(tr) & set(va)) and not (set(tr) & set(te)) and not (set(va) & set(te))
    assert all(abs(y.loc[i].mean() - 0.3) < 0.02 for i in (tr, va, te))
    assert list(stratified_split(y, CFG)[0]) == list(tr)


def test_classification_metrics_match_hand_calculation():
    m = classification_metrics([1, 1, 0, 0], [0.9, 0.4, 0.6, 0.1], 0.5)
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (1, 1, 1, 1)
    assert m["precision"] == m["recall"] == 0.5 and m["roc_auc"] == 0.75


def test_pick_threshold_meets_recall_target_with_best_precision():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 2000)
    p = np.clip(y * 0.3 + rng.normal(0.4, 0.2, 2000), 0, 1)
    thr = pick_threshold(y, p, 0.7)
    assert classification_metrics(y, p, thr)["recall"] >= 0.7
    assert pick_threshold(y, np.zeros(2000), 0.7) is not None  # unreachable target falls back


def test_regression_metrics():
    m = regression_metrics([1, 2, 3], [1, 2, 4])
    assert m["mae"] == pytest.approx(1 / 3, abs=1e-3) and m["r2"] == pytest.approx(0.5)


def test_experiment_log_rebuilds_comparison(tmp_path):
    log = ExperimentLog(tmp_path / "l.csv")
    log.reset()
    log.log("churn", "a", {"x": 1}, 1.2, 0.7, 0.71, {"roc_auc": 0.8, "recall": 0.7})
    log.log("churn", "b", {}, 0.5, None, 0.8, {"roc_auc": 0.9, "recall": 0.6})
    log.log("clv", "r", {}, 0.1, test_metrics={"r2": 0.1})
    table = log.comparison_table("churn")
    assert table["model"].tolist() == ["b", "a"] and len(table) == 2


def test_monthly_revenue_flags_partial_month_and_forecast_does_not_overlap():
    rows = []
    for i, m in enumerate(pd.period_range("2020-01", "2021-09", freq="M")):
        rows.append((1, f"i{i}", m.start_time + pd.Timedelta(days=5), 1, 100.0 + i, "kitchen", False))
    rows.append((1, "last", "2021-10-04", 1, 50.0, "kitchen", False))  # data ends Oct 4: partial month
    hist = monthly_revenue(tx(rows))
    assert bool(hist["is_partial"].iloc[-1]) and not hist["is_partial"].iloc[:-1].any()
    fc = seasonal_forecast(hist, 4)
    assert len(fc) == 4 and fc.index.min() == hist.index.max() + 1 and fc["is_forecast"].all() and (fc["revenue"] > 0).all()
    assert not set(fc.index) & set(hist.index)


def test_forecast_short_history_falls_back_to_mean():
    hist = monthly_revenue(
        tx([(1, f"i{i}", f"2021-{i:02d}-15", 1, 10.0, "kitchen", False) for i in range(1, 7)] + [(1, "z", "2021-06-30", 1, 10.0, "kitchen", False)])
    )
    fc = seasonal_forecast(hist, 2)
    assert fc["revenue"].nunique() == 1


def test_segment_naming_is_unique_and_business_readable():
    prof = pd.DataFrame(
        {
            "recency_days": [20, 300, 25, 200],
            "frequency": [15, 2, 2, 6],
            "monetary_total": [5000, 80, 120, 2000],
            "avg_basket_size": [10] * 4,
            "return_rate": [0.02, 0.02, 0.02, 0.4],
            "discount_sensitivity": [0.1] * 4,
            "seasonal_concentration": [0.2] * 4,
            "purchase_trend": [0.0] * 4,
        }
    )
    names = name_segments(prof, prof[SEG_COLS].mean())
    labels = [v[0] for v in names.values()]
    assert len(set(labels)) == 4 and "Loyal champions" in labels and "Return-heavy buyers" in labels
    assert all(len(v[1]) > 20 for v in names.values())


def test_risk_band_boundaries():
    assert [risk_band(p, 0.5, 0.75) for p in (0.2, 0.5, 0.74, 0.75, 0.99)] == ["low", "medium", "medium", "high", "high"]
    assert risk_band(0.8, 0.9, 0.75) == "low" and risk_band(0.95, 0.9, 0.75) == "high"  # never above "low" below the model threshold


def test_narrative_and_formatting():
    drivers = [
        {"feature": "recency_days", "label": "days since last purchase", "value_text": "200 days", "shap": 0.2},
        {"feature": "frequency", "label": "number of orders", "value_text": "9 orders", "shap": -0.1},
    ]
    high = template_narrative(7, 0.82, "high", drivers)
    assert "82%" in high and "win-back" in high and "Helping retention" in high
    assert "no urgent intervention" in template_narrative(7, 0.2, "low", drivers) and "win-back" not in template_narrative(7, 0.2, "low", drivers)
    assert fmt_value("return_rate", 0.123) == "12%" and fmt_value("recency_days", 12.4) == "12 days" and fmt_value("frequency", 3) == "3 orders"
    assert pretty("aff_home_decor") == "share of spend in home decor" and pretty("country_France") == "customer located in France"


def test_ollama_narrate_status_and_failure(monkeypatch):
    def ok(req: httpx.Request) -> httpx.Response:
        if req.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": "llama3.2:latest"}]})
        return httpx.Response(200, json={"message": {"content": " Short explanation. "}})

    def boom(req: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    for handler, expect in ((ok, True), (boom, False)):
        transport = httpx.MockTransport(handler)
        monkeypatch.setattr(httpx, "post", lambda url, transport=transport, **kw: httpx.Client(transport=transport).post(url, **kw))
        monkeypatch.setattr(httpx, "get", lambda url, transport=transport, **kw: httpx.Client(transport=transport).get(url, **kw))
        c = OllamaClient(CFG)
        st = c.status()
        assert st["available"] is expect
        if expect:
            assert st["model_pulled"] and c.narrate({"a": 1}) == "Short explanation." and c.ask("q?", {"k": 1}) == "Short explanation."
        else:
            with pytest.raises(OllamaUnavailable):
                c.narrate({"a": 1})
