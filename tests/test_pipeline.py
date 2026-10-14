"""The pipeline end to end on a tiny synthetic dataset: artifacts, database, metrics, report."""

from __future__ import annotations

import json
import sqlite3

import pandas as pd

from src.serving.scoring import ModelBundle


def test_artifacts_and_logs_exist(pipeline_run):
    cfg, m = pipeline_run
    art = cfg.path("artifacts_dir")
    for f in (
        "meta.json",
        "metrics.json",
        "reference.joblib",
        "clv_model.joblib",
        "segmentation.joblib",
        "ann_numpy.npz",
        "autoencoder_numpy.npz",
        "ann.keras",
        "lstm.keras",
        "experiment_log.csv",
        "model_comparison.csv",
    ):
        assert (art / f).exists(), f
    assert len(list((art / "classical").glob("*.joblib"))) == 6
    assert {"ann", "lstm"} <= {r["model"] for r in m["comparison_table"]}
    assert len(m["comparison_table"]) == 8  # 6 classical + ANN + LSTM
    assert (cfg.path("docs_dir") / "REPORT.md").exists() and (cfg.path("docs_dir") / "eda_report.md").exists()
    for fig in ("ann_loss.png", "lstm_loss.png", "autoencoder_loss.png", "shap_summary.png"):
        assert (cfg.path("docs_dir") / "figures" / fig).exists()


def test_metrics_follow_prd_discipline(pipeline_run):
    _, m = pipeline_run
    prod = m["production_model"]
    assert prod in {"random_forest", "xgboost", "lightgbm"}
    assert m["classical"][prod]["val_auc"] == max(v["val_auc"] for k, v in m["classical"].items() if k in {"random_forest", "xgboost", "lightgbm"})
    for name, r in m["classical"].items():
        assert 0 < r["cv_auc"] <= 1 and {"roc_auc", "recall", "precision", "f1", "accuracy", "tn", "fp", "fn", "tp"} <= set(r["test"]), name
    assert m["deep"]["ann"]["numpy_parity"] < 1e-5  # NumPy inference copy equals Keras
    assert m["segmentation"]["best_k"] in (3, 4) and len(m["segmentation"]["profiles"]) == m["segmentation"]["best_k"]
    assert m["clv"]["best"] in {"ridge", "xgboost_direct", "hurdle"}
    assert m["latency_ms"]["p95"] < 400
    assert m["features"]["model_columns"] < m["features"]["all_columns"]  # VIF pruning happened


def test_database_is_populated_and_consistent(pipeline_run):
    cfg, m = pipeline_run
    con = sqlite3.connect(cfg.path("db_path"))
    n = lambda t: con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]  # noqa: E731
    assert n("customers") == n("predictions") == n("segments") == m["portfolio"]["customers"] > 100
    assert n("segment_profiles") == m["segmentation"]["best_k"] and n("transactions") > 1000
    assert con.execute("SELECT COUNT(*) FROM monthly_revenue WHERE is_forecast = 1").fetchone()[0] == cfg.forecast["months_ahead"]
    assert con.execute("SELECT MIN(churn_prob), MAX(churn_prob) FROM predictions").fetchone()[0] >= 0
    assert con.execute("PRAGMA foreign_key_check").fetchall() == []


def test_bundle_reloads_from_disk_and_scores(pipeline_run):
    cfg, _ = pipeline_run
    b = ModelBundle(cfg)
    feats = b.features_from_records(
        [
            {"recency_days": 400, "frequency": 1, "monetary_total": 20, "tenure_days": 400},
            {"recency_days": 5, "frequency": 30, "monetary_total": 9000, "tenure_days": 600},
        ]
    )
    s = b.score(feats)
    assert s.loc[0, "churn_prob"] > s.loc[1, "churn_prob"]  # lapsed one-off buyer is riskier than an active loyal one
    assert s["risk_band"].isin(["low", "medium", "high"]).all()
    ex = b.explain(feats, 0, s.loc[0])
    assert len(ex["drivers"]) == 5 and ex["narrative"].startswith("Customer 0")


def test_report_reflects_logged_numbers(pipeline_run):
    cfg, m = pipeline_run
    text = (cfg.path("docs_dir") / "REPORT.md").read_text()
    assert "SYNTHETIC" in text and m["production_model"] in text
    assert f"{m['classical'][m['production_model']]['test']['roc_auc']:.3f}" in text
    assert "Appendix A" in text and "Future enhancements" in text and "Method notes" in text


def test_experiment_log_has_every_run(pipeline_run):
    cfg, _ = pipeline_run
    log = pd.read_csv(cfg.path("artifacts_dir") / "experiment_log.csv")
    assert {"churn", "clv", "anomaly", "next_category"} <= set(log["task"])
    assert log["params"].map(json.loads).notna().all() and (log["train_seconds"] >= 0).all()


def test_eda_report_has_no_missing_numbers(pipeline_run):
    """Regression: hypotheses once printed 'nan' for a feature that VIF pruning had removed."""
    import re

    cfg, m = pipeline_run
    text = (cfg.path("docs_dir") / "eda_report.md").read_text()
    assert not re.search(r"[+-]?\bnan\b", text, flags=re.I)
    assert "H2." in text and len(m["eda"]["hypotheses"]) == 6
    assert "monetary_total" in m["eda"]["corr_churn"] or len(m["eda"]["corr_churn"]) >= 5
