"""End-to-end pipeline: raw data -> features -> models -> segments -> explanations -> SQLite -> report.

Run:  python -m src.pipeline            (auto-detects real data, falls back to synthetic demo data)
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.data.cleaning import clean_transactions, customer_lines, top_countries
from src.data.loader import load_raw
from src.data.validation import validate_raw
from src.features.build import AFF_COLS, build_features, build_sequences, feature_columns
from src.features.eda import run_eda, select_by_vif
from src.forecast import monthly_revenue, seasonal_forecast
from src.models.classical import choose_production_model, save_models, train_classical_models
from src.models.clv import train_clv, train_next_category
from src.models.common import ExperimentLog, stratified_split
from src.models.deep import train_ann, train_autoencoder, train_lstm
from src.segmentation.cluster import fit_segments
from src.serving import db
from src.serving.scoring import ModelBundle
from src.utils.config import Config, load_config
from src.utils.logging import get_logger

warnings.filterwarnings("ignore", category=FutureWarning)
log = get_logger("pipeline")


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)), encoding="utf-8")


def run_pipeline(cfg: Config) -> dict[str, Any]:
    """Execute every stage and return the metrics dictionary (also written to models_artifacts/metrics.json)."""
    t_start = time.time()
    art, docs, interim, processed = cfg.path("artifacts_dir"), cfg.path("docs_dir"), cfg.path("interim_dir"), cfg.path("processed_dir")
    fig_dir = docs / "figures"
    for d in (art, docs, interim, processed, fig_dir):
        d.mkdir(parents=True, exist_ok=True)
    explog = ExperimentLog(art / "experiment_log.csv")
    explog.reset()
    metrics: dict[str, Any] = {"seed": cfg["seed"]}

    # 1. data ---------------------------------------------------------------------------
    raw, source = load_raw(cfg)
    metrics["data_source"] = source
    metrics["validation"] = validate_raw(raw, cfg)
    clean, clean_report = clean_transactions(raw, cfg)
    metrics["cleaning"] = clean_report
    lines = customer_lines(clean)
    clean.to_csv(interim / "clean_transactions.csv.gz", index=False)

    last_date = clean["invoice_date"].max()
    train_cutoff = last_date.normalize() - pd.Timedelta(days=cfg.features["churn_horizon_days"])
    score_cutoff = last_date.normalize() + pd.Timedelta(days=1)
    levels = top_countries(lines["country"], cfg.models["country_levels"])
    metrics["cutoffs"] = {
        "train_cutoff": str(train_cutoff.date()),
        "score_cutoff": str(score_cutoff.date()),
        "horizon_days": cfg.features["churn_horizon_days"],
    }

    # 2. features -----------------------------------------------------------------------
    feats, reference = build_features(lines, train_cutoff, cfg, levels)
    feats.to_csv(processed / "customer_features_train.csv.gz")
    all_cols = feature_columns(levels)
    y = feats["churn"]
    split = stratified_split(y, cfg)
    tr, va, te = split
    model_cols, vif_drops = select_by_vif(feats.loc[tr, all_cols], cfg.features["vif_threshold"])
    metrics["features"] = {
        "n_customers": int(len(feats)),
        "churn_rate": round(float(y.mean()), 4),
        "all_columns": len(all_cols),
        "model_columns": len(model_cols),
        "vif_dropped": vif_drops,
    }
    log.info("Features: %d customers, churn rate %.1f%%, %d/%d columns kept after VIF", len(feats), 100 * y.mean(), len(model_cols), len(all_cols))
    metrics["eda"] = run_eda(feats.loc[tr], clean, model_cols, fig_dir, docs / "eda_report.md", corr_cols=all_cols)

    # 3. classical churn models ---------------------------------------------------------
    X = feats[model_cols]
    res = train_classical_models(X, y, split, cfg, explog)
    prod = choose_production_model(res)
    save_models(res, art)
    metrics["classical"] = res["metrics"]
    metrics["production_model"] = prod
    threshold = res["thresholds"][prod]

    # 4. CLV and next category ----------------------------------------------------------
    clv = train_clv(X, feats["clv_target"], split, cfg, explog)
    joblib.dump(clv[clv["best"]]["model"], art / "clv_model.joblib")
    metrics["clv"] = {k: {kk: vv for kk, vv in v.items() if kk != "model"} for k, v in clv.items() if k != "best"} | {"best": clv["best"]}
    nxt = train_next_category(X, feats["next_category"], split, cfg, explog)
    if nxt:
        joblib.dump(nxt["model"], art / "next_category.joblib")
        metrics["next_category"] = nxt["metrics"]

    # 5. deep learning ------------------------------------------------------------------
    seqs = build_sequences(lines, train_cutoff, feats.index, cfg.deep_learning["lstm_max_len"])
    ann = train_ann(X, y, split, cfg, explog, art, fig_dir)
    lstm = train_lstm(seqs, y, split, cfg, explog, art, fig_dir)
    ae = train_autoencoder(feats[all_cols], split, cfg, explog, art, fig_dir)
    metrics["deep"] = {"ann": ann["metrics"], "lstm": lstm["metrics"], "autoencoder": ae["metrics"]}

    # error correlation between best tree model and ANN (justifies keeping ANN as a secondary check)
    err_tree = res["test_proba"][prod] - y.loc[te].to_numpy()
    err_ann = ann["test_proba"] - y.loc[te].to_numpy()
    metrics["ann_tree_error_correlation"] = round(float(np.corrcoef(err_tree, err_ann)[0, 1]), 4)

    # 6. artifacts needed by the serving bundle -----------------------------------------
    score_feats, _ = build_features(lines, score_cutoff, cfg, levels, reference=reference, with_labels=False)
    spend90 = (score_feats["clv_historical"] / score_feats["tenure_days"].clip(lower=90) * 90).clip(lower=0)
    defaults = {c: float(feats.loc[tr, c].mean() if c in AFF_COLS else feats.loc[tr, c].median()) for c in all_cols if not c.startswith("country_")}
    meta = {
        "production_model": prod,
        "threshold": threshold,
        "model_columns": model_cols,
        "all_columns": all_cols,
        "country_levels": levels,
        "feature_defaults": defaults,
        "anomaly_threshold": ae["threshold"],
        "value_cuts": {
            "high": float(spend90.quantile(cfg.risk["high_value_quantile"])),
            "mid": float(spend90.quantile(cfg.risk["mid_value_quantile"])),
        },
        "cutoffs": metrics["cutoffs"],
        "data_source": source,
        "trained_at": pd.Timestamp.now().isoformat(timespec="seconds"),
    }
    _write_json(art / "meta.json", meta)
    joblib.dump(reference, art / "reference.joblib")
    X.loc[tr].sample(min(800, len(tr)), random_state=cfg["seed"]).to_csv(art / "lime_background.csv.gz", index=False)

    # 7. segmentation on today's customers ------------------------------------------------
    seg = fit_segments(score_feats, cfg, art)
    metrics["segmentation"] = seg["report"]

    # 8. score everyone through the SAME bundle the API uses ---------------------------
    bundle = ModelBundle(cfg, art)
    scored = bundle.score(score_feats)
    import tensorflow as tf

    lstm_model = tf.keras.models.load_model(art / "lstm.keras")
    seq_now = build_sequences(lines, score_cutoff, score_feats.index, cfg.deep_learning["lstm_max_len"])
    scored["lstm_prob"] = lstm_model.predict(seq_now, verbose=0).ravel().round(4)
    scored = scored.drop(columns=["segment_id", "segment_label", "pca_x", "pca_y"]).join(seg["assignments"])
    metrics["portfolio"] = {
        "customers": int(len(scored)),
        "high_risk": int((scored["risk_band"] == "high").sum()),
        "flagged_churn": int(scored["churn_flag"].sum()),
        "anomalies": int(scored["is_anomaly"].sum()),
        "revenue_at_risk_90d": round(float(scored["priority_score"].sum()), 2),
    }

    # 9. explainability -----------------------------------------------------------------
    expl = bundle.explainer
    sample = X.loc[te].sample(min(400, len(te)), random_state=cfg["seed"])
    metrics["explain"] = {"global_importance": expl.global_importance(sample)}
    expl.save_summary_plot(sample, fig_dir / "shap_summary.png")
    top3 = [d["feature"] for d in metrics["explain"]["global_importance"][:3]]
    metrics["explain"]["pdp"] = expl.pdp(X.loc[tr], top3)
    overlaps = []
    if expl.lime is None:
        log.warning("LIME is not installed; skipping the SHAP vs LIME agreement check (python -m pip install lime)")
    else:
        for cid in X.loc[te].sample(min(15, len(te)), random_state=1).index:
            s_top = {d["feature"] for d in expl.top_drivers(X.loc[cid], expl.shap_values(X.loc[[cid]]).iloc[0], 5)}
            l_rules = [d["rule"] for d in expl.lime_drivers(X.loc[cid], 5)]
            overlaps.append(sum(any(f in rule for rule in l_rules) for f in s_top) / 5)
    metrics["explain"]["shap_lime_top5_agreement"] = round(float(np.mean(overlaps)), 3) if overlaps else None

    # 10. latency of the model path -----------------------------------------------------
    lat = []
    for cid in score_feats.sample(min(60, len(score_feats)), random_state=3).index:
        t0 = time.perf_counter()
        one = score_feats.loc[[cid]]
        row = bundle.score(one)
        bundle.explain(one, cid, row.iloc[0])
        lat.append((time.perf_counter() - t0) * 1000)
    metrics["latency_ms"] = {
        "p50": round(float(np.percentile(lat, 50)), 1),
        "p95": round(float(np.percentile(lat, 95)), 1),
        "note": "model + SHAP path, excludes HTTP",
    }

    # 11. revenue + forecast, then persist ----------------------------------------------
    hist = monthly_revenue(clean)
    fc = seasonal_forecast(hist, cfg.forecast["months_ahead"])
    rev = pd.concat([hist, fc]).reset_index().rename(columns={"index": "month"})
    rev["month"] = rev["month"].astype(str)

    engine = db.get_engine(cfg.path("db_path"))
    db.reset_schema(engine)
    cust = pd.DataFrame(
        {
            "customer_id": score_feats.index.astype("int64"),
            "country": score_feats["country"].to_numpy(),
            "first_purchase": (score_cutoff - pd.to_timedelta(score_feats["tenure_days"], unit="D")).to_numpy(),
            "last_purchase": (score_cutoff - pd.to_timedelta(score_feats["recency_days"], unit="D")).to_numpy(),
            "frequency": score_feats["frequency"].astype(int).to_numpy(),
            "monetary_total": score_feats["monetary_total"].round(2).to_numpy(),
            "clv_historical": score_feats["clv_historical"].to_numpy(),
            "avg_spend_90d": scored["avg_spend_90d"].to_numpy(),
            "value_tier": scored["value_tier"].to_numpy(),
            "engagement_score": score_feats["engagement_score"].to_numpy(),
            "features_json": [json.dumps(r) for r in score_feats[all_cols].round(4).to_dict("records")],
        }
    )
    db.append(engine, "customers", cust)
    pred = scored.reset_index()[
        [
            "customer_id",
            "churn_prob",
            "churn_flag",
            "risk_band",
            "clv_pred_90d",
            "priority_score",
            "ann_prob",
            "lstm_prob",
            "anomaly_score",
            "is_anomaly",
            "next_category",
        ]
    ]
    pred["customer_id"] = pred["customer_id"].astype("int64")
    pred["model_name"], pred["threshold"], pred["scored_at"] = prod, threshold, pd.Timestamp.now()
    db.append(engine, "predictions", pred)
    segs = seg["assignments"].reset_index()[["customer_id", "segment_id", "segment_label", "gmm_cluster", "pca_x", "pca_y"]]
    db.append(engine, "segments", segs)
    prof = pd.DataFrame(seg["report"]["profiles"]).rename(columns={"label": "label"})
    db.append(engine, "segment_profiles", prof[[c for c in db.SegmentProfile.__table__.columns.keys() if c in prof.columns]])
    db.append(engine, "monthly_revenue", rev[["month", "revenue", "orders", "customers", "is_partial", "is_forecast"]])
    if cfg.database["store_transactions"]:
        tx = clean[
            [
                "invoice",
                "invoice_date",
                "customer_id",
                "stock_code",
                "description",
                "category",
                "quantity",
                "price",
                "line_total",
                "is_return",
                "country",
            ]
        ]
        db.append(engine, "transactions", tx)

    # 12. comparison table, metrics file, report --------------------------------------------
    table = explog.comparison_table("churn")
    table.to_csv(art / "model_comparison.csv", index=False)
    metrics["comparison_table"] = table.to_dict("records")
    metrics["runtime_seconds"] = round(time.time() - t_start, 1)
    _write_json(art / "metrics.json", metrics)
    from src.report import write_report  # local import keeps pipeline import light

    write_report(metrics, cfg, docs / "REPORT.md")
    log.info("Pipeline finished in %.0fs. Production model: %s (threshold %.3f)", metrics["runtime_seconds"], prod, threshold)
    return metrics


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description="Run the full customer-behaviour pipeline")
    ap.add_argument("--config", default=None, help="path to a YAML config (default: config/config.yaml)")
    ap.add_argument("--source", choices=["auto", "file", "ucimlrepo", "synthetic"], default=None)
    args = ap.parse_args()
    overrides = {"data.source": args.source} if args.source else None
    run_pipeline(load_config(args.config, overrides))


if __name__ == "__main__":
    main()
