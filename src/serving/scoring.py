"""Shared scoring bundle: loads training artifacts and scores customers (used by the pipeline AND the API)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.data.cleaning import clean_transactions, customer_lines, encode_country
from src.data.loader import standardize
from src.explainability.explain import ChurnExplainer, template_narrative
from src.features.build import build_features, engagement_score
from src.models.deep import ann_proba, autoencoder_scores
from src.segmentation.cluster import assign_segments
from src.utils.config import Config
from src.utils.logging import get_logger

log = get_logger(__name__)


def risk_band(p: float, threshold: float, high_cut: float) -> str:
    """low below the model threshold, high at/above max(high_cut, threshold), medium in between."""
    return "high" if p >= max(high_cut, threshold) else ("medium" if p >= threshold else "low")


class ModelBundle:
    """All fitted artifacts needed to turn customer features into predictions and explanations."""

    def __init__(self, cfg: Config, art_dir: Path | None = None, load_lime: bool = True):
        self.cfg = cfg
        self.art = Path(art_dir or cfg.path("artifacts_dir"))
        self.meta: dict[str, Any] = json.loads((self.art / "meta.json").read_text())
        self.model_cols: list[str] = self.meta["model_columns"]
        self.all_cols: list[str] = self.meta["all_columns"]
        self.levels: list[str] = self.meta["country_levels"]
        self.threshold: float = self.meta["threshold"]
        self.reference = joblib.load(self.art / "reference.joblib")
        self.model = joblib.load(self.art / "classical" / f"{self.meta['production_model']}.joblib")
        self.clv_model = joblib.load(self.art / "clv_model.joblib")
        nc = self.art / "next_category.joblib"
        self.next_cat = joblib.load(nc) if nc.exists() else None
        bg = None
        if load_lime and (self.art / "lime_background.csv.gz").exists():
            bg = pd.read_csv(self.art / "lime_background.csv.gz")
        self.explainer = ChurnExplainer(self.model, self.model_cols, bg)

    # ---- feature preparation -------------------------------------------------
    def features_from_records(self, records: list[dict[str, Any]]) -> pd.DataFrame:
        """Complete partial customer records: derive missing fields, fill training defaults, encode country."""
        df = pd.DataFrame(records)
        d = self.meta["feature_defaults"]
        if "monetary_avg" not in df and {"monetary_total", "frequency"} <= set(df):
            df["monetary_avg"] = df["monetary_total"] / df["frequency"].clip(lower=1)
        if "clv_historical" not in df and "monetary_total" in df:
            df["clv_historical"] = df["monetary_total"] * (1 - df.get("return_rate", 0.0))
        country = df["country"] if "country" in df else pd.Series("United Kingdom", index=df.index)
        df = df.drop(columns=["country"], errors="ignore")
        for c in self.all_cols:
            if c.startswith("country_"):
                continue
            if c not in df.columns:
                df[c] = d[c]
            df[c] = df[c].fillna(d[c])
        df["engagement_score"] = engagement_score(df, self.reference)
        df = df.join(encode_country(country, self.levels))
        df["country"] = country
        return df

    def features_from_transactions(self, raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
        """Run the SAME cleaning and feature code as training on raw transactions; cutoff = day after the last row."""
        std = standardize(raw)
        clean, report = clean_transactions(std, self.cfg)
        lines = customer_lines(clean)
        cutoff = std["invoice_date"].max().normalize() + pd.Timedelta(days=1)
        feats, _ = build_features(lines, cutoff, self.cfg, self.levels, reference=self.reference, with_labels=False)
        return feats, report

    # ---- scoring --------------------------------------------------------------
    def score(self, feats: pd.DataFrame) -> pd.DataFrame:
        """Predictions for a complete feature table: churn, CLV, segment, anomaly, ANN check, next category."""
        X = feats[self.model_cols]
        p = pd.Series(self.model.predict_proba(X)[:, 1], index=feats.index)
        risk = self.cfg.risk
        out = pd.DataFrame(index=feats.index)
        out["churn_prob"] = p.round(4)
        out["churn_flag"] = p >= self.threshold
        out["risk_band"] = [risk_band(v, self.threshold, risk["high_cut"]) for v in p]
        out["clv_pred_90d"] = np.expm1(self.clv_model.predict(X)).clip(0).round(2)
        spend90 = (feats["clv_historical"] / feats["tenure_days"].clip(lower=90) * 90).clip(lower=0)
        out["avg_spend_90d"] = spend90.round(2)
        out["priority_score"] = (p * spend90).round(2)
        cuts = self.meta["value_cuts"]
        out["value_tier"] = np.where(spend90 >= cuts["high"], "high", np.where(spend90 >= cuts["mid"], "mid", "low"))
        out["ann_prob"] = ann_proba(feats, self.art).round(4)
        ae = autoencoder_scores(feats, self.art)
        out["anomaly_score"] = ae.round(4)
        out["is_anomaly"] = ae > self.meta["anomaly_threshold"]
        if self.next_cat is not None:
            out["next_category"] = self.next_cat.predict(X)
        else:
            out["next_category"] = None
        seg = assign_segments(feats, self.art)
        return out.join(seg)

    def explain(self, feats: pd.DataFrame, customer_id: Any, scored: pd.Series, k: int = 5) -> dict[str, Any]:
        """SHAP drivers plus a deterministic plain-language narrative for one customer."""
        row = feats.loc[customer_id]
        sv = self.explainer.shap_values(feats.loc[[customer_id]]).iloc[0]
        drivers = self.explainer.top_drivers(row, sv, k)
        narrative = template_narrative(customer_id, float(scored["churn_prob"]), str(scored["risk_band"]), drivers)
        return {"drivers": drivers, "base_value": round(self.explainer.base_value(), 4), "narrative": narrative}
