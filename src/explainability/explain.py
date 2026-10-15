"""Explainability (PRD 10): SHAP (global + local), LIME (local cross-check), partial dependence."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.inspection import partial_dependence

from src.features.build import EXPLAIN_NAMES
from src.utils.logging import get_logger

try:  # LIME is only a cross-check for SHAP, so a missing install must not break training or serving
    from lime.lime_tabular import LimeTabularExplainer
except ImportError:  # pragma: no cover - exercised by test_lime_is_optional
    LimeTabularExplainer = None

log = get_logger(__name__)

ACTIONS = {
    "recency_days": "Send a personalised win-back message; the long silence is the main warning sign.",
    "frequency": "Encourage a repeat purchase with a time-limited offer on their favourite category.",
    "gap_mean": "Their ordering rhythm has slowed; trigger a reminder near their usual re-order window.",
    "gap_std": "Their timing is erratic; use a gentle, recurring reminder rather than a one-off push.",
    "purchase_trend": "Order frequency is falling; offer loyalty perks before they drift away.",
    "monetary_total": "Low overall spend; try bundles or cross-sell to raise basket value.",
    "monetary_avg": "Small baskets; free-shipping thresholds or bundles can lift order value.",
    "return_rate": "High returns suggest poor product fit; review sizing, descriptions or quality.",
    "discount_sensitivity": "Mostly buys on promotion; a targeted discount is likely to work.",
    "engagement_score": "Overall engagement is low; add them to a re-engagement journey.",
}
DEFAULT_ACTION = "Review this customer's recent activity and consider a tailored retention offer."


def _positive_class(values: Any) -> np.ndarray:
    """Normalise SHAP output across model families to an (n, features) array for the positive class."""
    if isinstance(values, list):
        return np.asarray(values[1])
    values = np.asarray(values)
    return values[:, :, 1] if values.ndim == 3 else values


def fmt_value(feature: str, v: float) -> str:
    """Human-friendly value formatting for narratives."""
    if feature in {"recency_days", "tenure_days", "gap_mean", "gap_std"}:
        return f"{v:.0f} days"
    if feature in {"monetary_total", "monetary_avg", "clv_historical"}:
        return f"{v:,.0f}"
    if feature in {"return_rate", "discount_sensitivity", "seasonal_concentration"} or feature.startswith("aff_"):
        return f"{v * 100:.0f}%"
    if feature == "frequency":
        return f"{v:.0f} orders"
    return f"{v:.2f}"


def pretty(feature: str) -> str:
    """Business-readable feature name."""
    if feature in EXPLAIN_NAMES:
        return EXPLAIN_NAMES[feature]
    if feature.startswith("aff_"):
        return f"share of spend in {feature[4:].replace('_', ' ')}"
    if feature.startswith("country_"):
        return f"customer located in {feature[8:]}"
    return feature.replace("_", " ")


class ChurnExplainer:
    """Wraps a fitted tree model with SHAP, plus a LIME explainer built on the training data."""

    def __init__(self, model, columns: list[str], X_train: pd.DataFrame | None = None):
        self.model, self.columns = model, columns
        self.tree = shap.TreeExplainer(model)
        self.lime = None
        if X_train is not None and LimeTabularExplainer is not None:
            self.lime = LimeTabularExplainer(
                X_train[columns].to_numpy(),
                feature_names=columns,
                class_names=["retained", "churned"],
                mode="classification",
                discretize_continuous=True,
                random_state=42,
            )

    def shap_values(self, X: pd.DataFrame) -> pd.DataFrame:
        """SHAP values (log-odds contribution to churn) per customer and feature."""
        vals = _positive_class(self.tree.shap_values(X[self.columns]))
        return pd.DataFrame(vals, index=X.index, columns=self.columns)

    def base_value(self) -> float:
        ev = np.asarray(self.tree.expected_value).ravel()
        return float(ev[-1] if len(ev) > 1 else ev[0])

    def top_drivers(self, row: pd.Series, shap_row: pd.Series, k: int = 5) -> list[dict[str, Any]]:
        """Top-k features by absolute SHAP contribution, with direction and formatted value."""
        order = shap_row.abs().sort_values(ascending=False).index[:k]
        return [
            {
                "feature": f,
                "label": pretty(f),
                "value": float(row[f]),
                "value_text": fmt_value(f, float(row[f])),
                "shap": round(float(shap_row[f]), 4),
                "direction": "increases churn risk" if shap_row[f] > 0 else "reduces churn risk",
            }
            for f in order
        ]

    def lime_drivers(self, row: pd.Series, k: int = 5) -> list[dict[str, Any]]:
        """LIME local explanation as a cross-check on SHAP (slower; used on demand)."""
        if self.lime is None:
            raise RuntimeError("LIME is unavailable. Install it in this environment with: python -m pip install lime")
        exp = self.lime.explain_instance(
            row[self.columns].to_numpy(dtype=float),
            lambda a: self.model.predict_proba(pd.DataFrame(a, columns=self.columns)),
            num_features=k,
            num_samples=1500,
        )
        return [{"rule": rule, "weight": round(float(w), 4)} for rule, w in exp.as_list()]

    def global_importance(self, X: pd.DataFrame, top: int = 15) -> list[dict[str, Any]]:
        """Mean |SHAP| ranking over a (sampled) population."""
        sv = self.shap_values(X)
        imp = sv.abs().mean().sort_values(ascending=False).head(top)
        return [{"feature": f, "label": pretty(f), "importance": round(float(v), 4)} for f, v in imp.items()]

    def save_summary_plot(self, X: pd.DataFrame, path: Path) -> None:
        """Beeswarm summary plot for the report."""
        sv = self.shap_values(X)
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.figure(figsize=(7, 5))
        shap.summary_plot(sv.to_numpy(), X[self.columns], max_display=12, show=False)
        plt.tight_layout()
        plt.savefig(path, dpi=130, bbox_inches="tight")
        plt.close()

    def pdp(self, X: pd.DataFrame, features: list[str], grid: int = 20) -> dict[str, dict[str, list[float]]]:
        """Partial dependence curves (probability of churn) for the given features."""
        out: dict[str, dict[str, list[float]]] = {}
        for f in features:
            res = partial_dependence(self.model, X[self.columns], [f], grid_resolution=grid, kind="average")
            out[f] = {"x": [round(float(v), 3) for v in res["grid_values"][0]], "y": [round(float(v), 4) for v in res["average"][0]]}
        return out


def template_narrative(customer_id: Any, prob: float, band: str, drivers: list[dict[str, Any]]) -> str:
    """Deterministic plain-language explanation used when Ollama is unavailable (and as its grounding)."""
    risk = [d for d in drivers if d["shap"] > 0][:3]
    protect = [d for d in drivers if d["shap"] < 0][:2]
    parts = [f"Customer {customer_id} has a {prob * 100:.0f}% estimated chance of not buying in the next 90 days ({band} risk)."]
    if risk:
        parts.append("Main warning signs: " + "; ".join(f"{d['label']} is {d['value_text']}" for d in risk) + ".")
    if protect:
        parts.append("Helping retention: " + "; ".join(f"{d['label']} is {d['value_text']}" for d in protect) + ".")
    if band == "low":
        parts.append("Suggested action: no urgent intervention; keep regular communication going and protect what is working.")
    else:
        prefix = "Suggested action: " if band == "high" else "Suggested action (monitor): "
        parts.append(prefix + (ACTIONS.get(risk[0]["feature"], DEFAULT_ACTION) if risk else DEFAULT_ACTION))
    return " ".join(parts)
