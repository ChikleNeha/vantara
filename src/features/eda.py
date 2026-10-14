"""Exploratory analysis and multicollinearity control (PRD 6.4): correlations, VIF, country and seasonal views."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.utils.logging import get_logger

log = get_logger(__name__)


def vif_table(X: pd.DataFrame) -> pd.Series:
    """Variance inflation factors from the inverse correlation matrix (ridge-stabilised for exact collinearity)."""
    Xs = X.loc[:, X.std() > 0]
    corr = np.corrcoef(Xs.to_numpy(dtype=float), rowvar=False)
    inv = np.linalg.inv(corr + 1e-6 * np.eye(len(corr)))
    return pd.Series(np.diag(inv), index=Xs.columns).sort_values(ascending=False)


def select_by_vif(
    X: pd.DataFrame, threshold: float = 10.0, protected: tuple[str, ...] = ("recency_days", "frequency")
) -> tuple[list[str], list[dict[str, Any]]]:
    """Iteratively drop the feature with the largest VIF until all are at or below ``threshold``.

    Constant columns are dropped first. ``protected`` features are never removed. Returns (kept, drop_log).
    """
    kept = [c for c in X.columns if X[c].std() > 0]
    dropped: list[dict[str, Any]] = [{"feature": c, "reason": "constant"} for c in X.columns if c not in kept]
    while len(kept) > 2:
        v = vif_table(X[kept])
        candidates = v[~v.index.isin(protected)]
        if candidates.empty or candidates.iloc[0] <= threshold:
            break
        worst = candidates.index[0]
        dropped.append({"feature": worst, "reason": f"VIF {candidates.iloc[0]:.1f} > {threshold}"})
        kept.remove(worst)
    return kept, dropped


def run_eda(
    feats: pd.DataFrame, clean: pd.DataFrame, model_cols: list[str], out_dir: Path, doc_path: Path, corr_cols: list[str] | None = None
) -> dict[str, Any]:
    """Write EDA figures and a markdown report ending with testable hypotheses."""
    out_dir.mkdir(parents=True, exist_ok=True)
    # Correlations cover every engineered feature (including ones VIF pruning later removes) so the hypotheses can cite them.
    corr_set = corr_cols or model_cols
    corr_churn = feats[corr_set].corrwith(feats["churn"]).dropna().sort_values(key=np.abs, ascending=False)
    corr_clv = feats[corr_set].corrwith(np.log1p(feats["clv_target"])).dropna().sort_values(key=np.abs, ascending=False)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.2))
    for ax, col, title in zip(
        axes, ["recency_days", "frequency", "monetary_total"], ["Recency (days)", "Frequency (orders)", "Total spend (log)"], strict=True
    ):
        vals = np.log1p(feats[col]) if col == "monetary_total" else feats[col]
        ax.hist(vals, bins=30, color="#4C6EF5")
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_dir / "eda_rfm_distributions.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 3.2))
    feats.groupby(pd.cut(feats["recency_days"], [0, 30, 60, 120, 240, 480, 2000]), observed=True)["churn"].mean().plot.bar(ax=ax, color="#F03E3E")
    ax.set_ylabel("churn rate")
    ax.set_xlabel("recency bucket (days)")
    fig.tight_layout()
    fig.savefig(out_dir / "eda_churn_by_recency.png", dpi=120)
    plt.close(fig)

    monthly = clean[~clean["is_return"]].groupby(clean["invoice_date"].dt.month)["line_total"].sum()
    fig, ax = plt.subplots(figsize=(7, 3.2))
    monthly.plot.bar(ax=ax, color="#12B886")
    ax.set_xlabel("calendar month")
    ax.set_ylabel("revenue")
    fig.tight_layout()
    fig.savefig(out_dir / "eda_seasonality.png", dpi=120)
    plt.close(fig)

    by_country = clean[~clean["is_return"]].groupby("country")["line_total"].sum().sort_values(ascending=False)
    uk_share = float(by_country.iloc[0] / by_country.sum())
    q4_share = float(monthly.reindex([10, 11, 12]).fillna(0).sum() / monthly.sum())
    vif = vif_table(feats[model_cols])

    top_pos = corr_churn.head(3)
    hyps = [
        f"H1. Recency is the dominant churn signal (corr with churn {corr_churn.get('recency_days', float('nan')):+.2f}); models should rank it first.",
        f"H2. Order frequency and spend protect against churn (corr {corr_churn.get('frequency', float('nan')):+.2f} and {corr_churn.get('monetary_total', float('nan')):+.2f}); loyal high-spend customers need little intervention.",
        f"H3. A declining monthly order trend (purchase_trend corr {corr_churn.get('purchase_trend', float('nan')):+.2f}) adds signal beyond RFM.",
        f"H4. Q4 carries {q4_share:.0%} of revenue, so seasonal customers will look 'churned' outside the season; seasonal_concentration should help separate them.",
        f"H5. The top market holds {uk_share:.0%} of revenue; country features will matter little for the main market but may flag niche overseas buyers.",
        "H6. Return-heavy customers form a separate behavioural group; the autoencoder should surface them as anomalies.",
    ]
    md = [
        "# Exploratory Data Analysis",
        "",
        f"Customers analysed: **{len(feats):,}**; churn rate in the 90-day window: **{feats['churn'].mean():.1%}**.",
        "",
        "![RFM](figures/eda_rfm_distributions.png)",
        "![Churn by recency](figures/eda_churn_by_recency.png)",
        "![Seasonality](figures/eda_seasonality.png)",
        "",
        "## Strongest correlations with churn",
        *[f"- `{k}`: {v:+.3f}" for k, v in top_pos.items()],
        "",
        "## Strongest correlations with log next-90-day spend",
        *[f"- `{k}`: {v:+.3f}" for k, v in corr_clv.head(3).items()],
        "",
        f"## Multicollinearity (VIF on the final model features, max {vif.max():.1f})",
        *[f"- `{k}`: {v:.1f}" for k, v in vif.head(5).items()],
        "",
        "## Hypotheses for the modelling stage",
        *[f"- {h}" for h in hyps],
        "",
    ]
    doc_path.parent.mkdir(parents=True, exist_ok=True)
    doc_path.write_text("\n".join(md), encoding="utf-8")
    log.info("EDA report written to %s", doc_path)
    return {
        "corr_churn": {k: round(float(v), 4) for k, v in corr_churn.head(10).items()},
        "corr_clv": {k: round(float(v), 4) for k, v in corr_clv.head(10).items()},
        "vif": {k: round(float(v), 2) for k, v in vif.items()},
        "top_market_share": round(uk_share, 4),
        "q4_revenue_share": round(q4_share, 4),
        "hypotheses": hyps,
    }
