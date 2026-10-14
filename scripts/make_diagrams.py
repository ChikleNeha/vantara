"""Draw the architecture, ER and workflow diagrams into docs/figures (matplotlib only, no Graphviz needed)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parents[1] / "docs" / "figures"
INK, PAPER, ACCENT, WARM, GREEN = "#1b212b", "#fffdf9", "#2f5d8a", "#c2432b", "#2e7d6a"


def box(ax, x, y, w, h, text, fc=PAPER, ec=INK, fs=9, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12", fc=fc, ec=ec, lw=1.3))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=INK, weight="bold" if bold else "normal", wrap=True)


def arrow(ax, p, q, label=None, color=INK, rad=0.0):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=12, lw=1.3, color=color, connectionstyle=f"arc3,rad={rad}"))
    if label:
        ax.text((p[0] + q[0]) / 2, (p[1] + q[1]) / 2 + 0.12, label, ha="center", fontsize=7.5, color=color)


def canvas(w, h, title):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.axis("off")
    ax.set_title(title, fontsize=13, weight="bold", loc="left", color=INK)
    return fig, ax


def architecture():
    fig, ax = canvas(12, 6.2, "System architecture")
    box(ax, 0.3, 4.5, 2.3, 1.0, "Online Retail II\n(xlsx / UCI / synthetic)", fc="#f3efe6", fs=8.5)
    box(ax, 3.1, 4.5, 3.2, 1.0, "Pipeline\npython -m src.pipeline\nclean · features · train · score", fc="#dde8f2", bold=True, fs=8.5)
    box(ax, 6.9, 4.5, 2.3, 1.0, "models_artifacts/\nmodels, scalers,\nmetrics", fc="#f3efe6", fs=8.5)
    box(ax, 9.7, 4.5, 2.1, 1.0, "SQLite\ncustomers · predictions\nsegments · revenue", fc="#f3efe6", fs=8.5)
    arrow(ax, (2.6, 5.0), (3.1, 5.0))
    arrow(ax, (6.3, 5.0), (6.9, 5.0))
    # scored rows: elbow path under the artifacts box
    ax.plot([4.7, 4.7, 10.75], [4.5, 4.15, 4.15], color=INK, lw=1.3)
    arrow(ax, (10.75, 4.15), (10.75, 4.5))
    ax.text(5.5, 4.22, "scored rows", fontsize=7.5, ha="center", va="bottom", color=INK)
    box(
        ax,
        4.6,
        2.2,
        4.4,
        1.3,
        "FastAPI service\n/api/predict · /predict/batch · /customers\n/segments · /revenue · /explain · /assistant",
        fc="#dde8f2",
        bold=True,
        fs=8,
    )
    arrow(ax, (8.0, 4.5), (7.4, 3.5))
    ax.text(8.05, 3.85, "load at startup", fontsize=7.5, color=INK)
    arrow(ax, (10.75, 4.15), (9.0, 3.2), "read-only queries", rad=-0.1)
    box(
        ax,
        0.4,
        2.2,
        3.5,
        1.3,
        "React dashboard (Vite + Tailwind)\nOverview · Segments · Leaderboard\nExplorer · Batch · Insights · Ask",
        fc="#f9e3dd",
        bold=True,
        fs=8,
    )
    arrow(ax, (3.9, 3.0), (4.6, 3.0))
    arrow(ax, (4.6, 2.75), (3.9, 2.75), "JSON", rad=0.0)
    box(ax, 9.3, 0.6, 2.4, 1.1, "Ollama (local LLM)\nnarratives + Q&A only", fc="#e0f0eb", fs=8.5)
    arrow(ax, (7.8, 2.2), (9.3, 1.3), "facts → plain English", color=GREEN)
    ax.text(
        0.6,
        0.75,
        "Predictions always come from the trained models.\nIf Ollama is down, deterministic templates are used.",
        fontsize=8.5,
        color=INK,
        va="center",
    )
    fig.savefig(OUT / "architecture.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def er():
    fig, ax = canvas(12.8, 7.8, "Database schema (SQLite)")
    tables = {
        "customers": (
            0.4,
            3.3,
            [
                "PK customer_id",
                "country",
                "first_purchase, last_purchase",
                "frequency, monetary_total",
                "clv_historical, avg_spend_90d",
                "value_tier, engagement_score",
                "features_json",
            ],
        ),
        "predictions": (
            4.6,
            4.2,
            [
                "PK/FK customer_id",
                "churn_prob, churn_flag",
                "risk_band, priority_score",
                "clv_pred_90d",
                "ann_prob, lstm_prob",
                "anomaly_score, is_anomaly",
                "next_category, model_name",
                "threshold, scored_at",
            ],
        ),
        "segments": (8.8, 4.4, ["PK/FK customer_id", "segment_id, segment_label", "gmm_cluster", "pca_x, pca_y"]),
        "segment_profiles": (8.8, 1.9, ["PK segment_id", "label, description", "size, share", "mean recency, frequency, spend ..."]),
        "transactions": (
            4.6,
            0.5,
            [
                "PK id",
                "invoice, invoice_date",
                "customer_id (indexed)",
                "stock_code, description, category",
                "quantity, price, line_total",
                "is_return, country",
            ],
        ),
        "monthly_revenue": (0.4, 0.4, ["PK month", "revenue, orders, customers", "is_partial, is_forecast"]),
    }
    pos = {}
    for name, (x, y, cols) in tables.items():
        h = 0.34 * len(cols) + 0.5
        ax.add_patch(FancyBboxPatch((x, y), 3.3, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=PAPER, ec=INK, lw=1.3))
        ax.add_patch(FancyBboxPatch((x, y + h - 0.42), 3.3, 0.42, boxstyle="round,pad=0.02,rounding_size=0.08", fc=ACCENT, ec=INK, lw=1.3))
        ax.text(x + 0.15, y + h - 0.21, name, color="white", weight="bold", fontsize=10, va="center")
        for i, c in enumerate(cols):
            ax.text(x + 0.15, y + h - 0.7 - i * 0.34, c, fontsize=8.2, va="center", color=INK)
        pos[name] = (x, y, 3.3, h)
    c, p, s, sp, t = (pos[k] for k in ("customers", "predictions", "segments", "segment_profiles", "transactions"))
    arrow(ax, (c[0] + c[2], c[1] + c[3] - 0.9), (p[0], p[1] + p[3] - 0.9), "1 : 1")
    arrow(ax, (c[0] + c[2] * 0.7, c[1]), (t[0], t[1] + t[3] - 0.9), "1 : many", rad=0.25)
    arrow(ax, (p[0] + p[2], p[1] + p[3] - 0.9), (s[0], s[1] + s[3] - 0.9), "1 : 1", rad=-0.05)
    arrow(ax, (s[0] + s[2] / 2, s[1]), (sp[0] + sp[2] / 2, sp[1] + sp[3]), "many : 1 (segment_id)")
    fig.savefig(OUT / "er_diagram.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def workflow():
    fig, ax = canvas(13, 4.6, "Modelling workflow (strict point-in-time discipline)")
    steps = [
        ("1 Load &\nvalidate", "#f3efe6"),
        ("2 Clean\n(returns, dupes,\nadmin codes)", "#f3efe6"),
        ("3 Features\nas of cutoff T\n(history only)", "#dde8f2"),
        ("4 Labels from\nT .. T+90d\n(churn, CLV)", "#f9e3dd"),
        ("5 70/15/15\nstratified split\n+ VIF pruning", "#dde8f2"),
        ("6 Train 6 classical\n+ ANN + LSTM\n5-fold CV (train)", "#dde8f2"),
        ("7 Pick model on\nvalidation; set\nthreshold (recall ≥ .70)", "#e0f0eb"),
        ("8 Test set\nscored once", "#f9e3dd"),
    ]
    w, gap = 1.35, 0.22
    for i, (t, fc) in enumerate(steps):
        x = 0.2 + i * (w + gap)
        box(ax, x, 2.3, w, 1.3, t, fc=fc, fs=7.6)
        if i:
            arrow(ax, (x - gap, 2.95), (x, 2.95))
    box(
        ax,
        0.2,
        0.5,
        4.6,
        1.0,
        "Also from step 3:\nsegmentation (K-Means vs GMM) · autoencoder anomalies\nhurdle CLV · next-category model",
        fc="#f3efe6",
        fs=8.5,
    )
    box(ax, 5.2, 0.5, 4.2, 1.0, "9 SHAP (global + local) · LIME cross-check\npartial dependence · Ollama narrative", fc="#e0f0eb", fs=8.5)
    box(ax, 9.8, 0.5, 3.0, 1.0, "10 Persist: SQLite + artifacts\nauto-generated REPORT.md", fc="#dde8f2", fs=8.5)
    fig.savefig(OUT / "workflow.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    architecture()
    er()
    workflow()
    print("wrote", sorted(p.name for p in OUT.glob("*.png")))
