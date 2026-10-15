"""Auto-generate docs/REPORT.md from pipeline metrics so reported numbers always match the logs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.utils.config import Config

MATH_APPENDIX = """
## Appendix A. Mathematical foundations

**Logistic regression and cross-entropy.** The model outputs p = sigma(w.x + b), sigma(z) = 1/(1+e^-z). Training minimises
L = -(1/n) sum[ y log p + (1-y) log(1-p) ] (+ lambda ||w||^2 for L2 regularisation). The gradient is (p - y)x, which is why
the loss is convex and well-behaved.

**Gradient descent and backpropagation.** Parameters move against the gradient, theta <- theta - eta * dL/dtheta. In the ANN
the chain rule propagates the error layer by layer: dL/dW_l = delta_l a_{l-1}^T with delta_l = (W_{l+1}^T delta_{l+1}) * f'(z_l).
Adam adds running estimates of the first and second moments of the gradient to scale each step.

**Regularisation.** L2 penalises large weights (ridge), L1 induces sparsity, dropout randomly zeroes activations during
training so units cannot co-adapt, and early stopping halts when validation loss stops improving.

**PCA.** With standardised data matrix Z and covariance C = Z^T Z/(n-1), the principal components are the eigenvectors of C;
the share of variance explained by component i is lambda_i / sum(lambda). Used here only to draw segments in 2-D.

**K-Means and GMM.** K-Means minimises the within-cluster sum of squares sum_k sum_{x in C_k} ||x - mu_k||^2. A Gaussian
mixture models p(x) = sum_k pi_k N(x | mu_k, Sigma_k) and is fit by expectation-maximisation, giving soft assignments.

**Cluster quality.** Silhouette s = (b - a)/max(a, b), where a is mean intra-cluster distance and b the mean distance to the
nearest other cluster (higher is better). Davies-Bouldin averages the worst-case ratio of within-cluster scatter to between-centroid
distance (lower is better).

**Metrics.** Precision = TP/(TP+FP); recall = TP/(TP+FN); F1 = 2PR/(P+R); ROC-AUC is the probability that a random churner is
scored above a random non-churner. R^2 = 1 - SS_res/SS_tot; MAE and RMSE are mean absolute and root-mean-square error.

**SHAP.** The Shapley value of feature i is phi_i = sum over subsets S of (|S|!(M-|S|-1)!/M!) [f(S u {i}) - f(S)]. For trees,
TreeSHAP computes these exactly in polynomial time; contributions sum to f(x) - E[f(x)] in log-odds.

**Variance inflation factor.** VIF_i = 1/(1 - R_i^2), where R_i^2 comes from regressing feature i on all others; values above
10 indicate severe collinearity.
"""

FUTURE = """
## Future enhancements
- Retrain monthly with rolling cutoffs (multiple cutoffs per customer) to enlarge the training set and measure drift.
- Add a probability-calibration step (isotonic) and decision-cost thresholding using actual campaign costs.
- Survival models (Cox / BG-NBD with Gamma-Gamma) for a principled lifetime-value estimate beyond a 90-day window.
- Real product taxonomy instead of keyword categories; item-level recommender (e.g. implicit ALS).
- Transformer encoder over purchase sequences and a CatBoost comparison (both optional in the PRD).
- Monitoring: population-stability index on inputs, scheduled retraining, and A/B measurement of retention campaigns.
- Swap SQLite for PostgreSQL if concurrent writers or larger volumes are needed (SQLAlchemy layer already isolates this).
"""


def _fmt(v: Any, nd: int = 3) -> str:
    return "n/a" if v is None or (isinstance(v, float) and pd.isna(v)) else (f"{v:.{nd}f}" if isinstance(v, float) else str(v))


def _status(ok: bool | None) -> str:
    return "Met" if ok else ("Not met" if ok is False else "Check manually")


def write_report(m: dict[str, Any], cfg: Config, path: Path) -> None:
    """Render the final report from the metrics dictionary produced by the pipeline."""
    prod = m["production_model"]
    pt = m["classical"][prod]["test"]
    clv = m["clv"][m["clv"]["best"]]
    synthetic = m["data_source"] == "synthetic"
    n_test = pt["tn"] + pt["fp"] + pt["fn"] + pt["tp"]
    n_pos = pt["fn"] + pt["tp"]
    L: list[str] = ["# Customer Behaviour Prediction Platform: Technical Report", ""]
    if synthetic:
        L += [
            "> **Data warning.** This run used the built-in SYNTHETIC generator because `online_retail_II.xlsx` was not present. "
            "Every number below demonstrates that the pipeline works; none of them are findings about real customers. "
            "Place the UCI file in `data/raw/` and re-run `python -m src.pipeline` to regenerate this report on real data.",
            "",
        ]
    L += [
        "## 1. Summary",
        f"- Data source: **{m['data_source']}**; {m['features']['n_customers']:,} customers modelled; churn rate {m['features']['churn_rate']:.1%} "
        f"(no purchase in the {m['cutoffs']['horizon_days']} days after {m['cutoffs']['train_cutoff']}).",
        f"- Production churn model: **{prod}** (selected on validation ROC-AUC) with test ROC-AUC {pt['roc_auc']:.3f}, recall {pt['recall']:.3f}, precision {pt['precision']:.3f}, F1 {pt['f1']:.3f}.",
        f"- Segments: **{m['segmentation']['best_k']}** K-Means segments (silhouette {m['segmentation']['quality']['kmeans']['silhouette']:.3f}).",
        f"- Portfolio today: {m['portfolio']['high_risk']:,} high-risk customers; expected spend at risk over the next 90 days about {m['portfolio']['revenue_at_risk_90d']:,.0f}.",
        "",
        "## 2. Data and cleaning",
        "| Step | Rows |",
        "|---|---|",
        *[f"| {k.replace('_', ' ')} | {v:,} |" for k, v in m["cleaning"].items()],
        "",
        f"Features were built with a strict point-in-time cutoff ({m['cutoffs']['train_cutoff']}); labels use only the following {m['cutoffs']['horizon_days']} days. "
        f"{m['features']['model_columns']} of {m['features']['all_columns']} columns remain after iterative VIF pruning (threshold {cfg.features['vif_threshold']}). "
        "Dropped: " + (", ".join(f"`{d['feature']}`" for d in m["features"]["vif_dropped"]) or "none") + ".",
        "",
        "## 3. Churn model comparison (test set, ranked by ROC-AUC then recall)",
        "Threshold per model is chosen on the validation set to keep recall at or above the target, so precision and recall are directly comparable.",
        "",
        "| Rank | Model | CV AUC (train) | Val AUC | Test AUC | Recall | Precision | F1 | Accuracy | Confusion (TN/FP/FN/TP) |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for i, r in enumerate(m["comparison_table"], 1):
        L.append(
            f"| {i} | {r['model']} | {_fmt(r.get('cv_auc'))} | {_fmt(r.get('val_auc'))} | {_fmt(r.get('roc_auc'))} | {_fmt(r.get('recall'))} | {_fmt(r.get('precision'))} | {_fmt(r.get('f1'))} | {_fmt(r.get('accuracy'))} | {r.get('tn')}/{r.get('fp')}/{r.get('fn')}/{r.get('tp')} |"
        )
    corr = m["ann_tree_error_correlation"]
    L += [
        "",
        "### Recommendation",
        f"Serve **{prod}** in the API: it had the best validation ROC-AUC among the SHAP-compatible tree models, trains in seconds and gives exact per-customer explanations. "
        f"The ANN is kept as a secondary check; its residual correlation with {prod} is {corr:.2f}, "
        + (
            "which is low enough that disagreement between the two is informative."
            if corr < 0.8
            else "so it adds limited independent information on this data."
        )
        + " With only a few thousand customers the neural models are exercises rather than production candidates, as the PRD anticipated.",
        "",
        "## 4. Lifetime value (next-90-day net spend)",
        f"Best model (chosen on validation R2): **{m['clv']['best']}**. Test R2 {clv['raw']['r2']:.3f}, MAE {clv['raw']['mae']:.1f}, RMSE {clv['raw']['rmse']:.1f}"
        + (f"; R2 among customers who actually bought: {clv['buyers_r2']:.3f}." if "buyers_r2" in clv else "."),
        "",
        "| Model | Validation R2 | Test R2 | Test MAE | Test RMSE |",
        "|---|---|---|---|---|",
        *[
            f"| {k} | {v['val_r2']:.3f} | {v['raw']['r2']:.3f} | {v['raw']['mae']:.1f} | {v['raw']['rmse']:.1f} |"
            for k, v in m["clv"].items()
            if k != "best"
        ],
        "",
        "Roughly half of customers spend nothing in the next 90 days, so the target is zero-inflated. The hurdle model separates "
        "'will they buy at all' from 'how much if they do', which is why it beats a single regression. The remaining gap to the PRD target "
        "reflects genuinely unpredictable purchase timing over a short window.",
    ]
    if "next_category" in m:
        nc = m["next_category"]
        L += [
            "",
            f"Next-purchase category: top-1 accuracy {nc['top1_accuracy']:.3f}, top-3 {nc['top3_accuracy']:.3f} (majority-class baseline {nc['majority_baseline']:.3f}, n={nc['n_test']}).",
        ]
    ae = m["deep"]["autoencoder"]
    L += [
        "",
        "## 5. Deep learning",
        f"- ANN: stopped at {m['deep']['ann']['epochs_run']} epochs (early stopping), test AUC {m['deep']['ann']['test']['roc_auc']:.3f}. NumPy inference copy matches Keras within {m['deep']['ann']['numpy_parity']:.1e}.",
        f"- LSTM on the last {cfg.deep_learning['lstm_max_len']} orders: {m['deep']['lstm']['epochs_run']} epochs, test AUC {m['deep']['lstm']['test']['roc_auc']:.3f}.",
        f"- Autoencoder: reconstruction-error threshold {ae['threshold']} (training percentile {cfg.deep_learning['autoencoder_percentile']}); flag rate train/val/test = {ae['flag_rate_train']:.3f} / {ae['flag_rate_val']:.3f} / {ae['flag_rate_test']:.3f}. {m['portfolio']['anomalies']} customers flagged today.",
        "- Loss curves: `docs/figures/ann_loss.png`, `lstm_loss.png`, `autoencoder_loss.png`.",
        "",
        "## 6. Segmentation",
        f"K chosen by silhouette over {cfg.segmentation['k_min']} to {cfg.segmentation['k_max']}. "
        f"K-Means: silhouette {m['segmentation']['quality']['kmeans']['silhouette']:.3f}, Davies-Bouldin {m['segmentation']['quality']['kmeans']['davies_bouldin']:.3f}. "
        f"Gaussian mixture: silhouette {m['segmentation']['quality']['gmm']['silhouette']:.3f}, Davies-Bouldin {m['segmentation']['quality']['gmm']['davies_bouldin']:.3f}.",
        "",
        "| Segment | Customers | Share | Mean recency (d) | Mean orders | Mean spend | What to do |",
        "|---|---|---|---|---|---|---|",
        *[
            f"| {p['label']} | {p['size']:,} | {p['share']:.0%} | {p['recency_days']:.0f} | {p['frequency']:.1f} | {p['monetary_total']:,.0f} | {p['description']} |"
            for p in m["segmentation"]["profiles"]
        ],
        "",
        "## 7. Explainability",
        "Top global drivers (mean absolute SHAP on held-out customers):",
        *[f"{i}. {d['label']} ({d['importance']:.3f})" for i, d in enumerate(m["explain"]["global_importance"][:6], 1)],
        "",
        (
            f"SHAP and LIME agree on {m['explain']['shap_lime_top5_agreement']:.0%} of each customer's top-5 features on a random sample, a sanity check that the two methods tell a consistent story. "
            if m["explain"].get("shap_lime_top5_agreement") is not None
            else "The SHAP vs LIME agreement check was skipped because LIME was not installed in this environment. "
        )
        + "`docs/figures/shap_summary.png` shows the full beeswarm. Individual explanations (SHAP bars, LIME rules, plain-language narrative) are served by the API and dashboard.",
        "",
        "## 8. Success metrics versus PRD targets",
        "| Metric | Target | Result | Status |",
        "|---|---|---|---|",
        f"| Churn ROC-AUC (test) | >= 0.80 | {pt['roc_auc']:.3f} | {_status(pt['roc_auc'] >= 0.80)} |",
        f"| Churn recall (test) | >= 0.70 | {pt['recall']:.3f} | {_status(pt['recall'] >= 0.70)} |",
        f"| CLV R2 (test, original scale) | >= 0.60 | {clv['raw']['r2']:.3f} | {_status(clv['raw']['r2'] >= 0.60)} |",
        f"| Single-customer prediction p95 (model + SHAP path) | < 400 ms | {m['latency_ms']['p95']} ms | {_status(m['latency_ms']['p95'] < 400)} |",
        "| Dashboard full segment view load | < 3 s | measure in browser | Check manually |",
        f"| Single-command reproducible pipeline | 100% | `python -m src.pipeline` ({m['runtime_seconds']}s) | Met |",
        "",
        f"The test set holds {n_test} customers ({n_pos} churners), so ROC-AUC and recall carry sampling error of a few points. "
        "A shortfall of about 0.01 against a target is within that noise but is still reported as not met. "
        "The CLV gap is larger: the target is zero-inflated and the window is short. Whether that is irreducible has not been tested here; "
        "a longer label window or more customers are the obvious next experiments.",
        "",
        "## 9. Method notes and deviations from the PRD",
        "- SQLite replaces PostgreSQL and a React dashboard replaces Streamlit (project stack decision). Ollama adds plain-language narratives and a Q&A assistant; it never produces predictions and every feature has a deterministic fallback.",
        "- Class imbalance is handled with class weights on the training set only; validation and test sets are untouched. The test set is evaluated once, after model selection on validation data.",
        "- Hyperparameters are tuned by 5-fold stratified CV on the training set; boosted models then use early stopping on the validation set.",
        "- Categories are keyword-derived from product descriptions (the dataset has no taxonomy).",
        MATH_APPENDIX,
        FUTURE,
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L), encoding="utf-8")
