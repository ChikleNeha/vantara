"""Customer segmentation (PRD 9): K-Means (primary) and Gaussian Mixture (comparison)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from src.models.deep import dl_transform
from src.utils.config import Config
from src.utils.logging import get_logger

log = get_logger(__name__)

SEG_COLS = [
    "recency_days",
    "frequency",
    "monetary_total",
    "avg_basket_size",
    "return_rate",
    "discount_sensitivity",
    "seasonal_concentration",
    "purchase_trend",
]


def _prep(X: pd.DataFrame, scaler: StandardScaler | None = None) -> tuple[np.ndarray, StandardScaler]:
    z = dl_transform(X[SEG_COLS])
    if scaler is None:
        scaler = StandardScaler().fit(z)
    return scaler.transform(z), scaler


def _quality(Z: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    return {
        "silhouette": round(float(silhouette_score(Z, labels)), 4),
        "davies_bouldin": round(float(davies_bouldin_score(Z, labels)), 4),
        "calinski_harabasz": round(float(calinski_harabasz_score(Z, labels)), 1),
    }


def name_segments(profile: pd.DataFrame, pop: pd.DataFrame) -> dict[int, tuple[str, str]]:
    """Rule-based business names from each cluster's position relative to the whole population.

    ``profile`` holds cluster means; ``pop`` holds population means. Names are deduplicated.
    """
    ranks = {c: profile[c].rank(pct=True) for c in SEG_COLS}
    names: dict[int, tuple[str, str]] = {}
    used: dict[str, int] = {}
    for seg in profile.index:
        rec, freq, mon = ranks["recency_days"][seg], ranks["frequency"][seg], ranks["monetary_total"][seg]
        ret = profile.loc[seg, "return_rate"] > max(0.12, 2 * pop["return_rate"])
        seasonal = profile.loc[seg, "seasonal_concentration"] > max(0.55, 1.5 * pop["seasonal_concentration"])
        disc = profile.loc[seg, "discount_sensitivity"] > 1.4 * pop["discount_sensitivity"]
        if ret:
            label, desc = "Return-heavy buyers", "High share of purchases sent back; investigate product fit and fulfilment."
        elif seasonal:
            label, desc = "Seasonal gift shoppers", "Orders concentrated in Oct to Dec; target with pre-season campaigns."
        elif rec >= 0.6 and mon >= 0.5:
            label, desc = "Lapsed high-value", "Strong past spend but gone quiet; best win-back opportunity."
        elif rec <= 0.4 and freq >= 0.5 and mon >= 0.5:
            label, desc = "Loyal champions", "Recent, frequent, high spend; protect and reward."
        elif disc and mon < 0.6:
            label, desc = "Discount seekers", "Buy mostly on promotion; sensitive to price offers."
        elif rec <= 0.4 and freq < 0.5:
            label, desc = "New and promising", "Recent buyers with few orders; nurture toward a second purchase."
        elif rec >= 0.6:
            label, desc = "Dormant low-value", "Inactive and low spend; low-cost reactivation only."
        else:
            label, desc = "Steady mid-value", "Regular but modest spend; upsell and cross-sell."
        used[label] = used.get(label, 0) + 1
        names[seg] = (label if used[label] == 1 else f"{label} ({used[label]})", desc)
    return names


def fit_segments(X: pd.DataFrame, cfg: Config, art_dir: Path) -> dict[str, Any]:
    """Pick K by silhouette over [k_min, k_max], fit K-Means + GMM, build profiles and PCA coordinates."""
    Z, scaler = _prep(X)
    seg = cfg.segmentation
    sweep = []
    for k in range(seg["k_min"], seg["k_max"] + 1):
        km = KMeans(k, n_init=10, random_state=cfg["seed"]).fit(Z)
        sweep.append({"k": k, "inertia": round(float(km.inertia_), 2), "silhouette": round(float(silhouette_score(Z, km.labels_)), 4)})
    best_k = max(sweep, key=lambda r: r["silhouette"])["k"]
    km = KMeans(best_k, n_init=10, random_state=cfg["seed"]).fit(Z)
    gmm = GaussianMixture(best_k, covariance_type="full", n_init=3, random_state=cfg["seed"]).fit(Z)
    gmm_labels = gmm.predict(Z)

    pca = PCA(2, random_state=cfg["seed"]).fit(Z)
    xy = pca.transform(Z)
    labels = km.labels_
    raw_cols = SEG_COLS + ["clv_historical", "engagement_score"]
    profile = X[raw_cols].groupby(labels).mean()
    names = name_segments(profile, X[SEG_COLS].mean())
    prof_rows = []
    for s in profile.index:
        row = {
            "segment_id": int(s),
            "label": names[s][0],
            "description": names[s][1],
            "size": int((labels == s).sum()),
            "share": round(float((labels == s).mean()), 4),
        }
        row.update({c: round(float(profile.loc[s, c]), 3) for c in raw_cols})
        prof_rows.append(row)

    art_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({"scaler": scaler, "kmeans": km, "pca": pca, "names": {int(k): v for k, v in names.items()}}, art_dir / "segmentation.joblib")
    assign = pd.DataFrame(
        {
            "segment_id": labels,
            "segment_label": [names[s][0] for s in labels],
            "gmm_cluster": gmm_labels,
            "pca_x": xy[:, 0].round(4),
            "pca_y": xy[:, 1].round(4),
        },
        index=X.index,
    )
    result = {
        "best_k": best_k,
        "sweep": sweep,
        "quality": {"kmeans": _quality(Z, labels), "gmm": _quality(Z, gmm_labels)},
        "profiles": prof_rows,
        "explained_variance": [round(float(v), 4) for v in pca.explained_variance_ratio_],
    }
    log.info(
        "Segmentation: k=%d silhouette=%.3f (GMM %.3f) segments=%s",
        best_k,
        result["quality"]["kmeans"]["silhouette"],
        result["quality"]["gmm"]["silhouette"],
        [r["label"] for r in prof_rows],
    )
    return {"assignments": assign, "report": result}


def assign_segments(X: pd.DataFrame, art_dir: Path) -> pd.DataFrame:
    """Assign new customers to the fitted K-Means segments (used by the live scoring API)."""
    bundle = joblib.load(art_dir / "segmentation.joblib")
    Z, _ = _prep(X, bundle["scaler"])
    labels = bundle["kmeans"].predict(Z)
    xy = bundle["pca"].transform(Z)
    return pd.DataFrame(
        {"segment_id": labels, "segment_label": [bundle["names"][int(s)][0] for s in labels], "pca_x": xy[:, 0].round(4), "pca_y": xy[:, 1].round(4)},
        index=X.index,
    )
