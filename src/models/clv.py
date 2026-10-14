"""Customer lifetime value regression and next-purchase-category model."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.models.common import ExperimentLog, regression_metrics
from src.utils.config import Config
from src.utils.logging import get_logger

log = get_logger(__name__)


class HurdleCLV:
    """Two-stage expected-spend model: P(any positive spend) x E[spend | spend > 0].

    ``predict`` returns log1p(expected spend) so every CLV model in the bundle shares one interface.
    """

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.stage1 = xgb.XGBClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=3, subsample=0.8, colsample_bytree=0.8, random_state=seed, n_jobs=1, tree_method="hist"
        )
        self.stage2 = xgb.XGBRegressor(
            n_estimators=800,
            learning_rate=0.03,
            max_depth=3,
            subsample=0.8,
            colsample_bytree=0.8,
            early_stopping_rounds=40,
            random_state=seed,
            n_jobs=1,
            tree_method="hist",
        )
        self.smear = 1.0

    def fit(self, X_tr: pd.DataFrame, y_tr: pd.Series, X_va: pd.DataFrame, y_va: pd.Series) -> "HurdleCLV":
        """Fit both stages; stage 2 uses buyers only, with early stopping on validation buyers."""
        self.stage1.fit(X_tr, (y_tr > 0).astype(int))
        b_tr, b_va = (y_tr > 0).to_numpy(), (y_va > 0).to_numpy()
        self.stage2.fit(X_tr[b_tr], np.log1p(y_tr[b_tr]), eval_set=[(X_va[b_va], np.log1p(y_va[b_va]))], verbose=False)
        resid = np.log1p(y_tr[b_tr]) - self.stage2.predict(X_tr[b_tr])
        self.smear = float(np.mean(np.exp(resid)))  # Duan smearing: unbiased mean after a log transform
        return self

    def expected_spend(self, X: pd.DataFrame) -> np.ndarray:
        """Expected next-90-day net spend in original currency units."""
        p_buy = self.stage1.predict_proba(X)[:, 1]
        cond = np.clip(np.exp(self.stage2.predict(X)) * self.smear - 1, 0, None)
        return p_buy * cond

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """log1p of the expected spend (interface shared with the other CLV models)."""
        return np.log1p(self.expected_spend(X))


def train_clv(X: pd.DataFrame, y: pd.Series, split, cfg: Config, explog: ExperimentLog) -> dict[str, Any]:
    """Predict next-90-day net spend. Candidates: ridge baseline, direct XGBoost on log1p(spend), and the hurdle model.

    The winner is chosen by validation R-squared on the original scale; test metrics are only reported.
    """
    tr, va, te = split
    y_log = np.log1p(y)
    seed = cfg["seed"]
    results: dict[str, Any] = {}

    def record(name: str, model, p_va: np.ndarray, p_te: np.ndarray, t0: float, params: dict) -> None:
        raw = regression_metrics(y.loc[te], p_te)
        entry = {"model": model, "raw": raw, "val_r2": regression_metrics(y.loc[va], p_va)["r2"]}
        buyers = (y.loc[te] > 0).to_numpy()
        if buyers.sum() > 5:
            entry["buyers_r2"] = regression_metrics(y.loc[te][buyers], p_te[buyers])["r2"]
        results[name] = entry
        explog.log("clv", name, params, time.time() - t0, test_metrics=raw)

    t0 = time.time()
    ridge = Pipeline([("s", StandardScaler()), ("m", Ridge(alpha=5.0))]).fit(X.loc[tr], y_log.loc[tr])
    record("ridge", ridge, np.expm1(ridge.predict(X.loc[va])).clip(0), np.expm1(ridge.predict(X.loc[te])).clip(0), t0, {"alpha": 5.0})

    t0 = time.time()
    reg = xgb.XGBRegressor(
        n_estimators=800,
        learning_rate=0.03,
        max_depth=3,
        subsample=0.8,
        colsample_bytree=0.8,
        early_stopping_rounds=40,
        random_state=seed,
        n_jobs=1,
        tree_method="hist",
    )
    reg.fit(X.loc[tr], y_log.loc[tr], eval_set=[(X.loc[va], y_log.loc[va])], verbose=False)
    record(
        "xgboost_direct",
        reg,
        np.expm1(reg.predict(X.loc[va])).clip(0),
        np.expm1(reg.predict(X.loc[te])).clip(0),
        t0,
        {"best_iteration": int(reg.best_iteration)},
    )

    t0 = time.time()
    hurdle = HurdleCLV(seed).fit(X.loc[tr], y.loc[tr], X.loc[va], y.loc[va])
    record(
        "hurdle",
        hurdle,
        hurdle.expected_spend(X.loc[va]),
        hurdle.expected_spend(X.loc[te]),
        t0,
        {"stage2_iterations": int(hurdle.stage2.best_iteration)},
    )

    results["best"] = max((k for k in results), key=lambda k: results[k]["val_r2"])
    for k, v in results.items():
        if k != "best":
            log.info("CLV %-15s val R2=%.3f test R2=%.3f MAE=%.1f", k, v["val_r2"], v["raw"]["r2"], v["raw"]["mae"])
    return results


def train_next_category(X: pd.DataFrame, target: pd.Series, split, cfg: Config, explog: ExperimentLog) -> dict[str, Any] | None:
    """Multi-class model for the category of a customer's next order (only customers who bought in the window)."""
    tr, va, te = split
    mask = target.notna()
    tr_i, te_i = tr[mask.loc[tr].to_numpy()], te[mask.loc[te].to_numpy()]
    if len(tr_i) < 50 or len(te_i) < 10:
        log.warning("Not enough labelled customers for next-category model; skipping")
        return None
    t0 = time.time()
    clf = RandomForestClassifier(n_estimators=250, min_samples_leaf=3, class_weight="balanced_subsample", random_state=cfg["seed"], n_jobs=-1)
    clf.fit(X.loc[tr_i], target.loc[tr_i])
    proba = clf.predict_proba(X.loc[te_i])
    classes = clf.classes_
    top1 = float((classes[proba.argmax(1)] == target.loc[te_i].to_numpy()).mean())
    top3_idx = np.argsort(-proba, axis=1)[:, :3]
    top3 = float(np.mean([target.loc[te_i].iloc[i] in classes[top3_idx[i]] for i in range(len(te_i))]))
    majority = float(target.loc[tr_i].value_counts(normalize=True).iloc[0])
    metrics = {"top1_accuracy": round(top1, 4), "top3_accuracy": round(top3, 4), "majority_baseline": round(majority, 4), "n_test": int(len(te_i))}
    explog.log("next_category", "random_forest", {"n_estimators": 250}, time.time() - t0, test_metrics=metrics)
    log.info("Next-category top1=%.3f top3=%.3f (majority baseline %.3f)", top1, top3, majority)
    return {"model": clf, "metrics": metrics}
