"""Classical churn classifiers (PRD 8.1): LR, DT, RF, XGBoost, LightGBM, KNN."""

from __future__ import annotations

import time
import warnings
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import loguniform, randint, uniform
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from src.models.common import ExperimentLog, classification_metrics, pick_threshold
from src.utils.config import Config
from src.utils.logging import get_logger

log = get_logger(__name__)
warnings.filterwarnings("ignore", message=".*eval_set.*")

TREE_MODELS = {"random_forest", "xgboost", "lightgbm"}


def _specs(cfg: Config, pos_weight: float) -> dict[str, dict[str, Any]]:
    seed = cfg["seed"]
    return {
        "logistic_regression": {
            "est": Pipeline([("scale", StandardScaler()), ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed))]),
            "grid": {"clf__C": [0.01, 0.1, 1.0, 10.0]},
            "search": "grid",
        },
        "decision_tree": {
            "est": DecisionTreeClassifier(class_weight="balanced", random_state=seed),
            "grid": {"max_depth": [3, 4, 5, 6, 8], "min_samples_leaf": [10, 20, 40]},
            "search": "grid",
        },
        "random_forest": {
            "est": RandomForestClassifier(class_weight="balanced_subsample", random_state=seed, n_jobs=-1),
            "grid": {"n_estimators": randint(150, 400), "max_depth": randint(4, 14), "min_samples_leaf": randint(2, 20)},
            "search": "random",
        },
        "xgboost": {
            "est": xgb.XGBClassifier(
                n_estimators=300, scale_pos_weight=pos_weight, eval_metric="auc", random_state=seed, n_jobs=1, tree_method="hist"
            ),
            "grid": {
                "learning_rate": loguniform(0.02, 0.2),
                "max_depth": randint(2, 7),
                "subsample": uniform(0.6, 0.4),
                "colsample_bytree": uniform(0.5, 0.5),
            },
            "search": "random",
        },
        "lightgbm": {
            "est": lgb.LGBMClassifier(n_estimators=300, class_weight="balanced", random_state=seed, n_jobs=1, verbose=-1),
            "grid": {
                "learning_rate": loguniform(0.02, 0.2),
                "num_leaves": randint(6, 40),
                "subsample": uniform(0.6, 0.4),
                "colsample_bytree": uniform(0.5, 0.5),
                "min_child_samples": randint(5, 40),
            },
            "search": "random",
        },
        "knn": {
            "est": Pipeline([("scale", StandardScaler()), ("clf", KNeighborsClassifier())]),
            "grid": {"clf__n_neighbors": [5, 11, 21, 41], "clf__weights": ["uniform", "distance"]},
            "search": "grid",
        },
    }


def _refit_with_early_stopping(name: str, best_params: dict, cfg: Config, pos_weight: float, X_tr, y_tr, X_va, y_va):
    """Refit boosted models with a generous tree budget and early stopping on the validation fold."""
    seed = cfg["seed"]
    patience = 30
    if name == "xgboost":
        m = xgb.XGBClassifier(
            **{**best_params, "n_estimators": 1000},
            scale_pos_weight=pos_weight,
            eval_metric="auc",
            early_stopping_rounds=patience,
            random_state=seed,
            n_jobs=1,
            tree_method="hist",
        )
        m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        return m, {"best_iteration": int(m.best_iteration)}
    m = lgb.LGBMClassifier(**{**best_params, "n_estimators": 1000}, class_weight="balanced", random_state=seed, n_jobs=1, verbose=-1)
    m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], eval_metric="auc", callbacks=[lgb.early_stopping(patience, verbose=False)])
    return m, {"best_iteration": int(m.best_iteration_)}


def train_classical_models(
    X: pd.DataFrame, y: pd.Series, split: tuple[pd.Index, pd.Index, pd.Index], cfg: Config, explog: ExperimentLog
) -> dict[str, Any]:
    """Tune with 5-fold stratified CV on the training set only, then evaluate on validation and test once.

    Returns {"models", "val_proba", "test_proba", "thresholds", "metrics"}.
    """
    tr, va, te = split
    X_tr, y_tr, X_va, y_va, X_te, y_te = X.loc[tr], y.loc[tr], X.loc[va], y.loc[va], X.loc[te], y.loc[te]
    pos_weight = float((y_tr == 0).sum() / max((y_tr == 1).sum(), 1))
    cv = StratifiedKFold(cfg.models["cv_folds"], shuffle=True, random_state=cfg["seed"])
    out: dict[str, Any] = {"models": {}, "val_proba": {}, "test_proba": {}, "thresholds": {}, "metrics": {}}

    for name, spec in _specs(cfg, pos_weight).items():
        t0 = time.time()
        if spec["search"] == "grid":
            search = GridSearchCV(spec["est"], spec["grid"], scoring="roc_auc", cv=cv, n_jobs=1)
        else:
            search = RandomizedSearchCV(
                spec["est"], spec["grid"], n_iter=cfg.models["search_iter"], scoring="roc_auc", cv=cv, n_jobs=1, random_state=cfg["seed"]
            )
        search.fit(X_tr, y_tr)
        model, params, extra = search.best_estimator_, dict(search.best_params_), {}
        cv_auc = float(search.best_score_)
        if name in {"xgboost", "lightgbm"}:
            model, extra = _refit_with_early_stopping(name, params, cfg, pos_weight, X_tr, y_tr, X_va, y_va)
        p_va = model.predict_proba(X_va)[:, 1]
        thr = pick_threshold(y_va, p_va, cfg.split["target_recall"])
        p_te = model.predict_proba(X_te)[:, 1]
        test_m = {**classification_metrics(y_te, p_te, thr), "recall_at_0.5": classification_metrics(y_te, p_te, 0.5)["recall"]}
        val_auc = classification_metrics(y_va, p_va, thr)["roc_auc"]
        explog.log("churn", name, {**params, **extra}, time.time() - t0, cv_auc, val_auc, test_m)
        out["models"][name], out["val_proba"][name], out["test_proba"][name] = model, p_va, p_te
        out["thresholds"][name] = thr
        out["metrics"][name] = {"cv_auc": round(cv_auc, 4), "val_auc": val_auc, "test": test_m, "params": {**params, **extra}}
        log.info("%-20s cv_auc=%.3f val_auc=%.3f test_auc=%.3f recall=%.3f", name, cv_auc, val_auc, test_m["roc_auc"], test_m["recall"])
    return out


def choose_production_model(results: dict[str, Any]) -> str:
    """Production model = best validation ROC-AUC among tree-based models (SHAP-compatible).

    Selection uses the validation set so the test set stays untouched by model choice.
    """
    candidates = {k: v["val_auc"] for k, v in results["metrics"].items() if k in TREE_MODELS}
    return max(candidates, key=candidates.get)


def save_models(results: dict[str, Any], artifacts_dir) -> None:
    """Persist every fitted model (including its scaler where the model is a Pipeline)."""
    d = artifacts_dir / "classical"
    d.mkdir(parents=True, exist_ok=True)
    for name, model in results["models"].items():
        joblib.dump(model, d / f"{name}.joblib")


def cv_score_unfitted(est, X, y, cfg: Config) -> float:
    """5-fold stratified CV ROC-AUC for an unfitted estimator (used by tests and ad-hoc checks)."""
    cv = StratifiedKFold(cfg.models["cv_folds"], shuffle=True, random_state=cfg["seed"])
    return float(np.mean(cross_val_score(est, X, y, cv=cv, scoring="roc_auc")))
