"""Shared modelling helpers: stratified split, metrics, threshold choice, experiment log."""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    precision_recall_curve,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from src.utils.config import Config


def stratified_split(y: pd.Series, cfg: Config) -> tuple[pd.Index, pd.Index, pd.Index]:
    """70/15/15 (configurable) stratified split on the churn label with a fixed seed."""
    s = cfg.split
    idx = y.index
    train_idx, rest_idx = train_test_split(idx, train_size=s["train"], stratify=y, random_state=cfg["seed"])
    val_share = s["val"] / (s["val"] + s["test"])
    val_idx, test_idx = train_test_split(rest_idx, train_size=val_share, stratify=y.loc[rest_idx], random_state=cfg["seed"])
    return train_idx, val_idx, test_idx


def classification_metrics(y_true, proba, threshold: float = 0.5) -> dict[str, Any]:
    """Accuracy, precision, recall, F1, ROC-AUC and confusion matrix at ``threshold``."""
    y_true = np.asarray(y_true)
    pred = (np.asarray(proba) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "threshold": round(float(threshold), 4),
        "accuracy": round(float(accuracy_score(y_true, pred)), 4),
        "precision": round(float(precision_score(y_true, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, proba)), 4),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def regression_metrics(y_true, pred) -> dict[str, float]:
    """MAE, RMSE and R-squared."""
    y_true, pred = np.asarray(y_true), np.asarray(pred)
    return {
        "mae": round(float(mean_absolute_error(y_true, pred)), 4),
        "rmse": round(float(np.sqrt(np.mean((y_true - pred) ** 2))), 4),
        "r2": round(float(r2_score(y_true, pred)), 4),
    }


def pick_threshold(y_val, p_val, target_recall: float) -> float:
    """Highest-precision threshold on validation data that still meets ``target_recall``.

    Falls back to the F1-maximising threshold when the recall target is unreachable.
    """
    prec, rec, thr = precision_recall_curve(y_val, p_val)
    prec, rec = prec[:-1], rec[:-1]
    ok = rec >= target_recall
    if ok.any():
        best = np.argmax(np.where(ok, prec, -1))
        return float(thr[best])
    f1 = 2 * prec * rec / np.clip(prec + rec, 1e-9, None)
    return float(thr[int(np.argmax(f1))])


class ExperimentLog:
    """Append-only CSV log of every model run so comparison tables are regenerated, not retyped."""

    FIELDS = ["run_id", "timestamp", "task", "model", "params", "cv_auc", "val_auc", "test_metrics", "train_seconds"]

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def reset(self) -> None:
        """Start a fresh log (called once at the start of a pipeline run)."""
        with open(self.path, "w", newline="", encoding="utf-8") as fh:
            csv.DictWriter(fh, self.FIELDS).writeheader()

    def log(
        self,
        task: str,
        model: str,
        params: dict,
        train_seconds: float,
        cv_auc: float | None = None,
        val_auc: float | None = None,
        test_metrics: dict | None = None,
    ) -> None:
        """Write one run."""
        if not self.path.exists():
            self.reset()
        row = {
            "run_id": f"{task}-{model}-{int(time.time() * 1000) % 10**9}",
            "timestamp": pd.Timestamp.now().isoformat(timespec="seconds"),
            "task": task,
            "model": model,
            "params": json.dumps(params, default=str),
            "cv_auc": cv_auc,
            "val_auc": val_auc,
            "test_metrics": json.dumps(test_metrics or {}),
            "train_seconds": round(train_seconds, 2),
        }
        with open(self.path, "a", newline="", encoding="utf-8") as fh:
            csv.DictWriter(fh, self.FIELDS).writerow(row)

    def comparison_table(self, task: str = "churn") -> pd.DataFrame:
        """Rebuild the consolidated comparison table from logged runs, ranked by test ROC-AUC then recall."""
        df = pd.read_csv(self.path)
        df = df[df["task"] == task].drop_duplicates("model", keep="last")
        rows = []
        for _, r in df.iterrows():
            m = json.loads(r["test_metrics"])
            rows.append({"model": r["model"], "cv_auc": r["cv_auc"], "val_auc": r["val_auc"], **m, "train_seconds": r["train_seconds"]})
        out = pd.DataFrame(rows)
        if out.empty:
            return out
        return out.sort_values(["roc_auc", "recall"], ascending=False).reset_index(drop=True)
