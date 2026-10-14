"""Deep learning models (PRD 8.2): ANN churn classifier, LSTM on purchase sequences, autoencoder anomalies."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.models.common import ExperimentLog, classification_metrics, pick_threshold
from src.models.numpy_nn import export_keras_dense, numpy_forward
from src.utils.config import Config
from src.utils.logging import get_logger

log = get_logger(__name__)

SKEWED = ["recency_days", "frequency", "monetary_total", "monetary_avg", "clv_historical", "avg_basket_size", "gap_mean", "gap_std", "tenure_days"]
AE_COLS = [
    "recency_days",
    "frequency",
    "monetary_total",
    "monetary_avg",
    "avg_basket_size",
    "return_rate",
    "discount_sensitivity",
    "gap_std",
    "purchase_trend",
]


def dl_transform(X: pd.DataFrame) -> pd.DataFrame:
    """Signed log1p on heavy-tailed columns so the neural nets see roughly symmetric inputs."""
    out = X.copy()
    for c in SKEWED:
        if c in out.columns:
            out[c] = np.sign(out[c]) * np.log1p(np.abs(out[c]))
    return out


def _tf():
    import tensorflow as tf  # imported lazily: only the training pipeline needs TensorFlow

    return tf


def _plot_history(history, title: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    ax.plot(history["loss"], label="train")
    ax.plot(history["val_loss"], label="validation")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _callbacks(patience: int):
    tf = _tf()
    return [tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True)]


def _class_weights(y: pd.Series) -> dict[int, float]:
    n, pos = len(y), float(y.sum())
    return {0: n / (2 * max(n - pos, 1)), 1: n / (2 * max(pos, 1))}


def train_ann(X: pd.DataFrame, y: pd.Series, split, cfg: Config, explog: ExperimentLog, art_dir: Path, fig_dir: Path) -> dict[str, Any]:
    """Feed-forward network: Dense-BN-ReLU-Dropout blocks, class weights, early stopping."""
    tf = _tf()
    tf.keras.utils.set_random_seed(cfg["seed"])
    tr, va, te = split
    dl = cfg.deep_learning
    scaler = StandardScaler().fit(dl_transform(X.loc[tr]))
    Z = {k: scaler.transform(dl_transform(X.loc[idx])) for k, idx in {"tr": tr, "va": va, "te": te}.items()}
    L = tf.keras.layers
    reg = tf.keras.regularizers.l2(1e-4)
    model = tf.keras.Sequential(
        [
            L.Input((Z["tr"].shape[1],)),
            L.Dense(64, kernel_regularizer=reg),
            L.BatchNormalization(),
            L.Activation("relu"),
            L.Dropout(0.3),
            L.Dense(32, kernel_regularizer=reg),
            L.BatchNormalization(),
            L.Activation("relu"),
            L.Dropout(0.2),
            L.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="binary_crossentropy")
    t0 = time.time()
    hist = model.fit(
        Z["tr"],
        y.loc[tr].to_numpy(),
        validation_data=(Z["va"], y.loc[va].to_numpy()),
        epochs=dl["epochs"],
        batch_size=dl["batch_size"],
        class_weight=_class_weights(y.loc[tr]),
        callbacks=_callbacks(dl["patience"]),
        verbose=0,
    )
    p_va, p_te = model.predict(Z["va"], verbose=0).ravel(), model.predict(Z["te"], verbose=0).ravel()
    thr = pick_threshold(y.loc[va], p_va, cfg.split["target_recall"])
    test_m = classification_metrics(y.loc[te], p_te, thr)
    val_auc = classification_metrics(y.loc[va], p_va, thr)["roc_auc"]
    explog.log("churn", "ann", {"epochs_run": len(hist.history["loss"]), "layers": "64-32-1"}, time.time() - t0, None, val_auc, test_m)
    _plot_history(hist.history, "ANN training vs validation loss", fig_dir / "ann_loss.png")

    art_dir.mkdir(parents=True, exist_ok=True)
    model.save(art_dir / "ann.keras")
    export_keras_dense(model, art_dir / "ann_numpy.npz")
    joblib.dump(scaler, art_dir / "ann_scaler.joblib")
    parity = float(np.abs(numpy_forward(art_dir / "ann_numpy.npz", Z["te"]).ravel() - p_te).max())
    log.info(
        "ANN test_auc=%.3f recall=%.3f epochs=%d numpy-parity-maxdiff=%.2e", test_m["roc_auc"], test_m["recall"], len(hist.history["loss"]), parity
    )
    return {
        "metrics": {"val_auc": val_auc, "test": test_m, "epochs_run": len(hist.history["loss"]), "numpy_parity": parity},
        "threshold": thr,
        "val_proba": p_va,
        "test_proba": p_te,
        "scaler": scaler,
    }


def train_lstm(seqs: np.ndarray, y: pd.Series, split, cfg: Config, explog: ExperimentLog, art_dir: Path, fig_dir: Path) -> dict[str, Any]:
    """LSTM over the last N orders (value, gap, recency of order, category) predicting churn."""
    tf = _tf()
    tf.keras.utils.set_random_seed(cfg["seed"])
    tr, va, te = split
    pos = {cid: i for i, cid in enumerate(y.index)}
    ix = {k: np.array([pos[c] for c in idx]) for k, idx in {"tr": tr, "va": va, "te": te}.items()}
    L = tf.keras.layers
    dl = cfg.deep_learning
    model = tf.keras.Sequential(
        [
            L.Input(seqs.shape[1:]),
            L.Masking(mask_value=0.0),
            L.LSTM(32, dropout=0.2),
            L.Dense(16, activation="relu"),
            L.Dropout(0.2),
            L.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="binary_crossentropy")
    t0 = time.time()
    hist = model.fit(
        seqs[ix["tr"]],
        y.loc[tr].to_numpy(),
        validation_data=(seqs[ix["va"]], y.loc[va].to_numpy()),
        epochs=dl["epochs"],
        batch_size=dl["batch_size"],
        class_weight=_class_weights(y.loc[tr]),
        callbacks=_callbacks(dl["patience"]),
        verbose=0,
    )
    p_va, p_te = model.predict(seqs[ix["va"]], verbose=0).ravel(), model.predict(seqs[ix["te"]], verbose=0).ravel()
    thr = pick_threshold(y.loc[va], p_va, cfg.split["target_recall"])
    test_m = classification_metrics(y.loc[te], p_te, thr)
    val_auc = classification_metrics(y.loc[va], p_va, thr)["roc_auc"]
    explog.log(
        "churn", "lstm", {"epochs_run": len(hist.history["loss"]), "units": 32, "seq_len": seqs.shape[1]}, time.time() - t0, None, val_auc, test_m
    )
    _plot_history(hist.history, "LSTM training vs validation loss", fig_dir / "lstm_loss.png")
    art_dir.mkdir(parents=True, exist_ok=True)
    model.save(art_dir / "lstm.keras")
    p_all = model.predict(seqs, verbose=0).ravel()
    log.info("LSTM test_auc=%.3f recall=%.3f epochs=%d", test_m["roc_auc"], test_m["recall"], len(hist.history["loss"]))
    return {
        "metrics": {"val_auc": val_auc, "test": test_m, "epochs_run": len(hist.history["loss"])},
        "threshold": thr,
        "proba_all": pd.Series(p_all, index=y.index),
        "val_proba": p_va,
        "test_proba": p_te,
    }


def train_autoencoder(X: pd.DataFrame, split, cfg: Config, explog: ExperimentLog, art_dir: Path, fig_dir: Path) -> dict[str, Any]:
    """Dense autoencoder on spending features; error above the training percentile flags anomalies."""
    tf = _tf()
    tf.keras.utils.set_random_seed(cfg["seed"])
    tr, va, te = split
    dl = cfg.deep_learning
    scaler = StandardScaler().fit(dl_transform(X.loc[tr, AE_COLS]))
    Z = {k: scaler.transform(dl_transform(X.loc[idx, AE_COLS])) for k, idx in {"tr": tr, "va": va, "te": te}.items()}
    d = Z["tr"].shape[1]
    L = tf.keras.layers
    model = tf.keras.Sequential(
        [L.Input((d,)), L.Dense(16, activation="relu"), L.Dense(4, activation="relu"), L.Dense(16, activation="relu"), L.Dense(d)]
    )
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse")
    t0 = time.time()
    hist = model.fit(
        Z["tr"],
        Z["tr"],
        validation_data=(Z["va"], Z["va"]),
        epochs=dl["epochs"],
        batch_size=dl["batch_size"],
        callbacks=_callbacks(dl["patience"]),
        verbose=0,
    )

    def err(z: np.ndarray) -> np.ndarray:
        return ((model.predict(z, verbose=0) - z) ** 2).mean(axis=1)

    e_tr, e_va, e_te = err(Z["tr"]), err(Z["va"]), err(Z["te"])
    threshold = float(np.percentile(e_tr, dl["autoencoder_percentile"]))
    metrics = {
        "threshold": round(threshold, 4),
        "flag_rate_train": round(float((e_tr > threshold).mean()), 4),
        "flag_rate_val": round(float((e_va > threshold).mean()), 4),
        "flag_rate_test": round(float((e_te > threshold).mean()), 4),
        "val_error_p50": round(float(np.median(e_va)), 4),
        "val_error_p99": round(float(np.percentile(e_va, 99)), 4),
    }
    explog.log(
        "anomaly", "autoencoder", {"epochs_run": len(hist.history["loss"]), "arch": f"{d}-16-4-16-{d}"}, time.time() - t0, test_metrics=metrics
    )
    _plot_history(hist.history, "Autoencoder training vs validation loss", fig_dir / "autoencoder_loss.png")
    art_dir.mkdir(parents=True, exist_ok=True)
    model.save(art_dir / "autoencoder.keras")
    export_keras_dense(model, art_dir / "autoencoder_numpy.npz")
    joblib.dump({"scaler": scaler, "threshold": threshold, "columns": AE_COLS}, art_dir / "autoencoder_meta.joblib")
    log.info(
        "Autoencoder threshold=%.3f flag rates train/val/test=%.3f/%.3f/%.3f",
        threshold,
        metrics["flag_rate_train"],
        metrics["flag_rate_val"],
        metrics["flag_rate_test"],
    )
    return {"metrics": metrics, "scaler": scaler, "threshold": threshold, "model": model}


def autoencoder_scores(X: pd.DataFrame, art_dir: Path) -> pd.Series:
    """Reconstruction error per customer using the NumPy copy of the autoencoder (no TensorFlow needed)."""
    meta = joblib.load(art_dir / "autoencoder_meta.joblib")
    z = meta["scaler"].transform(dl_transform(X[meta["columns"]]))
    recon = numpy_forward(art_dir / "autoencoder_numpy.npz", z)
    return pd.Series(((recon - z) ** 2).mean(axis=1), index=X.index)


def ann_proba(X: pd.DataFrame, art_dir: Path) -> pd.Series:
    """ANN churn probability using the NumPy copy of the network."""
    scaler = joblib.load(art_dir / "ann_scaler.joblib")
    z = scaler.transform(dl_transform(X[scaler.feature_names_in_]))
    return pd.Series(numpy_forward(art_dir / "ann_numpy.npz", z).ravel(), index=X.index)
