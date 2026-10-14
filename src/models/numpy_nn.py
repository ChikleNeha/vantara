"""Tiny NumPy inference engine for the dense Keras models (ANN, autoencoder).

Lets the API serve deep-learning predictions without installing TensorFlow.
Supported layers: Dense (any of linear/relu/sigmoid), BatchNormalization, Activation, Dropout.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

_ACTS = {"linear", "relu", "sigmoid"}


def export_keras_dense(model, path: Path) -> None:
    """Save a Sequential Keras model's layers to ``path`` (.npz) as an ordered list of ops."""
    ops: list[str] = []
    arrays: dict[str, np.ndarray] = {}

    def add_act(name: str) -> None:
        if name not in _ACTS:
            raise ValueError(f"Unsupported activation for NumPy export: {name}")
        if name != "linear":
            ops.append(f"{len(ops)}:{name}")

    for layer in model.layers:
        cls = layer.__class__.__name__
        if cls == "Dense":
            w, b = layer.get_weights()
            k = len(ops)
            arrays[f"w{k}"], arrays[f"b{k}"] = w, b
            ops.append(f"{k}:dense")
            add_act(layer.activation.__name__)
        elif cls == "BatchNormalization":
            g, beta, mean, var = layer.get_weights()
            k = len(ops)
            arrays.update({f"g{k}": g, f"beta{k}": beta, f"m{k}": mean, f"v{k}": var, f"eps{k}": np.array(layer.epsilon)})
            ops.append(f"{k}:bn")
        elif cls == "Activation":
            add_act(layer.activation.__name__)
        elif cls in ("Dropout", "InputLayer"):
            continue
        else:
            raise ValueError(f"Unsupported layer for NumPy export: {cls}")
    np.savez(path, ops=np.array(ops), **arrays)


def numpy_forward(path: Path, x: np.ndarray) -> np.ndarray:
    """Run inference for a model exported by ``export_keras_dense``."""
    data = np.load(path, allow_pickle=False)
    h = np.asarray(x, dtype="float64")
    for op in data["ops"]:
        k, kind = str(op).split(":")
        if kind == "dense":
            h = h @ data[f"w{k}"] + data[f"b{k}"]
        elif kind == "bn":
            h = data[f"g{k}"] * (h - data[f"m{k}"]) / np.sqrt(data[f"v{k}"] + float(data[f"eps{k}"])) + data[f"beta{k}"]
        elif kind == "relu":
            h = np.maximum(h, 0.0)
        elif kind == "sigmoid":
            h = 1.0 / (1.0 + np.exp(-h))
        else:
            raise ValueError(f"Unsupported op {kind}")
    return h
