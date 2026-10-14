"""Feature engineering, including the point-in-time leakage tests required by the PRD (section 7)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.build import build_features, build_sequences, engagement_score, feature_columns, fit_reference
from src.features.eda import select_by_vif, vif_table
from src.utils.config import load_config
from tests.conftest import tx

CFG = load_config()
LEVELS = ["United Kingdom"]
CUTOFF = pd.Timestamp("2021-01-01")


@pytest.fixture()
def lines():
    return tx(
        [
            (1, "i1", "2020-06-01", 2, 10.0, "kitchen", False),
            (1, "i2", "2020-09-01", 1, 30.0, "lighting", False),
            (1, "C3", "2020-09-10", -1, 30.0, "lighting", True),
            (1, "i4", "2020-12-01", 3, 10.0, "kitchen", False),
            (2, "i5", "2020-03-01", 1, 100.0, "garden", False),
            # post-cutoff (labels only)
            (1, "i6", "2021-02-01", 4, 5.0, "kitchen", False),
            (3, "i7", "2021-01-15", 1, 1.0, "garden", False),
        ]
    )


def test_rfm_values_are_exact(lines):
    f, _ = build_features(lines, CUTOFF, CFG, LEVELS)
    c1 = f.loc[1]
    assert c1["frequency"] == 3 and c1["recency_days"] == 31 and c1["tenure_days"] == pytest.approx(214, abs=1)
    assert c1["monetary_total"] == pytest.approx(20 + 30 + 30)
    assert c1["clv_historical"] == pytest.approx(80 - 30)  # return netted out
    assert c1["return_rate"] == pytest.approx(1 / 6)
    assert 3 not in f.index  # customer first seen after the cutoff is not in the cohort


def test_labels_use_only_the_horizon_window(lines):
    f, _ = build_features(lines, CUTOFF, CFG, LEVELS)
    assert f.loc[1, "churn"] == 0 and f.loc[2, "churn"] == 1
    assert f.loc[1, "clv_target"] == pytest.approx(20.0)
    assert f.loc[1, "next_category"] == "kitchen"


def test_no_future_leakage_into_features(lines):
    """Features must be identical whether or not post-cutoff rows exist, or what they contain."""
    full, ref = build_features(lines, CUTOFF, CFG, LEVELS)
    past_only, _ = build_features(lines[lines["invoice_date"] < CUTOFF], CUTOFF, CFG, LEVELS, reference=ref, with_labels=False)
    cols = feature_columns(LEVELS)
    pd.testing.assert_frame_equal(full[cols], past_only[cols])
    tampered = lines.copy()
    tampered.loc[tampered["invoice_date"] >= CUTOFF, ["quantity", "price"]] = [999, 999]
    tampered["line_total"] = tampered["quantity"] * tampered["price"]
    t_feats, _ = build_features(tampered, CUTOFF, CFG, LEVELS, reference=ref)
    pd.testing.assert_frame_equal(full[cols], t_feats[cols])


def test_cutoff_boundary_is_strict(lines):
    at = lines.copy()
    at.loc[len(at)] = [2, "edge", CUTOFF, 1, 1.0, "garden", False, 1.0, "A1", "United Kingdom", "ITEM"]
    f, _ = build_features(at, CUTOFF, CFG, LEVELS)
    assert f.loc[2, "frequency"] == 1  # a purchase exactly at the cutoff belongs to the label window
    assert f.loc[2, "churn"] == 0


def test_no_purchases_before_cutoff_raises(lines):
    with pytest.raises(ValueError):
        build_features(lines, pd.Timestamp("2019-01-01"), CFG, LEVELS)


def test_trend_sign_and_affinity_sums_to_one():
    rows, k = [], 0
    for m in range(1, 13):  # customer 1 buys more and more often
        for _ in range(m // 3 + 1):
            k += 1
            rows.append((1, f"a{k}", f"2020-{m:02d}-10", 1, 10.0, "kitchen", False))
    for m in (1, 2, 3):  # customer 2 fades out
        for _ in range(4 - m):
            k += 1
            rows.append((2, f"b{k}", f"2020-{m:02d}-12", 1, 10.0, "garden", False))
    f, _ = build_features(tx(rows), CUTOFF, CFG, LEVELS, with_labels=False)
    assert f.loc[1, "purchase_trend"] > 0 > f.loc[2, "purchase_trend"]
    aff = f.filter(like="aff_").sum(axis=1)
    assert np.allclose(aff, 1.0)


def test_engagement_score_orders_customers(lines):
    f, ref = build_features(lines, CUTOFF, CFG, LEVELS)
    s = engagement_score(f, ref)
    assert s.between(0, 100).all() and s.loc[1] > s.loc[2]
    assert set(fit_reference(f)) == {"recency_days", "frequency", "monetary_total"}


def test_sequences_are_left_padded_and_time_ordered(lines):
    seq = build_sequences(lines, CUTOFF, pd.Index([1, 2]), max_len=5)
    assert seq.shape == (2, 5, 12)
    assert (seq[0, :2] == 0).all() and (seq[0, 2:] != 0).any()  # customer 1: 3 orders -> 2 padded steps
    assert seq[0, 2, 2] > seq[0, 4, 2]  # older orders are further from the cutoff


def test_vif_drops_collinear_features_but_protects_core():
    rng = np.random.default_rng(0)
    a = rng.normal(size=500)
    X = pd.DataFrame(
        {
            "recency_days": a,
            "frequency": rng.normal(size=500),
            "dup": a * 2 + rng.normal(scale=0.01, size=500),
            "const": 1.0,
            "ok": rng.normal(size=500),
        }
    )
    kept, dropped = select_by_vif(X, 10.0)
    assert "recency_days" in kept and "dup" not in kept and "const" not in kept
    assert {d["feature"] for d in dropped} == {"dup", "const"}
    assert vif_table(X[["frequency", "ok"]]).max() < 2
