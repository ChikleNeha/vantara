"""Cleaning, categories, validation and loader."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data import cleaning as C
from src.data.categories import assign_category, categorize
from src.data.loader import load_raw, standardize
from src.data.synthetic import generate_transactions
from src.data.validation import DataValidationError, validate_raw
from src.utils.config import load_config

CFG = load_config(overrides={"data.source": "synthetic", "data.synthetic.n_customers": 300})


def test_category_whole_word_matching():
    assert assign_category("VINTAGE HOME DECOR SIGN") == "home_decor"  # not 'TAG' inside VINTAGE
    assert assign_category("WHITE MUGS SET") == "kitchen"
    assert assign_category(None) == "other"
    assert categorize(pd.Series(["RED LANTERN", None])).tolist() == ["lighting", "other"]


def test_flag_returns_and_dedup():
    df = pd.DataFrame(
        {
            "invoice": ["1", "C2", "3", "3"],
            "stock_code": ["A"] * 4,
            "quantity": [1, -1, 2, 2],
            "invoice_date": pd.to_datetime(["2020-01-01"] * 4),
            "price": [1.0] * 4,
            "customer_id": [1, 1, 2, 2],
        }
    )
    flagged = C.flag_returns(df)
    assert flagged["is_return"].tolist() == [False, True, False, False]
    assert len(C.drop_exact_duplicates(flagged)) == 3


def test_description_fill_uses_stock_code_mode():
    df = pd.DataFrame({"stock_code": ["A", "A", "A", "B"], "description": [" red  mug", "RED MUG", None, None]})
    out = C.standardize_descriptions(df)
    assert out["description"].tolist() == ["RED MUG", "RED MUG", "RED MUG", "UNKNOWN ITEM"]


def test_price_and_outlier_rules():
    df = pd.DataFrame({"price": [1.0, 0.0, -2.0, 2.0, 3.0], "quantity": [5, 5, 5, 50000, 6], "is_return": [False] * 5})
    out, stats = C.handle_prices_and_outliers(df, CFG)
    assert stats["invalid_price_rows"] == 2 and stats["data_entry_errors"] == 1
    assert len(out) == 2


def test_admin_codes_removed():
    df = pd.DataFrame({"stock_code": ["POST", "85123A", "DOT", "M", "20001"]})
    prod, admin = C.split_admin_codes(df, CFG)
    assert prod["stock_code"].tolist() == ["85123A", "20001"] and len(admin) == 3


def test_country_encoding_is_fixed_vocabulary():
    enc = C.encode_country(pd.Series(["France", "Peru", "United Kingdom"]), ["United Kingdom", "France"])
    assert list(enc.columns) == ["country_United Kingdom", "country_France", "country_Other"]
    assert enc.loc[1, "country_Other"] == 1 and enc.sum(axis=1).eq(1).all()


def test_synthetic_has_known_quality_problems():
    raw = generate_transactions(300, seed=1)
    assert raw["Customer ID"].isna().any() and raw["Invoice"].str.startswith("C").any() and (raw["Price"] <= 0).any()
    assert raw.duplicated().any()


def test_full_cleaning_chain():
    raw, src = load_raw(CFG)
    assert src == "synthetic"
    clean, rep = C.clean_transactions(raw, CFG)
    assert (clean["price"] > 0).all() and rep["clean_rows"] == len(clean) < rep["raw_rows"]
    assert not clean["stock_code"].isin(["POST"]).any()
    assert len(C.customer_lines(clean)) < len(clean)  # guests excluded from modelling


def test_loader_falls_back_when_file_missing():
    cfg = load_config(overrides={"data.source": "auto", "paths.raw_dir": "/nonexistent", "data.synthetic.n_customers": 200})
    df, src = load_raw(cfg)
    assert src == "synthetic" and not df.empty


def test_loader_file_only_raises_when_missing():
    cfg = load_config(overrides={"data.source": "file", "paths.raw_dir": "/nonexistent"})
    with pytest.raises(RuntimeError):
        load_raw(cfg)


def test_standardize_renames_and_sorts():
    raw = pd.DataFrame(
        {
            "Invoice": [2, 1],
            "StockCode": ["b", "a"],
            "Description": ["x", "y"],
            "Quantity": [1, 1],
            "InvoiceDate": ["2020-02-01", "2020-01-01"],
            "Price": [1, 1],
            "Customer ID": [1.0, 2.0],
            "Country": [" UK", "UK "],
        }
    )
    out = standardize(raw)
    assert out["invoice"].tolist() == ["1", "2"] and out["country"].tolist() == ["UK", "UK"] and out["stock_code"].tolist() == ["A", "B"]


def test_validation_passes_and_fails():
    raw, _ = load_raw(CFG)
    assert validate_raw(raw, CFG)["rows"] == len(raw)
    with pytest.raises(DataValidationError, match="Missing required columns"):
        validate_raw(raw.drop(columns=["price"]), CFG)
    short = raw[raw["invoice_date"] < raw["invoice_date"].min() + pd.Timedelta(days=30)]
    with pytest.raises(DataValidationError):
        validate_raw(short, CFG)
    bad = raw.copy()
    bad.loc[: len(bad) // 2, "description"] = np.nan
    with pytest.raises(DataValidationError, match="Description null rate"):
        validate_raw(bad, CFG)
