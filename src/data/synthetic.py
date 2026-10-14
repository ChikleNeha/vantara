"""Synthetic transaction generator that mimics the Online Retail II schema.

Used when the real UCI file is not available (offline machines, CI, tests). It
reproduces the data-quality problems listed in PRD section 5.3: missing Customer
IDs, returns (invoices starting with "C" and negative quantities), zero prices,
administrative stock codes, exact duplicates, and messy descriptions. Customers
have latent behaviour (activity rate, churn time, seasonality, discount
sensitivity) so that churn is genuinely predictable, but noisy.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

_ADJ = ["RED", "BLUE", "VINTAGE", "RETRO", "WHITE", "PINK", "SPOTTY", "FLORAL", "WOODEN", "GOLD", "IVORY", "SWEET"]
_NOUNS = {
    "home_decor": ["WALL SIGN", "PHOTO FRAME", "CUSHION COVER", "HEART DECOR", "WALL CLOCK", "MIRROR"],
    "kitchen": ["MUG", "TEAPOT", "CAKE STAND", "BOWL", "JAR", "PLATE"],
    "gifts_cards": ["GIFT WRAP", "GREETING CARD", "RIBBON", "CHRISTMAS TAG", "XMAS CARD"],
    "lighting": ["LANTERN", "LAMP", "CANDLE HOLDER", "LIGHT CHAIN"],
    "storage": ["STORAGE BOX", "TIN", "BASKET", "BAG", "CASE"],
    "garden": ["GARDEN TROWEL", "BIRD FEEDER", "WATERING CAN", "HERB PLANTER"],
    "toys_games": ["WOODEN TOY", "PUZZLE", "DOLL", "BOARD GAME", "KIDS SET"],
    "stationery": ["NOTEBOOK", "PENCIL SET", "STICKER SHEET", "PAPER PAD"],
}
_CATS = list(_NOUNS)
_COUNTRIES = ["Germany", "France", "EIRE", "Spain", "Netherlands", "Belgium", "Switzerland", "Portugal", "Australia"]


def _catalog(rng: np.random.Generator, n_products: int = 420) -> pd.DataFrame:
    rows = []
    for i in range(n_products):
        cat = _CATS[i % len(_CATS)]
        noun = rng.choice(_NOUNS[cat])
        desc = f"{rng.choice(_ADJ)} {noun} {i // len(_CATS) + 1}"
        rows.append((str(20000 + i), desc, cat, float(np.round(rng.lognormal(0.9, 0.55), 2) + 0.29)))
    return pd.DataFrame(rows, columns=["StockCode", "Description", "category", "base_price"])


def generate_transactions(
    n_customers: int = 3000,
    start: str = "2009-12-01",
    end: str = "2011-12-09",
    seed: int = 42,
) -> pd.DataFrame:
    """Return a raw transaction table with the Online Retail II column names."""
    rng = np.random.default_rng(seed)
    cat = _catalog(rng)
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    months = pd.period_range(start_ts, end_ts, freq="M")
    n_months = len(months)

    # ---- latent customer traits ------------------------------------------------
    rate = np.clip(rng.lognormal(np.log(0.45), 0.8, n_customers), 0.05, 4.0)
    acquired = np.minimum((rng.beta(0.8, 1.4, n_customers) * n_months).astype(int), n_months - 1)
    loyalty = rng.beta(2, 2, n_customers)
    lifetime = rng.exponential(12 + 40 * loyalty)
    active_until = acquired + np.maximum(lifetime, 0.5)
    decay = np.where(rng.random(n_customers) < 0.20, rng.uniform(0.04, 0.2, n_customers), 0.0)
    seasonal = rng.random(n_customers) < 0.20
    reseller = rng.random(n_customers) < 0.10
    basket = np.where(reseller, rng.lognormal(np.log(18), 0.4, n_customers), rng.lognormal(np.log(7), 0.5, n_customers))
    return_p = rng.beta(1, 14, n_customers)
    disc_sens = rng.beta(1, 4, n_customers)
    prefs = rng.dirichlet(np.full(len(_CATS), 0.6), n_customers)
    country = np.where(rng.random(n_customers) < 0.9, "United Kingdom", rng.choice(_COUNTRIES, n_customers))

    # ---- orders per customer-month --------------------------------------------
    month_idx = np.arange(n_months)[None, :]
    month_of_year = np.array([p.month for p in months])[None, :]
    age = np.clip(month_idx - acquired[:, None], 0, None)
    season = np.where(seasonal[:, None], np.where(month_of_year >= 10, 4.0, 0.25), 1.0)
    lam = rate[:, None] * season * np.exp(-decay[:, None] * age)
    active = (month_idx >= acquired[:, None]) & (month_idx < active_until[:, None])
    counts = np.where(active, rng.poisson(lam), 0)

    cust_idx = np.repeat(np.arange(n_customers), counts.sum(axis=1))
    order_month = np.concatenate([np.repeat(np.arange(n_months), counts[i]) for i in range(n_customers)])
    days_in_month = np.array([p.days_in_month for p in months])[order_month]
    order_dt = (
        months[order_month].to_timestamp().to_numpy()
        + (rng.random(len(order_month)) * days_in_month).astype(int).astype("timedelta64[D]")
        + (rng.integers(8, 18, len(order_month))).astype("timedelta64[h]")
        + (rng.integers(0, 60, len(order_month))).astype("timedelta64[m]")
    )
    # guest checkouts (no Customer ID), roughly 12% extra volume
    n_guest = int(len(cust_idx) * 0.12)
    guest_dt = pd.date_range(start_ts, end_ts, periods=n_guest).to_numpy() + rng.integers(0, 3000, n_guest).astype("timedelta64[m]")
    cust_idx = np.concatenate([cust_idx, np.full(n_guest, -1)])
    order_dt = np.concatenate([order_dt, guest_dt])
    order_dt = np.minimum(order_dt, np.datetime64(end_ts + pd.Timedelta(hours=23)))

    order = np.argsort(order_dt, kind="stable")
    cust_idx, order_dt = cust_idx[order], order_dt[order]
    n_orders = len(cust_idx)
    invoice_no = 489434 + np.arange(n_orders)

    g_basket = np.where(cust_idx >= 0, basket[np.maximum(cust_idx, 0)], 6.0)
    n_lines = 1 + rng.poisson(np.maximum(g_basket - 1, 0.5))
    line_order = np.repeat(np.arange(n_orders), n_lines)
    line_cust = cust_idx[line_order]

    # ---- product choice ----------------------------------------------------------
    cum = np.cumsum(prefs, axis=1)
    u = rng.random(len(line_order))
    cat_pick = np.where(line_cust >= 0, (u[:, None] > cum[np.maximum(line_cust, 0)]).sum(axis=1), rng.integers(0, len(_CATS), len(u)))
    cat_pick = np.minimum(cat_pick, len(_CATS) - 1)
    product_ix = np.empty(len(line_order), dtype=int)
    for c, name in enumerate(_CATS):
        members = np.flatnonzero(cat["category"].to_numpy() == name)
        mask = cat_pick == c
        product_ix[mask] = members[rng.integers(0, len(members), mask.sum())]

    qty_scale = np.where(line_cust >= 0, np.where(reseller[np.maximum(line_cust, 0)], 6, 1), 1)
    quantity = rng.geometric(0.28, len(line_order)) * qty_scale
    price = cat["base_price"].to_numpy()[product_ix]
    ds = np.where(line_cust >= 0, disc_sens[np.maximum(line_cust, 0)], 0.05)
    discounted = rng.random(len(price)) < ds * 0.9
    price = np.round(price * np.where(discounted, rng.uniform(0.6, 0.85, len(price)), 1.0), 2)

    lines = pd.DataFrame(
        {
            "Invoice": invoice_no[line_order].astype(str),
            "StockCode": cat["StockCode"].to_numpy()[product_ix],
            "Description": cat["Description"].to_numpy()[product_ix],
            "Quantity": quantity,
            "InvoiceDate": order_dt[line_order],
            "Price": price,
            "Customer ID": np.where(line_cust >= 0, 12000.0 + line_cust, np.nan),
            "Country": np.where(line_cust >= 0, country[np.maximum(line_cust, 0)], "United Kingdom"),
        }
    )

    # ---- returns / cancellations ---------------------------------------------
    order_ret_p = np.where(cust_idx >= 0, return_p[np.maximum(cust_idx, 0)], 0.01)
    returning = np.flatnonzero(rng.random(n_orders) < order_ret_p)
    ret_parts = []
    in_ret = lines[lines["Invoice"].isin(set(invoice_no[returning].astype(str)))]
    keep = in_ret.sample(frac=0.45, random_state=seed)
    if len(keep):
        ret = keep.copy()
        ret["Invoice"] = "C" + (900000 + pd.factorize(keep["Invoice"])[0]).astype(str)
        ret["Quantity"] = -ret["Quantity"]
        ret["InvoiceDate"] = ret["InvoiceDate"] + pd.to_timedelta(rng.integers(1, 15, len(ret)), unit="D")
        ret_parts.append(ret[ret["InvoiceDate"] <= end_ts + pd.Timedelta(hours=23)])

    # ---- administrative lines, zero prices, duplicates, messy text -------------
    admin = lines.drop_duplicates("Invoice").sample(frac=0.08, random_state=seed + 1).copy()
    admin["StockCode"], admin["Description"] = "POST", "POSTAGE"
    admin["Quantity"], admin["Price"] = 1, 18.0
    parts = [lines, admin] + ret_parts
    out = pd.concat(parts, ignore_index=True)

    zero = rng.random(len(out)) < 0.001
    out.loc[zero, "Price"] = 0.0
    out.loc[rng.random(len(out)) < 0.0005, "Price"] = -1.0
    messy = rng.random(len(out)) < 0.02
    out.loc[messy, "Description"] = out.loc[messy, "Description"].str.lower() + " "
    out.loc[rng.random(len(out)) < 0.004, "Description"] = np.nan
    out.loc[rng.random(len(out)) < 0.0003, "Quantity"] = 80995  # data-entry style outlier
    dupes = out.sample(frac=0.003, random_state=seed + 2)
    out = pd.concat([out, dupes], ignore_index=True)

    out = out.sort_values("InvoiceDate", kind="stable").reset_index(drop=True)
    out["InvoiceDate"] = pd.to_datetime(out["InvoiceDate"]).dt.floor("min")
    return out[["Invoice", "StockCode", "Description", "Quantity", "InvoiceDate", "Price", "Customer ID", "Country"]]
