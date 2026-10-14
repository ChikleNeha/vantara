"""SQLite persistence (PRD 12.1; SQLite replaces PostgreSQL by project decision). Schema = ER diagram source of truth."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sqlalchemy import Boolean, Column, DateTime, Float, ForeignKey, Index, Integer, String, Text, create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import declarative_base

Base = declarative_base()


class Customer(Base):
    __tablename__ = "customers"
    customer_id = Column(Integer, primary_key=True)
    country = Column(String)
    first_purchase = Column(DateTime)
    last_purchase = Column(DateTime)
    frequency = Column(Integer)
    monetary_total = Column(Float)
    clv_historical = Column(Float)
    avg_spend_90d = Column(Float)
    value_tier = Column(String, index=True)
    engagement_score = Column(Float)
    features_json = Column(Text)


class Prediction(Base):
    __tablename__ = "predictions"
    customer_id = Column(Integer, ForeignKey("customers.customer_id"), primary_key=True)
    churn_prob = Column(Float, index=True)
    churn_flag = Column(Boolean)
    risk_band = Column(String, index=True)
    clv_pred_90d = Column(Float)
    priority_score = Column(Float, index=True)
    ann_prob = Column(Float)
    lstm_prob = Column(Float)
    anomaly_score = Column(Float)
    is_anomaly = Column(Boolean)
    next_category = Column(String)
    model_name = Column(String)
    threshold = Column(Float)
    scored_at = Column(DateTime)


class Segment(Base):
    __tablename__ = "segments"
    customer_id = Column(Integer, ForeignKey("customers.customer_id"), primary_key=True)
    segment_id = Column(Integer, index=True)
    segment_label = Column(String)
    gmm_cluster = Column(Integer)
    pca_x = Column(Float)
    pca_y = Column(Float)


class SegmentProfile(Base):
    __tablename__ = "segment_profiles"
    segment_id = Column(Integer, primary_key=True)
    label = Column(String)
    description = Column(Text)
    size = Column(Integer)
    share = Column(Float)
    recency_days = Column(Float)
    frequency = Column(Float)
    monetary_total = Column(Float)
    avg_basket_size = Column(Float)
    return_rate = Column(Float)
    discount_sensitivity = Column(Float)
    seasonal_concentration = Column(Float)
    purchase_trend = Column(Float)
    clv_historical = Column(Float)
    engagement_score = Column(Float)


class MonthlyRevenue(Base):
    __tablename__ = "monthly_revenue"
    month = Column(String, primary_key=True)
    revenue = Column(Float)
    orders = Column(Float)
    customers = Column(Float)
    is_partial = Column(Boolean)
    is_forecast = Column(Boolean)


class Transaction(Base):
    __tablename__ = "transactions"
    id = Column(Integer, primary_key=True, autoincrement=True)
    invoice = Column(String)
    invoice_date = Column(DateTime)
    customer_id = Column(Integer, index=True)
    stock_code = Column(String)
    description = Column(String)
    category = Column(String)
    quantity = Column(Integer)
    price = Column(Float)
    line_total = Column(Float)
    is_return = Column(Boolean)
    country = Column(String)


Index("ix_tx_customer_date", Transaction.customer_id, Transaction.invoice_date)


def get_engine(db_path: Path | str) -> Engine:
    """SQLite engine with foreign keys on and WAL for concurrent dashboard reads."""
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(f"sqlite:///{db_path}", future=True)

    @event.listens_for(engine, "connect")
    def _pragmas(dbapi_conn, _):  # noqa: ANN001
        cur = dbapi_conn.cursor()
        cur.execute("PRAGMA journal_mode=WAL")
        cur.execute("PRAGMA foreign_keys=ON")
        cur.close()

    return engine


def reset_schema(engine: Engine) -> None:
    """Drop and recreate all tables (the pipeline owns the data; the API only reads)."""
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)


def append(engine: Engine, table: str, df: pd.DataFrame, chunksize: int = 50000) -> None:
    """Bulk insert a frame into an existing table."""
    df.to_sql(table, engine, if_exists="append", index=False, chunksize=chunksize)
