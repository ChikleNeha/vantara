"""Pydantic request/response models with clear validation messages (PRD 12.1)."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.data.categories import CATEGORIES


class CustomerRecord(BaseModel):
    """One customer's behaviour summary. Required: recency, frequency, total spend and tenure; the rest default to training medians."""

    model_config = ConfigDict(extra="forbid")

    customer_id: int | None = Field(None, description="Optional identifier echoed back in the response")
    recency_days: float = Field(..., ge=0, le=5000, description="Days since the last purchase")
    frequency: int = Field(..., ge=1, le=100000, description="Number of distinct orders")
    monetary_total: float = Field(..., ge=0, description="Total gross spend")
    tenure_days: float = Field(..., ge=0, le=10000, description="Days since the first purchase")
    monetary_avg: float | None = Field(None, ge=0, description="Average order value (default: total / orders)")
    avg_basket_size: float | None = Field(None, ge=0, description="Average units per order")
    gap_mean: float | None = Field(None, ge=0, description="Average days between orders")
    gap_std: float | None = Field(None, ge=0, description="Std-dev of days between orders")
    purchase_trend: float | None = Field(None, ge=-20, le=20, description="Slope of monthly order counts (negative = slowing)")
    seasonal_concentration: float | None = Field(None, ge=0, le=1, description="Share of orders in Oct to Dec")
    return_rate: float | None = Field(None, ge=0, le=1, description="Share of purchased units returned")
    discount_sensitivity: float | None = Field(None, ge=0, le=1, description="Share of lines bought below the usual price")
    country: str | None = Field(None, max_length=60)
    category_affinity: dict[str, float] | None = Field(None, description=f"Share of spend per category; keys from {CATEGORIES}")

    @model_validator(mode="after")
    def _consistent(self) -> "CustomerRecord":
        if self.tenure_days < self.recency_days:
            raise ValueError("tenure_days cannot be smaller than recency_days (first purchase must precede the last one)")
        if self.category_affinity:
            bad = [k for k in self.category_affinity if k not in CATEGORIES]
            if bad:
                raise ValueError(f"Unknown categories {bad}; allowed: {CATEGORIES}")
            if any(v < 0 or v > 1 for v in self.category_affinity.values()):
                raise ValueError("category_affinity values must be between 0 and 1")
        return self

    def to_row(self) -> dict[str, Any]:
        """Flat dict for the scoring bundle (affinities expanded into aff_* columns, None dropped)."""
        d = self.model_dump(exclude={"customer_id", "category_affinity"}, exclude_none=True)
        for k, v in (self.category_affinity or {}).items():
            d[f"aff_{k}"] = v
        return d


class Driver(BaseModel):
    feature: str
    label: str
    value: float
    value_text: str
    shap: float
    direction: str


class PredictionResponse(BaseModel):
    customer_id: int | None = None
    churn_probability: float
    churn_flag: bool
    risk_band: Literal["low", "medium", "high"]
    threshold: float
    clv_pred_90d: float
    segment: str
    next_category: str | None = None
    anomaly: bool
    ann_probability: float
    model: str
    drivers: list[Driver]
    narrative: str
    narrative_source: Literal["template", "ollama"] = "template"


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=500)
