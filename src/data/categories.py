"""Keyword-based product category assignment (Online Retail II has no category field)."""
from __future__ import annotations

import re

import pandas as pd

CATEGORIES = [
    "home_decor",
    "kitchen",
    "gifts_cards",
    "lighting",
    "storage",
    "garden",
    "toys_games",
    "stationery",
    "other",
]

# Order matters: the first matching rule wins.
_RULES: list[tuple[str, tuple[str, ...]]] = [
    ("lighting", ("LIGHT", "LAMP", "CANDLE", "LANTERN")),
    ("kitchen", ("MUG", "CUP", "TEAPOT", "PLATE", "BOWL", "CAKE", "JAR", "KITCHEN", "SPOON")),
    ("storage", ("BOX", "TIN", "BASKET", "BAG", "STORAGE", "CASE", "HOLDER", "RACK")),
    ("garden", ("GARDEN", "PLANT", "BIRD", "WATERING", "FLOWER", "HERB")),
    ("toys_games", ("TOY", "GAME", "PUZZLE", "DOLL", "PLAY", "CHILD", "KIDS")),
    ("stationery", ("PEN", "NOTEBOOK", "PENCIL", "STICKER", "PAPER", "STATIONERY", "ERASER")),
    ("gifts_cards", ("CARD", "GIFT", "WRAP", "RIBBON", "CHRISTMAS", "XMAS", "TAG")),
    ("home_decor", ("SIGN", "FRAME", "CUSHION", "HEART", "DECOR", "CLOCK", "MIRROR", "VINTAGE")),
]


_COMPILED = [
    (cat, re.compile(r"\b(?:" + "|".join(re.escape(k) for k in kws) + r")S?\b"))
    for cat, kws in _RULES
]


def assign_category(description: str | None) -> str:
    """Map a product description to one of ``CATEGORIES`` using whole-word keyword rules."""
    if not isinstance(description, str):
        return "other"
    text = description.upper()
    for category, pattern in _COMPILED:
        if pattern.search(text):
            return category
    return "other"


def categorize(descriptions: pd.Series) -> pd.Series:
    """Vectorised category assignment with a cache per unique description."""
    lookup = {d: assign_category(d) for d in descriptions.dropna().unique()}
    return descriptions.map(lookup).fillna("other")
