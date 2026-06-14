from __future__ import annotations
from typing import Dict, Sequence, Callable
from .models import CardSpec, Tier

CARD_REGISTRY: Dict[str, CardSpec] = {}

def register_card(
    key: str,
    title: str,
    category: str,
    *,
    min_tier: Tier = "free",
    cost: int = 1,
    heavy: bool = False,
    slots: Sequence[str] = (),
    tags: Sequence[str] = (),
):
    def deco(fn: Callable):
        if key in CARD_REGISTRY:
            raise KeyError(f"Duplicate card key: {key}")
        CARD_REGISTRY[key] = CardSpec(
            key=key, title=title, category=category,
            min_tier=min_tier, cost=cost, heavy=heavy, fn=fn,
            slots=tuple(slots), tags=tuple(tags)
        )
        return fn
    return deco
