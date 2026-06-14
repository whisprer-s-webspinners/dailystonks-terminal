from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import datetime as dt
import random

from .models import Tier
from .registry import CARD_REGISTRY

TIER_ORDER: Dict[Tier, int] = {"free": 0, "basic": 1, "pro": 2, "black": 3}

def tier_allows(user_tier: Tier, min_tier: Tier) -> bool:
    return TIER_ORDER[user_tier] >= TIER_ORDER[min_tier]

@dataclass
class SelectionLimits:
    max_cards: int
    max_cost: int
    heavy_max: int

def _deterministic_rng(as_of: dt.date, seed: int) -> random.Random:
    if seed != 0:
        return random.Random(seed)
    # deterministic by date
    return random.Random(int(as_of.strftime("%Y%m%d")))

def select_cards(
    *,
    as_of: dt.date,
    tier: Tier,
    slot_map: Dict[str, dict],
    tier_cfg: dict,
    overrides: Dict[str, str],
    seed: int,
) -> List[str]:
    rng = _deterministic_rng(as_of, seed)

    active_slots: List[str] = list(tier_cfg["active_slots"])
    defaults: Dict[str, str] = dict(tier_cfg.get("defaults", {}))
    limits_d = dict(tier_cfg.get("limits", {}))
    limits = SelectionLimits(
        max_cards=int(limits_d.get("max_cards", 12)),
        max_cost=int(limits_d.get("max_cost", 999)),
        heavy_max=int(limits_d.get("heavy_max", 999)),
    )

    chosen: List[str] = []
    cost = 0
    heavy = 0

    for slot in active_slots:
        allowed = slot_map[slot]["allowed"]
        # Apply override if valid
        if slot in overrides:
            key = overrides[slot]
            if key not in allowed:
                raise ValueError(f"Override {slot}={key} is not allowed for this slot")
        else:
            key = defaults.get(slot)

        # If no default, rotate deterministically through allowed list (filtered by tier)
        if not key:
            candidates = [k for k in allowed if k in CARD_REGISTRY and tier_allows(tier, CARD_REGISTRY[k].min_tier)]
            if not candidates:
                continue
            key = candidates[rng.randrange(0, len(candidates))]

        # Enforce tier/min_tier
        spec = CARD_REGISTRY.get(key)
        if not spec:
            # slot might reference a card not yet implemented in prototype
            continue
        if not tier_allows(tier, spec.min_tier):
            continue

        # Budget checks
        if len(chosen) >= limits.max_cards:
            break
        if cost + spec.cost > limits.max_cost:
            continue
        if spec.heavy and heavy + 1 > limits.heavy_max:
            continue

        chosen.append(key)
        cost += spec.cost
        heavy += 1 if spec.heavy else 0

    return chosen
