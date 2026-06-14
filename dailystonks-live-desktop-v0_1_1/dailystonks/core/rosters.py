
from __future__ import annotations

import hashlib
from datetime import date
from pathlib import Path
from typing import Any, Dict, Tuple

import yaml

TIER_RANK = {"free": 0, "basic": 1, "pro": 2, "gold": 3, "black": 4}

def _stable_int(s: str) -> int:
    h = hashlib.sha256(s.encode("utf-8")).hexdigest()
    return int(h[:16], 16)

def _pick(slot: str, candidates: list[str], seed: str) -> str:
    i = _stable_int(f"{seed}|{slot}") % len(candidates)
    return candidates[i]

def _min_tier_of(spec: Any) -> str:
    return getattr(spec, "min_tier", "free") or "free"

def list_rosters(config_dir: str = "config") -> list[str]:
    rosters_p = Path(config_dir) / "rosters.yaml"
    if not rosters_p.exists():
        return []
    rosters = yaml.safe_load(rosters_p.read_text(encoding="utf-8")) or {}
    spec = (rosters.get("rosters", rosters) or {})
    return sorted(list(spec.keys()))

def resolve_roster(
    roster_name: str,
    tier: str,
    seed: str | None = None,
    config_dir: str = "config",
    strict: bool = False,
) -> Tuple[Dict[str, str], Dict[str, Any]]:
    seed = seed or date.today().isoformat()

    rosters_p = Path(config_dir) / "rosters.yaml"
    tiers_p = Path(config_dir) / "tiers.yaml"
    slots_p = Path(config_dir) / "slots.yaml"

    rosters = yaml.safe_load(rosters_p.read_text(encoding="utf-8")) if rosters_p.exists() else {}
    tiers = yaml.safe_load(tiers_p.read_text(encoding="utf-8")) if tiers_p.exists() else {}
    slots_cfg = yaml.safe_load(slots_p.read_text(encoding="utf-8")) if slots_p.exists() else {}

    spec = (rosters.get("rosters", rosters) or {}).get(roster_name)
    if not spec:
        raise ValueError(f"Unknown roster '{roster_name}' (check {rosters_p})")

    tier_cfg = tiers.get(tier, {}) or {}
    active_slots = list(tier_cfg.get("active_slots", []) or [])

    import dailystonks.cards  # noqa: F401
    from .registry import CARD_REGISTRY

    def allowed_for_slot(slot: str) -> set[str]:
        return set((slots_cfg.get(slot, {}) or {}).get("allowed", []) or [])

    def ok_key(slot: str, key: str) -> tuple[bool, str]:
        if slot not in active_slots:
            return False, "slot_not_active_for_tier"
        if key not in CARD_REGISTRY:
            return False, "key_not_registered"
        if key not in allowed_for_slot(slot):
            return False, "key_not_allowed_for_slot"
        mt = _min_tier_of(CARD_REGISTRY[key])
        if TIER_RANK.get(mt, 0) > TIER_RANK.get(tier, 0):
            return False, "min_tier_too_high"
        return True, "ok"

    plan: dict[str, Any] = {
        "roster": roster_name,
        "tier": tier,
        "seed": seed,
        "active_slots": active_slots,
        "applied": {},
        "skipped": [],
    }
    overrides: dict[str, str] = {}

    fixed = spec.get("fixed", {}) or {}
    rotate = spec.get("rotate", {}) or {}

    for slot, key in fixed.items():
        ok, why = ok_key(slot, key)
        if ok:
            overrides[slot] = key
            plan["applied"][slot] = {"key": key, "mode": "fixed"}
        else:
            plan["skipped"].append({"slot": slot, "key": key, "mode": "fixed", "reason": why})
            if strict:
                raise ValueError(f"Roster fixed {slot}={key} invalid: {why}")

    for slot, cand in rotate.items():
        cand = list(cand or [])
        valid = []
        reasons = []
        for key in cand:
            ok, why = ok_key(slot, key)
            if ok:
                valid.append(key)
            else:
                reasons.append((key, why))

        if not valid:
            plan["skipped"].append({"slot": slot, "mode": "rotate", "reason": "no_valid_candidates", "details": reasons[:10]})
            if strict:
                raise ValueError(f"Roster rotate {slot} has no valid candidates.")
            continue

        picked = _pick(slot, valid, seed)
        overrides[slot] = picked
        plan["applied"][slot] = {"key": picked, "mode": "rotate", "candidates": valid}

    return overrides, plan
