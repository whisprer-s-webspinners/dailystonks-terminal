from __future__ import annotations
from pathlib import Path
import datetime as dt
from typing import Dict, List, Optional
import yaml

from .models import CardContext
from .selector import select_cards
from .registry import CARD_REGISTRY
from ..data.sp500 import SP500Universe
from ..data.marketdata import MarketData
from ..render.html import render_report_html

# Import cards to register them
from .. import cards  # noqa: F401

def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def run_report(
    *,
    tier: str,
    out_path: str,
    start: str,
    end: Optional[str],
    interval: str,
    universe: str,
    max_universe: int,
    tickers: List[str],
    seed: int,
    overrides: Dict[str, str],
    offline_synth: bool = False,
) -> None:
    as_of = dt.date.today()
    root = Path(__file__).resolve().parents[2]
    cfg_slots = load_yaml(str(root / "config" / "slots.yaml"))
    cfg_tiers = load_yaml(str(root / "config" / "tiers.yaml"))

    cache_dir = str(root / ".cache")
    sp500 = SP500Universe(csv_path=str(root / "data" / "sp500_constituents.csv"))
    market = MarketData(cache_dir=cache_dir, offline_synth=offline_synth)

    ctx = CardContext(
        as_of=as_of,
        start=start,
        end=end,
        interval=interval,
        tier=tier,
        universe=universe,
        max_universe=max_universe,
        tickers=tickers,
        market=market,
        sp500=sp500,
        cache_dir=cache_dir,
        signals={}
    )

    tier_cfg = cfg_tiers[tier]
    chosen_keys = select_cards(
        as_of=as_of,
        tier=tier,
        slot_map=cfg_slots,
        tier_cfg=tier_cfg,
        overrides=overrides,
        seed=seed,
    )

    results = []
    for key in chosen_keys:
        spec = CARD_REGISTRY[key]
        try:
            res = spec.fn(ctx)
        except Exception as e:
            from .models import CardResult
            res = CardResult(key=key, title=spec.title, warnings=[f"Card failed: {e!r}"])
        results.append(res)


    # --- Debug manifest: what was selected/executed (for verification) ---
    try:
        import json
        manifest_path = str(out_path) + ".manifest.json"
        manifest = {
            "as_of": as_of.isoformat(),
            "tier": tier,
            "tickers": tickers,
            "active_slots": list(tier_cfg.get("active_slots", [])),
            "overrides": overrides,
            "chosen_keys": chosen_keys,
            "executed": [{"key": r.key, "title": r.title} for r in results],
        }
        Path(manifest_path).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    except Exception:
        pass

    html = render_report_html(
        as_of=as_of,
        tier=tier,
        tickers=tickers,
        results=results,
    )

    outp = Path(out_path)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(html, encoding="utf-8")
