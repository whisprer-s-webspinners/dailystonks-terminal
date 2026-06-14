from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Literal
import datetime as _dt

Tier = Literal["free","basic","pro","black"]
ArtifactType = Literal["image/png","text/html","text/plain"]

@dataclass(frozen=True)
class Artifact:
    kind: ArtifactType
    name: str
    payload: bytes

@dataclass
class CardResult:
    key: str
    title: str
    summary: str = ""
    metrics: Dict[str, Any] = field(default_factory=dict)
    bullets: List[str] = field(default_factory=list)
    artifacts: List[Artifact] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

@dataclass(frozen=True)
class CardSpec:
    key: str
    title: str
    category: str
    min_tier: Tier
    cost: int
    heavy: bool
    fn: Callable[["CardContext"], CardResult]
    slots: Sequence[str] = ()
    tags: Sequence[str] = ()

@dataclass
class CardContext:
    as_of: _dt.date
    start: str
    end: Optional[str]
    interval: str
    tier: Tier
    universe: str
    max_universe: int
    tickers: List[str]
    market: Any  # MarketData
    sp500: Any   # SP500Universe
    cache_dir: str
    signals: Dict[str, float] = field(default_factory=dict)
