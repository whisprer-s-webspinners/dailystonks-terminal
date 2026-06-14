from __future__ import annotations

import argparse
import cmd
import datetime as dt
import html as html_escape
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import socket
import threading
import time
import traceback
from dataclasses import dataclass, field
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import webbrowser

# The live terminal renders HTML/images for browser display, not matplotlib GUI
# windows. Force a non-interactive backend before importing card modules.
try:  # pragma: no cover - backend availability is environment-dependent
    import matplotlib

    matplotlib.use("Agg", force=True)
except Exception:
    pass

import yaml

from .core.models import CardContext, CardResult
from .core.registry import CARD_REGISTRY
from .core.selector import select_cards
from .data.marketdata import MarketData
from .data.sp500 import SP500Universe
from .render.html import render_report_html

# Import card package for registry side effects.
from . import cards as _cards  # noqa: F401

TIERS = ("free", "basic", "pro", "black")
RENDER_LOCK = threading.RLock()


def engine_root() -> Path:
    # .../engine/dailystonks/live_terminal.py -> .../engine
    return Path(__file__).resolve().parents[1]


def load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def load_slot_map() -> Dict[str, dict]:
    return load_yaml(engine_root() / "config" / "slots.yaml")


def load_tier_map() -> Dict[str, dict]:
    return load_yaml(engine_root() / "config" / "tiers.yaml")


def normalize_tier(value: str) -> str:
    t = (value or "").strip().lower()
    aliases = {"gold": "black", "custom": "black"}
    t = aliases.get(t, t)
    return t if t in TIERS else "black"


def parse_tickers(value: str | Sequence[str]) -> List[str]:
    if isinstance(value, str):
        parts = value.split(",")
    else:
        parts = list(value)
    out: List[str] = []
    for part in parts:
        ticker = str(part).strip().upper()
        if ticker and ticker not in out:
            out.append(ticker)
    return out or ["SPY", "QQQ"]


def is_intraday_interval(interval: str) -> bool:
    i = (interval or "").strip().lower()
    if i in {"1d", "5d", "1wk", "1mo", "3mo"}:
        return False
    return bool(re.fullmatch(r"\d+(m|h)", i)) or i in {"60m", "90m"}


def auto_refresh_seconds(interval: str) -> int:
    """Return a practical polling cadence for Yahoo/yfinance-style data.

    yfinance is not a tick-stream. This maps its interval strings to sane
    refresh periods so the generated browser views update no faster than the
    underlying candle cadence is likely to justify.
    """
    i = (interval or "1d").strip().lower()
    table = {
        "1m": 60,
        "2m": 120,
        "5m": 300,
        "15m": 900,
        "30m": 1800,
        "60m": 3600,
        "90m": 5400,
        "1h": 3600,
        "1d": 900,
        "5d": 1800,
        "1wk": 3600,
        "1mo": 7200,
        "3mo": 14400,
    }
    return table.get(i, 900 if not is_intraday_interval(i) else 300)


def parse_refresh_seconds(value: str | int | float, interval: str) -> int:
    if isinstance(value, (int, float)):
        return max(5, int(value))
    s = str(value or "auto").strip().lower()
    if s in {"auto", "a", "default"}:
        return auto_refresh_seconds(interval)
    try:
        return max(5, int(float(s)))
    except ValueError:
        return auto_refresh_seconds(interval)


def clamp_intraday_start(start: str, interval: str, as_of: dt.date) -> Tuple[str, Optional[str]]:
    # Yahoo has finite intraday lookback windows. This keeps live views from
    # silently failing when somebody switches 2020-01-01 + 1m.
    limits = {"1h": 720, "60m": 720, "90m": 60, "30m": 60, "15m": 60, "5m": 30, "2m": 7, "1m": 7}
    i = (interval or "").strip().lower()
    if not is_intraday_interval(i):
        return start, None
    try:
        start_dt = dt.date.fromisoformat(start)
    except Exception:
        return start, f"Could not parse start date {start!r}; leaving it unchanged."
    min_dt = as_of - dt.timedelta(days=limits.get(i, 30))
    if start_dt < min_dt:
        return min_dt.isoformat(), f"{i} data is intraday; start was clamped from {start} to {min_dt.isoformat()}."
    return start, None


def safe_filename(text: str, default: str = "view") -> str:
    s = re.sub(r"[^A-Za-z0-9_.-]+", "_", text.strip())
    s = s.strip("._-")
    return s or default


def bounded_filename(text: str, *, default: str = "view", max_stem: int = 96) -> str:
    stem = safe_filename(text, default=default)
    if len(stem) <= max_stem:
        return stem
    digest = hashlib.sha1(stem.encode("utf-8", errors="ignore")).hexdigest()[:10]
    keep = max(16, max_stem - len(digest) - 1)
    return f"{stem[:keep].rstrip('._-')}_{digest}"


@dataclass
class LiveConfig:
    tier: str = "black"
    tickers: List[str] = field(default_factory=lambda: ["SPY", "QQQ", "AAPL", "MSFT"])
    start: str = "2024-01-01"
    end: Optional[str] = None
    interval: str = "1d"
    universe: str = "sp500"
    max_universe: int = 80
    seed: int = 0
    offline_synth: bool = False
    refresh_spec: str = "auto"
    cache_ttl_ratio: float = 0.80

    def refresh_seconds(self) -> int:
        return parse_refresh_seconds(self.refresh_spec, self.interval)

    def copy(self) -> "LiveConfig":
        return LiveConfig(
            tier=self.tier,
            tickers=list(self.tickers),
            start=self.start,
            end=self.end,
            interval=self.interval,
            universe=self.universe,
            max_universe=self.max_universe,
            seed=self.seed,
            offline_synth=self.offline_synth,
            refresh_spec=self.refresh_spec,
            cache_ttl_ratio=self.cache_ttl_ratio,
        )


def selected_report_keys(config: LiveConfig, slot_map: Dict[str, dict], tier_map: Dict[str, dict]) -> List[str]:
    tier = normalize_tier(config.tier)
    tier_cfg = tier_map[tier]
    return select_cards(
        as_of=dt.date.today(),
        tier=tier,
        slot_map=slot_map,
        tier_cfg=tier_cfg,
        overrides={},
        seed=config.seed,
    )


def slot_default_key(slot: str, config: LiveConfig, tier_map: Dict[str, dict]) -> Optional[str]:
    tier = normalize_tier(config.tier)
    defaults = tier_map.get(tier, {}).get("defaults", {}) or {}
    return defaults.get(slot.upper())


def make_context(config: LiveConfig, refresh_seconds: int) -> Tuple[CardContext, List[str]]:
    root = engine_root()
    as_of = dt.date.today()
    start, note = clamp_intraday_start(config.start, config.interval, as_of)
    notes = [note] if note else []

    # Keep same-cache reuse inside one refresh cycle, but expire before the next
    # scheduled pass. Existing email/report behaviour is untouched because this
    # TTL is only passed by the live terminal path.
    ttl = max(0.0, float(refresh_seconds) * float(config.cache_ttl_ratio))
    market = MarketData(
        cache_dir=str(root / ".cache"),
        offline_synth=config.offline_synth,
        cache_ttl_seconds=ttl,
    )
    sp500 = SP500Universe(csv_path=str(root / "data" / "sp500_constituents.csv"))
    ctx = CardContext(
        as_of=as_of,
        start=start,
        end=config.end,
        interval=config.interval,
        tier=normalize_tier(config.tier),
        universe=config.universe,
        max_universe=config.max_universe,
        tickers=list(config.tickers),
        market=market,
        sp500=sp500,
        cache_dir=str(root / ".cache"),
        signals={},
    )
    return ctx, notes


def execute_cards(keys: Sequence[str], config: LiveConfig, refresh_seconds: int) -> Tuple[List[CardResult], List[str]]:
    ctx, notes = make_context(config, refresh_seconds)
    results: List[CardResult] = []

    # Matplotlib pyplot and several card modules have process-global state.
    # Serialise rendering so multiple live views do not stomp each other's figs.
    with RENDER_LOCK:
        for key in keys:
            spec = CARD_REGISTRY.get(key)
            if spec is None:
                results.append(
                    CardResult(
                        key=key,
                        title=f"Unknown card: {key}",
                        warnings=["No registered DailyStonks card has this key."],
                    )
                )
                continue
            try:
                results.append(spec.fn(ctx))
            except Exception as exc:
                results.append(
                    CardResult(
                        key=key,
                        title=spec.title,
                        warnings=[f"Card failed: {type(exc).__name__}: {exc}"],
                    )
                )
    return results, notes


def inject_live_chrome(
    html: str,
    *,
    title: str,
    view_id: str,
    source: str,
    refresh_seconds: int,
    notes: Sequence[str],
    generated_at: dt.datetime,
) -> str:
    safe_title = html_escape.escape(title)
    safe_id = html_escape.escape(view_id)
    safe_source = html_escape.escape(source)
    safe_notes = "".join(f"<li>{html_escape.escape(n)}</li>" for n in notes)
    note_block = f"<ul class='live-notes'>{safe_notes}</ul>" if safe_notes else ""
    stamp = html_escape.escape(generated_at.strftime("%Y-%m-%d %H:%M:%S"))

    head_insert = f"""
<meta http-equiv="refresh" content="{int(refresh_seconds)}"/>
<style>
.livebar{{position:sticky;top:0;z-index:9999;margin:-16px -16px 16px -16px;padding:10px 16px;background:#08090d;border-bottom:1px solid #2a2d3a;box-shadow:0 4px 16px rgba(0,0,0,.35);font-family:Arial,Helvetica,sans-serif;}}
.livebar strong{{font-size:14px;}}
.livebar .muted{{opacity:.72;font-size:12px;margin-left:8px;}}
.live-notes{{margin:6px 0 0 18px;color:#ffd38a;font-size:12px;}}
</style>
<script>
(function(){{
  var refreshSeconds = {int(refresh_seconds)};
  var remaining = refreshSeconds;
  function tick(){{
    var el = document.getElementById('live-countdown');
    if (el) {{ el.textContent = String(remaining); }}
    remaining = Math.max(0, remaining - 1);
  }}
  window.addEventListener('load', function(){{ tick(); setInterval(tick, 1000); }});
}})();
</script>
"""
    body_insert = f"""
<div class="livebar">
  <strong>{safe_title}</strong>
  <span class="muted">view={safe_id} · source={safe_source} · generated={stamp} · browser reload in <span id="live-countdown">{int(refresh_seconds)}</span>s</span>
  {note_block}
</div>
"""
    if "</head>" in html:
        html = html.replace("</head>", head_insert + "\n</head>", 1)
    else:
        html = head_insert + html
    if "<body>" in html:
        html = html.replace("<body>", "<body>\n" + body_insert, 1)
    else:
        html = body_insert + html
    return html


def render_view_html(kind: str, keys: Sequence[str], config: LiveConfig, title: str, view_id: str) -> str:
    refresh_seconds = config.refresh_seconds()
    results, notes = execute_cards(keys, config, refresh_seconds)
    generated_at = dt.datetime.now()
    html = render_report_html(
        as_of=dt.date.today(),
        tier=normalize_tier(config.tier),
        tickers=config.tickers,
        results=results,
    )
    source = f"{kind}:" + ",".join(keys)
    notes = list(notes)
    if config.offline_synth:
        notes.append("offline_synth is ON: generated synthetic OHLCV instead of querying yfinance.")
    return inject_live_chrome(
        html,
        title=title,
        view_id=view_id,
        source=source,
        refresh_seconds=refresh_seconds,
        notes=notes,
        generated_at=generated_at,
    )


def render_error_html(title: str, view_id: str, refresh_seconds: int, exc: BaseException) -> str:
    tb = html_escape.escape("".join(traceback.format_exception(type(exc), exc, exc.__traceback__)))
    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return f"""<!doctype html>
<html><head><meta charset="utf-8"/><meta http-equiv="refresh" content="{int(refresh_seconds)}"/>
<title>{html_escape.escape(title)} failed</title>
<style>
body{{font-family:Consolas,Menlo,monospace;background:#0b0c10;color:#e6e6e6;padding:20px;}}
pre{{white-space:pre-wrap;background:#181018;border:1px solid #623; padding:14px; border-radius:10px;}}
</style></head>
<body><h1>{html_escape.escape(title)} failed</h1><p>View {html_escape.escape(view_id)} · {html_escape.escape(now)} · retrying in {int(refresh_seconds)}s.</p><pre>{tb}</pre></body></html>"""


class QuietStaticHandler(SimpleHTTPRequestHandler):
    def log_message(self, fmt: str, *args) -> None:  # noqa: D401
        return


class StaticServer:
    def __init__(self, directory: Path, host: str, port: int):
        self.directory = directory
        self.host = host
        self.port = port
        self.httpd: Optional[ThreadingHTTPServer] = None
        self.thread: Optional[threading.Thread] = None

    def start(self) -> str:
        directory = str(self.directory)

        class Handler(QuietStaticHandler):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, directory=directory, **kwargs)

        self.httpd = ThreadingHTTPServer((self.host, self.port), Handler)
        self.port = int(self.httpd.server_address[1])
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True, name="dailystonks-live-http")
        self.thread.start()
        return self.base_url

    @property
    def base_url(self) -> str:
        display_host = self.host
        if display_host in {"0.0.0.0", "::"}:
            display_host = "127.0.0.1"
        return f"http://{display_host}:{self.port}"

    def stop(self) -> None:
        if self.httpd is not None:
            self.httpd.shutdown()
            self.httpd.server_close()


@dataclass
class LiveViewJob:
    view_id: str
    title: str
    kind: str
    keys: List[str]
    config: LiveConfig
    out_path: Path
    base_url: str
    stop_event: threading.Event = field(default_factory=threading.Event)
    thread: Optional[threading.Thread] = None
    last_render: Optional[dt.datetime] = None
    last_error: Optional[str] = None
    render_count: int = 0

    @property
    def url(self) -> str:
        return f"{self.base_url}/{self.out_path.name}"

    @property
    def refresh_seconds(self) -> int:
        return self.config.refresh_seconds()

    def start(self) -> None:
        self.render_once()
        self.thread = threading.Thread(target=self._loop, daemon=True, name=f"dailystonks-live-{self.view_id}")
        self.thread.start()

    def _loop(self) -> None:
        while not self.stop_event.wait(self.refresh_seconds):
            self.render_once()

    def render_once(self) -> None:
        try:
            html = render_view_html(self.kind, self.keys, self.config, self.title, self.view_id)
            self.last_error = None
        except BaseException as exc:  # keep live page self-healing
            self.last_error = f"{type(exc).__name__}: {exc}"
            html = render_error_html(self.title, self.view_id, self.refresh_seconds, exc)
        tmp_path = self.out_path.with_suffix(self.out_path.suffix + ".tmp")
        tmp_path.write_text(html, encoding="utf-8")
        tmp_path.replace(self.out_path)
        self.last_render = dt.datetime.now()
        self.render_count += 1

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)


def describe_card(key: str) -> str:
    spec = CARD_REGISTRY[key]
    heavy = "heavy" if spec.heavy else "light"
    slots = ",".join(spec.slots) if spec.slots else "-"
    return f"{key:<42} {spec.category:<12} {heavy:<5} slots={slots:<12} {spec.title}"


class DailyStonksShell(cmd.Cmd):
    intro = "DailyStonks live terminal. Type 'help' for commands; 'open report' gets you moving fast."
    prompt = "stonks> "

    def __init__(self, config: LiveConfig, out_dir: Path, server: StaticServer, browser_enabled: bool = True):
        super().__init__()
        self.config = config
        self.out_dir = out_dir
        self.server = server
        self.browser_enabled = browser_enabled
        self.slot_map = load_slot_map()
        self.tier_map = load_tier_map()
        self.jobs: Dict[str, LiveViewJob] = {}
        self._job_counter = 0

    def emptyline(self) -> None:
        return None

    def do_config(self, arg: str) -> None:
        """Show current live config."""
        print(json.dumps({
            "tier": self.config.tier,
            "tickers": self.config.tickers,
            "start": self.config.start,
            "end": self.config.end,
            "interval": self.config.interval,
            "universe": self.config.universe,
            "max_universe": self.config.max_universe,
            "seed": self.config.seed,
            "offline_synth": self.config.offline_synth,
            "refresh_spec": self.config.refresh_spec,
            "resolved_refresh_seconds": self.config.refresh_seconds(),
            "out_dir": str(self.out_dir),
            "base_url": self.server.base_url,
        }, indent=2))

    def do_set(self, arg: str) -> None:
        """Set config: set tickers SPY,QQQ,NVDA | set interval 5m | set refresh auto|60 | set tier black | set offline on."""
        try:
            parts = shlex.split(arg)
        except ValueError as exc:
            print(f"Parse error: {exc}")
            return
        if len(parts) < 2:
            print("Usage: set tickers SPY,QQQ | set interval 5m | set refresh auto|60 | set tier black | set start YYYY-MM-DD | set offline on|off")
            return
        key = parts[0].lower().replace("-", "_")
        value = " ".join(parts[1:]).strip()
        if key == "tickers":
            self.config.tickers = parse_tickers(value)
        elif key == "tier":
            self.config.tier = normalize_tier(value)
        elif key == "start":
            self.config.start = value
        elif key == "end":
            self.config.end = None if value.lower() in {"none", "today", ""} else value
        elif key == "interval":
            self.config.interval = value.lower()
        elif key in {"refresh", "refresh_seconds"}:
            self.config.refresh_spec = value
        elif key == "universe":
            self.config.universe = value.lower()
        elif key in {"max_universe", "max-universe"}:
            try:
                self.config.max_universe = max(1, int(value))
            except ValueError:
                print("max_universe must be an integer.")
                return
        elif key == "seed":
            try:
                self.config.seed = int(value)
            except ValueError:
                print("seed must be an integer.")
                return
        elif key in {"offline", "offline_synth"}:
            self.config.offline_synth = value.lower() in {"1", "true", "yes", "on", "y"}
        else:
            print(f"Unknown config key: {key}")
            return
        print(f"OK: {key} = {value}")

    def do_slots(self, arg: str) -> None:
        """List slots and the card keys each slot can display."""
        for slot, meta in self.slot_map.items():
            print(f"\n{slot} — {meta.get('title', '')}")
            for key in meta.get("allowed", []) or []:
                mark = "*" if key == slot_default_key(slot, self.config, self.tier_map) else " "
                spec = CARD_REGISTRY.get(key)
                title = spec.title if spec else "<not registered>"
                print(f"  {mark} {key:<42} {title}")
        print("\n* = current tier default")

    def do_defaults(self, arg: str) -> None:
        """Show the default cards selected by the current tier."""
        tier = normalize_tier(arg.strip() or self.config.tier)
        cfg = self.tier_map[tier]
        print(f"{tier} active slots:")
        for slot in cfg.get("active_slots", []):
            key = cfg.get("defaults", {}).get(slot)
            title = CARD_REGISTRY[key].title if key in CARD_REGISTRY else "<not registered>"
            print(f"  {slot:<4} {key:<42} {title}")

    def do_cards(self, arg: str) -> None:
        """List registered cards. Optional filter: cards risk | cards heavy | cards S09 | cards price."""
        q = (arg or "").strip().lower()
        rows: List[str] = []
        for key in sorted(CARD_REGISTRY):
            spec = CARD_REGISTRY[key]
            hay = " ".join([key, spec.title, spec.category, " ".join(spec.slots), "heavy" if spec.heavy else "light"]).lower()
            if q and q not in hay:
                continue
            rows.append(describe_card(key))
        if not rows:
            print("No matching cards.")
            return
        for row in rows:
            print(row)
        print(f"\n{len(rows)} card(s).")

    def do_search(self, arg: str) -> None:
        """Alias for cards <filter>."""
        self.do_cards(arg)

    def do_open(self, arg: str) -> None:
        """Open a live browser view.

        Forms:
          open report
          open slot S06
          open slot S06 reversal.magic_full_chart
          open card price.candles_enhanced
        """
        try:
            parts = shlex.split(arg)
        except ValueError as exc:
            print(f"Parse error: {exc}")
            return
        if not parts:
            print("Usage: open report | open slot S06 [card_key] | open card <card_key>")
            return
        mode = parts[0].lower()
        config = self.config.copy()
        if mode == "report":
            keys = selected_report_keys(config, self.slot_map, self.tier_map)
            title = f"DailyStonks {config.tier.title()} live report"
            kind = "report"
        elif mode == "slot":
            if len(parts) < 2:
                print("Usage: open slot S06 [card_key]")
                return
            slot = parts[1].upper()
            if slot not in self.slot_map:
                print(f"Unknown slot {slot}. Type 'slots' to list available slots.")
                return
            key = parts[2] if len(parts) >= 3 else slot_default_key(slot, config, self.tier_map)
            if not key:
                print(f"No default card for {slot} in tier {config.tier}; pass a card key explicitly.")
                return
            allowed = set(self.slot_map[slot].get("allowed", []) or [])
            if key not in allowed:
                print(f"{key} is not allowed in {slot}. Use 'open card {key}' to view it independently, or type 'slots'.")
                return
            if key not in CARD_REGISTRY:
                print(f"{key} is listed for {slot} but is not registered by the loaded engine.")
                return
            keys = [key]
            title = f"DailyStonks {slot}: {CARD_REGISTRY[key].title}"
            kind = f"slot:{slot}"
        elif mode == "card":
            if len(parts) < 2:
                print("Usage: open card <card_key>")
                return
            key = parts[1]
            if key not in CARD_REGISTRY:
                print(f"Unknown card key: {key}. Type 'cards {key.split('.')[0]}' or 'cards' to search.")
                return
            keys = [key]
            title = f"DailyStonks card: {CARD_REGISTRY[key].title}"
            kind = "card"
        else:
            # Convenience: allow `open price.candles_basic`.
            key = parts[0]
            if key in CARD_REGISTRY:
                keys = [key]
                title = f"DailyStonks card: {CARD_REGISTRY[key].title}"
                kind = "card"
            else:
                print("Usage: open report | open slot S06 [card_key] | open card <card_key>")
                return
        job = self._create_job(title=title, kind=kind, keys=keys, config=config)
        print(f"Opened {job.view_id}: {job.url}")
        if self.browser_enabled:
            webbrowser.open_new(job.url)

    def _create_job(self, *, title: str, kind: str, keys: List[str], config: LiveConfig) -> LiveViewJob:
        self._job_counter += 1
        base = bounded_filename(kind + "_" + "_".join(keys), max_stem=80)
        view_id = f"v{self._job_counter:03d}"
        filename = bounded_filename(f"{view_id}_{base}", max_stem=96) + ".html"
        out_path = self.out_dir / filename
        job = LiveViewJob(
            view_id=view_id,
            title=title,
            kind=kind,
            keys=list(keys),
            config=config,
            out_path=out_path,
            base_url=self.server.base_url,
        )
        job.start()
        self.jobs[view_id] = job
        return job

    def do_views(self, arg: str) -> None:
        """List currently running live views."""
        if not self.jobs:
            print("No live views yet. Try: open report")
            return
        for view_id, job in sorted(self.jobs.items()):
            stamp = job.last_render.strftime("%H:%M:%S") if job.last_render else "never"
            err = f" ERROR={job.last_error}" if job.last_error else ""
            print(f"{view_id:<5} every={job.refresh_seconds:<5}s renders={job.render_count:<4} last={stamp:<8} {job.title} -> {job.url}{err}")

    def do_refresh(self, arg: str) -> None:
        """Force refresh one view or all views: refresh v001 | refresh all."""
        target = (arg or "all").strip().lower()
        jobs = list(self.jobs.values()) if target in {"", "all", "*"} else [self.jobs.get(target)]
        jobs = [j for j in jobs if j is not None]
        if not jobs:
            print("No matching view.")
            return
        for job in jobs:
            job.render_once()
            print(f"Refreshed {job.view_id}: {job.url}")

    def do_stop(self, arg: str) -> None:
        """Stop one live view or all views: stop v001 | stop all."""
        target = (arg or "").strip().lower()
        if target in {"all", "*"}:
            ids = list(self.jobs)
        elif target:
            ids = [target]
        else:
            print("Usage: stop v001 | stop all")
            return
        for view_id in ids:
            job = self.jobs.pop(view_id, None)
            if job is None:
                print(f"No such view: {view_id}")
                continue
            job.stop()
            print(f"Stopped {view_id}")

    def do_url(self, arg: str) -> None:
        """Print the local server base URL."""
        print(self.server.base_url)

    def do_exit(self, arg: str) -> bool:
        """Exit the live terminal."""
        self._shutdown()
        return True

    def do_quit(self, arg: str) -> bool:
        """Exit the live terminal."""
        return self.do_exit(arg)

    def do_EOF(self, arg: str) -> bool:  # noqa: N802 - cmd uses this name
        print()
        return self.do_exit(arg)

    def _shutdown(self) -> None:
        for job in list(self.jobs.values()):
            job.stop()
        self.jobs.clear()
        self.server.stop()
        print("DailyStonks live terminal stopped.")


def render_once_to_file(kind_spec: str, config: LiveConfig, out_dir: Path) -> Path:
    slot_map = load_slot_map()
    tier_map = load_tier_map()
    spec = (kind_spec or "report").strip()
    lower = spec.lower()
    if lower == "report":
        keys = selected_report_keys(config, slot_map, tier_map)
        title = f"DailyStonks {config.tier.title()} once report"
        kind = "report"
    elif lower.startswith("card:"):
        key = spec.split(":", 1)[1]
        if key not in CARD_REGISTRY:
            raise SystemExit(f"Unknown card key: {key}")
        keys = [key]
        title = f"DailyStonks card: {CARD_REGISTRY[key].title}"
        kind = "card"
    elif lower.startswith("slot:"):
        body = spec.split(":", 1)[1]
        parts = body.split(":")
        slot = parts[0].upper()
        if slot not in slot_map:
            raise SystemExit(f"Unknown slot: {slot}")
        key = parts[1] if len(parts) > 1 and parts[1] else slot_default_key(slot, config, tier_map)
        if not key:
            raise SystemExit(f"No default card for {slot}; use slot:{slot}:card.key")
        if key not in CARD_REGISTRY:
            raise SystemExit(f"Unknown card key: {key}")
        keys = [key]
        title = f"DailyStonks {slot}: {CARD_REGISTRY[key].title}"
        kind = f"slot:{slot}"
    else:
        if spec in CARD_REGISTRY:
            keys = [spec]
            title = f"DailyStonks card: {CARD_REGISTRY[spec].title}"
            kind = "card"
        else:
            raise SystemExit("--once must be report, card:<key>, slot:<slot>[:card_key], or a direct card key")

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / (bounded_filename("once_" + kind + "_" + "_".join(keys), max_stem=110) + ".html")
    html = render_view_html(kind, keys, config, title, "once")
    out_path.write_text(html, encoding="utf-8")
    return out_path


def default_out_dir() -> Path:
    return engine_root() / "out" / "live_terminal"


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="DailyStonks live terminal/browser dashboard")
    ap.add_argument("--tier", default="black", choices=list(TIERS))
    ap.add_argument("--tickers", default="SPY,QQQ,AAPL,MSFT", help="comma-separated spotlight tickers")
    ap.add_argument("--start", default="2024-01-01")
    ap.add_argument("--end", default=None, help="YYYY-MM-DD; default None lets yfinance use latest")
    ap.add_argument("--interval", default="1d", help="yfinance interval: 1d, 1h, 15m, 5m, 1m, ...")
    ap.add_argument("--universe", default="sp500", choices=["sp500"])
    ap.add_argument("--max-universe", type=int, default=80)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--refresh-seconds", default="auto", help="auto or seconds; auto follows interval")
    ap.add_argument("--offline-synth", action="store_true", help="use synthetic data instead of yfinance")
    ap.add_argument("--out-dir", default=str(default_out_dir()))
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765, help="0 = pick a free port")
    ap.add_argument("--no-browser", action="store_true", help="do not automatically open browser windows")
    ap.add_argument("--once", default=None, help="render once and exit: report | card:<key> | slot:<slot>[:card_key]")
    return ap


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_arg_parser().parse_args(argv)
    config = LiveConfig(
        tier=normalize_tier(args.tier),
        tickers=parse_tickers(args.tickers),
        start=args.start,
        end=args.end,
        interval=args.interval,
        universe=args.universe,
        max_universe=args.max_universe,
        seed=args.seed,
        offline_synth=bool(args.offline_synth),
        refresh_spec=str(args.refresh_seconds),
    )
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.once:
        out_path = render_once_to_file(args.once, config, out_dir)
        print(str(out_path))
        return

    server = StaticServer(directory=out_dir, host=args.host, port=args.port)
    base_url = server.start()
    print(f"DailyStonks live server: {base_url}")
    print(f"Output directory: {out_dir}")
    print("Tip: open report | slots | cards risk | set interval 5m | set refresh auto | views")
    shell = DailyStonksShell(config=config, out_dir=out_dir, server=server, browser_enabled=not args.no_browser)
    try:
        shell.cmdloop()
    except KeyboardInterrupt:
        print()
        shell._shutdown()


if __name__ == "__main__":
    main()
