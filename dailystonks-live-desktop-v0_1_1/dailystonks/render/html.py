from __future__ import annotations
import datetime as dt
from typing import List
from ..core.models import CardResult, Artifact
from ..core.utils import b64_png

def _artifact_html(a: Artifact) -> str:
    if a.kind == "image/png":
        b64 = b64_png(a.payload)
        return f'<div class="artifact"><img alt="{a.name}" src="data:image/png;base64,{b64}" /></div>'
    if a.kind == "text/html":
        return a.payload.decode("utf-8", errors="replace")
    # text/plain
    txt = a.payload.decode("utf-8", errors="replace")
    return f"<pre>{txt}</pre>"

def render_report_html(*, as_of: dt.date, tier: str, tickers: List[str], results: List[CardResult]) -> str:
    cards_html = []
    for r in results:
        metrics = ""
        if r.metrics:
            rows = "".join([f"<tr><td>{k}</td><td>{v}</td></tr>" for k,v in r.metrics.items()])
            metrics = f"<table class='metrics'>{rows}</table>"
        bullets = ""
        if r.bullets:
            bullets = "<ul>" + "".join([f"<li>{b}</li>" for b in r.bullets]) + "</ul>"
        warns = ""
        if r.warnings:
            warns = "<div class='warn'>" + "<br/>".join(r.warnings) + "</div>"
        arts = "".join([_artifact_html(a) for a in r.artifacts])

        cards_html.append(f"""
        <section class="card">
          <div class="card-header">
            <div class="card-title">{r.title}</div>
            <div class="card-key">{r.key}</div>
          </div>
          <div class="card-body">
            {warns}
            <div class="summary">{r.summary}</div>
            {metrics}
            {bullets}
            {arts}
          </div>
        </section>
        """)

    css = """    body{font-family:Arial,Helvetica,sans-serif;background:#0b0c10;color:#e6e6e6;margin:0;padding:16px;}
    .top{display:flex;justify-content:space-between;align-items:baseline;margin-bottom:16px;}
    .brand{font-size:20px;font-weight:700;}
    .meta{opacity:0.85}
    .grid{display:grid;grid-template-columns:repeat(1,minmax(0,1fr));gap:14px;}
    .card{background:#121318;border:1px solid #20222b;border-radius:12px;padding:12px;box-shadow:0 4px 18px rgba(0,0,0,0.25);}
    .card-header{display:flex;justify-content:space-between;gap:10px;align-items:baseline;margin-bottom:6px;}
    .card-title{font-size:16px;font-weight:700}
    .card-key{font-size:12px;opacity:0.65}
    .metrics{border-collapse:collapse;width:100%;margin:8px 0;}
    .metrics td{border:1px solid #2a2d3a;padding:6px;font-size:12px;}
    .artifact img{max-width:100%;height:auto;border-radius:10px;border:1px solid #2a2d3a;margin-top:8px;}
    .warn{background:#2a1c1c;border:1px solid #5a2a2a;padding:8px;border-radius:10px;margin:6px 0;color:#ffb3b3;font-size:12px;}
    ul{margin:8px 0 0 18px;}
    li{margin:4px 0;}
    """

    return f"""<!doctype html>
<html>
<head><meta charset="utf-8"/><meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>DailyStonks {tier.title()} Report</title>
<style>{css}</style>
</head>
<body>
  <div class="top">
    <div class="brand">DailyStonks — {tier.title()} Tier</div>
    <div class="meta">As of {as_of.isoformat()} · Tickers: {", ".join(tickers)}</div>
  </div>
  <div class="grid">
    {''.join(cards_html)}
  </div>
</body>
</html>
"""
