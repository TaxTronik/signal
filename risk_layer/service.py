"""Optionaler FastAPI-Service (Ausbaustufe 3, /Z80/).

Anbindungspunkt für den bestehenden Workflow (Zammad → n8n → Eskalation).
Bewusst NICHT Teil des MVP – benötigt das Extra ``service``:

    pip install ".[service]"
    uvicorn risk_layer.service:app --host 127.0.0.1 --port 8000

Betrieb strikt lokal (127.0.0.1); kein externer Netzaufruf zur Laufzeit (/NF10/).
"""

from __future__ import annotations

from typing import Any

try:
    from fastapi import FastAPI
    from pydantic import BaseModel
except ImportError as exc:  # pragma: no cover - optionaler Pfad
    raise ImportError(
        "FastAPI/Pydantic fehlen. Installation: pip install \".[service]\""
    ) from exc

from .audit import baue_audit
from .catalog import lade_katalog
from .pipeline import Pipeline
from .report import report_html

app = FastAPI(
    title="Risk-Layer",
    description="Schlüsselbegriff-Assistenz – Schicht-1-Auswertung (lokal).",
    version="0.1.0",
)

# Katalog einmalig laden und Matcher vorkompilieren.
_pipeline = Pipeline(lade_katalog())


class AnalyseRequest(BaseModel):
    text: str
    html: bool = False


@app.get("/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "katalog_version": _pipeline.katalog.version}


@app.post("/analyse")
def analyse_endpoint(req: AnalyseRequest) -> dict[str, Any]:
    report = _pipeline.run(req.text)
    antwort: dict[str, Any] = {
        "zusammenfassung": report.zusammenfassung(),
        "audit": baue_audit(report),
        "karten": [k.as_dict() for k in report.karten],
    }
    if req.html:
        antwort["html"] = report_html(report, req.text)
    return antwort
