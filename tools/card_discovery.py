"""Card-Discovery (Pass 2): kuratierte Anker je DETERMINISTISCH vorselektierter Norm,
über die OpenAI-API ODER ein lokales OpenAI-kompatibles Endpoint.

Governance (siehe risk_layer.discovery):
- Die Norm-Auswahl kommt aus dem Subset (tools/card_subset.py / --law), NIE vom Modell.
- Das Modell bildet nur Anker, wenn der Wortlaut sie trägt (sonst anchors: []).
- Jede Karte wird über DIESELBEN Evidenz-Gates wie Pass 1 zerlegt (verarbeite_karte):
  Faithfulness, Jaccard-Register, Graph-Existenz, Dedup, Deckel.
- KEINE Auto-Commits in Katalog/Graph; Output nur nach discovery/ (gitignored).
- Secrets NUR aus ENV (--api-key-env), nie geloggt.

Aufruf:
  python tools/card_subset.py --expand                 # Subset erzeugen
  OPENAI_API_KEY=... python tools/card_discovery.py --provider openai \
      --model <modell> --subset discovery/cards_subset.txt --limit 30
  python tools/card_discovery.py --provider local --base-url http://127.0.0.1:8000 \
      --model <lokal> --law AO
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

from risk_layer.discovery import (                       # noqa: E402
    Karte, _normbase, besteht_review_gate, besteht_training_gate, iter_paragraphen,
    roh_karte, server_verfuegbar, verarbeite_karte,
)
from risk_layer.graph import graph_oeffnen               # noqa: E402
from risk_layer.idschema import para_level               # noqa: E402

PROVIDER_BASEURL = {"local": "http://127.0.0.1:8000", "openai": "https://api.openai.com"}


def _atomar(pfad: Path, text: str) -> None:
    tmp = pfad.with_suffix(pfad.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, pfad)


def _verweist_ids(graph, node_id: str) -> set[str]:
    ids: set[str] = set()
    for v in graph.aufloesen(node_id).get("verweist_auf") or []:
        i = (v.get("id") or v.get("dst") or v.get("ziel") or "") if isinstance(v, dict) else str(v)
        if i:
            ids.add(i)
    return ids


def _openai_modelle(base_url: str, api_key: str) -> list[str]:
    """Modell-IDs des Accounts (GET /v1/models) — für den Preflight-Check, damit ein
    falscher --model nicht 30× als 404 endet."""
    req = urllib.request.Request(f"{_normbase(base_url)}/v1/models",
                                 headers={"Authorization": f"Bearer {api_key}"})
    with urllib.request.urlopen(req, timeout=15) as r:  # noqa: S310
        return sorted(m.get("id", "") for m in json.load(r).get("data", []) if m.get("id"))


def _lies_subset(pfad: Path) -> list[str]:
    out: list[str] = []
    for zeile in pfad.read_text(encoding="utf-8").splitlines():
        z = zeile.split("#", 1)[0].strip()
        if z:
            out.append(z)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Card-Discovery (Pass 2) lokal oder OpenAI")
    ap.add_argument("--provider", choices=("local", "openai"), default="openai")
    ap.add_argument("--base-url", default="", help="überschreibt Provider-Default")
    ap.add_argument("--api-key-env", default="OPENAI_API_KEY", help="ENV-Var mit dem API-Key")
    ap.add_argument("--model", default="", help="Modellname (für OpenAI Pflicht)")
    ap.add_argument("--subset", default="", help="Datei mit norm_ids (ein je Zeile, # Kommentar)")
    ap.add_argument("--law", default="", help="statt --subset: alle §§ eines Gesetzes")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--timeout", type=float, default=300.0)
    ap.add_argument("--max-tokens", type=int, default=8000,
                    help="max_completion_tokens — großzügig, da Reasoning-Modelle Budget beim "
                         "Denken verbrauchen (zu klein → leere Karten bei langen §§)")
    ap.add_argument("--out", default="discovery/cards")
    args = ap.parse_args()

    base_url = args.base_url or PROVIDER_BASEURL[args.provider]
    api_key = os.environ.get(args.api_key_env) if args.provider == "openai" else None
    if args.provider == "openai" and not api_key:
        print(f"Kein API-Key in ${args.api_key_env}.", file=sys.stderr)
        return 2
    if args.provider == "openai" and not args.model:
        print("--model ist für --provider openai Pflicht.", file=sys.stderr)
        return 2
    if not server_verfuegbar(base_url, timeout=5.0) and args.provider == "local":
        print(f"Kein Endpoint unter {base_url}.", file=sys.stderr)
        return 2
    if args.provider == "openai":                   # Preflight: Modell existiert + Auth ok?
        try:
            verfuegbar = _openai_modelle(base_url, api_key)
        except urllib.error.HTTPError as e:
            print(f"Modell-Liste abgelehnt (HTTP {e.code}) — Key/Zugang prüfen: "
                  f"{e.read().decode('utf-8', 'replace')[:200]}", file=sys.stderr)
            return 2
        except (urllib.error.URLError, OSError, ValueError) as e:
            print(f"Modell-Liste nicht abrufbar ({type(e).__name__}).", file=sys.stderr)
            return 2
        if args.model not in verfuegbar:
            print(f"Modell '{args.model}' ist im Account nicht verfügbar. Verfügbar:\n  "
                  + "\n  ".join(verfuegbar), file=sys.stderr)
            return 2

    graph = graph_oeffnen()
    if graph is None:
        print("Kein Normgraph — erst 'graph build'.", file=sys.stderr)
        return 2
    nodes = {n["id"]: n for n in iter_paragraphen(graph)}

    if args.subset:
        wunsch = _lies_subset(Path(args.subset))
    elif args.law:
        wunsch = [nid for nid, n in nodes.items() if n["law"] == args.law]
    else:
        print("Bitte --subset <datei> oder --law <LAW> angeben (deterministisches Subset).",
              file=sys.stderr)
        return 2
    fehlend = [w for w in wunsch if w not in nodes]
    subset = [w for w in wunsch if w in nodes]
    if fehlend:
        print(f"WARN: {len(fehlend)} norm_ids nicht im Graph (übersprungen), z.B. {fehlend[:3]}",
              file=sys.stderr)
    if args.limit:
        subset = subset[:args.limit]
    if not subset:
        print("Leeres Subset.", file=sys.stderr)
        return 2

    out = Path(args.out)
    karten_dir = out
    karten_dir.mkdir(parents=True, exist_ok=True)

    @lru_cache(maxsize=None)
    def ist_im_graph(nid: str) -> bool:
        return bool(graph.aufloesen(nid).get("gefunden"))

    def _datei(nid: str) -> Path:
        return karten_dir / (nid.replace(":", "_") + ".yaml")

    def _offen(nid: str) -> bool:                   # offen = fehlt ODER leer (leere Karten nachfahren)
        p = _datei(nid)
        if not p.exists():
            return True
        try:
            d = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError:
            return True
        return not (d.get("kandidaten") or d.get("anker_vorgeschlagen"))

    offen = [nid for nid in subset if _offen(nid)]
    print(f"Subset {len(subset)} Normen · offen {len(offen)} · provider={args.provider} "
          f"model={args.model or '(lokal-auto)'} concurrency={args.concurrency}", file=sys.stderr)

    fehler: list[dict] = []

    def _roh(nid: str):
        return nid, roh_karte(base_url, nodes[nid], model=args.model, timeout=args.timeout,
                              api_key=api_key, provider=args.provider,
                              n_predict=args.max_tokens, fehler_protokoll=fehler)

    # Phase 1: API-Calls PARALLEL (thread-safe). Phase 2: Graph-Zerlegung im Hauptthread.
    with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as ex:
        rohkarten = list(ex.map(_roh, offen))

    neu = 0
    for nid, card in rohkarten:
        node = nodes[nid]
        karte = verarbeite_karte(card, node=node, verweist_ids=_verweist_ids(graph, nid),
                                 ist_im_graph=ist_im_graph)
        _atomar(_datei(nid), yaml.safe_dump(karte.as_dict(), allow_unicode=True, sort_keys=False))
        neu += 1

    # Merge ALLER vorhandenen Karten → Review/Training/Failures
    review: list[dict] = []
    training: list[dict] = []
    n_karten = 0
    for pfad in sorted(karten_dir.glob("norm_*.yaml")):
        d = yaml.safe_load(pfad.read_text(encoding="utf-8")) or {}
        n_karten += 1
        for kd in d.get("kandidaten", []):
            from risk_layer.discovery import Kandidat
            k = Kandidat(**{f: kd.get(f) for f in Kandidat.__dataclass_fields__})
            if besteht_review_gate(k):
                review.append(kd)
            if besteht_training_gate(k):
                training.append(kd)

    kopf = {"stand": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "provider": args.provider, "modell": args.model,
            "karten": n_karten, "subset": len(subset)}
    _atomar(out.parent / "cards_review.yaml",
            yaml.safe_dump({**kopf, "n": len(review), "kandidaten": review},
                           allow_unicode=True, sort_keys=False))
    _atomar(out.parent / "cards_training_pairs.yaml",
            yaml.safe_dump({**kopf, "n": len(training), "kandidaten": training},
                           allow_unicode=True, sort_keys=False))
    if fehler:
        _atomar(out.parent / "cards_failures.json", json.dumps(fehler, ensure_ascii=False, indent=2))

    print(f"\nKarten gesamt {n_karten} (neu {neu}) · Review {len(review)} · "
          f"Training {len(training)}" + (f" · AUSFÄLLE {len(fehler)}" if fehler else ""))
    print(f"geschrieben: {karten_dir}/norm_*.yaml · {out.parent}/cards_review.yaml · "
          f"cards_training_pairs.yaml")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
