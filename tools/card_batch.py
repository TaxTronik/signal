"""Card-Discovery über die OpenAI **Batch-API** (asynchron, −50 % Preis).

Eigenes Tool neben dem synchronen card_discovery — gleicher Karten-Kontrakt
(CARD_SYSTEM/CARD_JSON_SCHEMA) und DIESELBE verarbeite_karte-Zerlegung beim Einlesen
(Faithfulness/Jaccard/Graph/Relation/Dedup). Zwei Schritte:

  submit  Subset -> JSONL (eine Request-Zeile je Norm, custom_id = norm_id) -> Files-
          Upload -> /v1/batches anlegen. Gibt die BATCH-ID + Manifest aus.
  fetch   --batch-id: Status pollen; wenn 'completed', Ergebnis-Datei laden, je
          custom_id die Norm zuordnen und über verarbeite_karte zu Karten zerlegen.

Governance unverändert: deterministisches Subset (Modell wählt nie), KEINE Auto-Commits
in Katalog/Graph, Output nur nach discovery/ (gitignored), Secrets NUR aus ENV.

Aufruf:
  OPENAI_API_KEY=… python tools/card_batch.py submit --model <m> \
      --subset discovery/cards_subset_central.txt
  OPENAI_API_KEY=… python tools/card_batch.py fetch --batch-id batch_xxx
"""

from __future__ import annotations

import argparse
import json
import os
import secrets
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools._common import utf8_konsole, verweist_ids     # noqa: E402

from risk_layer.discovery import (                       # noqa: E402
    CARD_JSON_SCHEMA, CARD_SYSTEM, Kandidat, json_objekt, normbase,
    baue_card_prompt, besteht_review_gate, besteht_training_gate,
    iter_paragraphen, verarbeite_karte,
)
from risk_layer.graph import graph_oeffnen               # noqa: E402
from risk_layer.persistenz import atomar_schreiben       # noqa: E402

utf8_konsole()


def _lies_subset(pfad: Path) -> list[str]:
    return [z.split("#", 1)[0].strip()
            for z in pfad.read_text(encoding="utf-8").splitlines()
            if z.split("#", 1)[0].strip()]


def _api(method: str, url: str, api_key: str, *, json_body=None, raw_body=None,
         content_type: str | None = None, timeout: float = 60.0) -> bytes:
    headers = {"Authorization": f"Bearer {api_key}"}     # Key NUR im Header, nie geloggt
    data = None
    if json_body is not None:
        data = json.dumps(json_body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    elif raw_body is not None:
        data = raw_body
        if content_type:
            headers["Content-Type"] = content_type
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=timeout) as r:  # noqa: S310
        return r.read()


def _upload_jsonl(base: str, api_key: str, jsonl: bytes) -> str:
    """Files-API (multipart): purpose=batch + die JSONL. Gibt file_id."""
    boundary = "----taxtronik" + secrets.token_hex(8)
    body = b"".join([
        f'--{boundary}\r\nContent-Disposition: form-data; name="purpose"\r\n\r\nbatch\r\n'.encode(),
        (f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="batch.jsonl"'
         f'\r\nContent-Type: application/jsonl\r\n\r\n').encode(),
        jsonl, b"\r\n", f"--{boundary}--\r\n".encode(),
    ])
    out = _api("POST", f"{base}/v1/files", api_key, raw_body=body,
               content_type=f"multipart/form-data; boundary={boundary}", timeout=180)
    return json.loads(out)["id"]


def _modelle(base: str, api_key: str) -> list[str]:
    out = _api("GET", f"{base}/v1/models", api_key)
    return sorted(m.get("id", "") for m in json.loads(out).get("data", []) if m.get("id"))


def _request_body(node: dict, model: str, max_tokens: int) -> dict:
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": CARD_SYSTEM},
            {"role": "user", "content": baue_card_prompt(node["zitat"], node["titel"], node["text"])},
        ],
        "response_format": {"type": "json_schema",
                            "json_schema": {"name": "discovery", "strict": True,
                                            "schema": CARD_JSON_SCHEMA}},
        "max_completion_tokens": max_tokens,
    }


def _node_from_graph(graph, nid: str):
    """Beliebigen Knoten (auch Absatz-/Nummer-Ebene) -> Node-Dict im iter_paragraphen-
    Shape. Für die Absatz-granulare Discovery der breiten Schema-§§ (Tatbestand-Anker).
    quell_norm bleibt die volle ID; para_level rollt sie auf die §-Zielnorm hoch."""
    r = graph.con.execute(
        "SELECT id,titel,law,para,absatz,nummer,text,gueltig_ab FROM knoten WHERE id=?",
        (nid,)).fetchone()
    if not r:
        return None
    text = (r["text"] or "").strip()
    if len(text) < 15:
        return None
    zitat = f"§ {r['para']} {r['law']}"
    if r["absatz"]:
        zitat = (f"§ {r['para']} Abs. {r['absatz']}"
                 + (f" Nr. {r['nummer']}" if r["nummer"] else "") + f" {r['law']}")
    titel = (r["titel"] or "").strip()
    if not titel:                                   # Absatz-Knoten erben den §-Titel
        p = graph.con.execute("SELECT titel FROM knoten WHERE id=?",
                              (f"norm:{r['law']}:{r['para']}",)).fetchone()
        titel = ((p["titel"] if p else "") or "").strip() or zitat
    return {"id": r["id"], "law": r["law"], "zitat": zitat, "titel": titel,
            "stand": r["gueltig_ab"], "text": text}


def submit(args, base: str, api_key: str) -> int:
    graph = graph_oeffnen()
    if graph is None:
        print("Kein Normgraph — erst 'graph build'.", file=sys.stderr)
        return 2
    nodes = {n["id"]: n for n in iter_paragraphen(graph)}
    wunsch = _lies_subset(Path(args.subset)) if args.subset else \
        [nid for nid, n in nodes.items() if n["law"] == args.law] if args.law else []
    if not wunsch:
        print("Bitte --subset <datei> oder --law <LAW> angeben.", file=sys.stderr)
        return 2
    for w in wunsch:                                # Absatz-/Nummer-IDs aus dem Graph nachziehen
        if w not in nodes:
            ng = _node_from_graph(graph, w)
            if ng:
                nodes[w] = ng
    subset = [w for w in wunsch if w in nodes]
    if args.limit:
        subset = subset[:args.limit]

    lines = [json.dumps({"custom_id": nid, "method": "POST", "url": "/v1/chat/completions",
                         "body": _request_body(nodes[nid], args.model, args.max_tokens)},
                        ensure_ascii=False) for nid in subset]
    jsonl = ("\n".join(lines) + "\n").encode("utf-8")
    out = Path(args.out).parent
    out.mkdir(parents=True, exist_ok=True)
    atomar_schreiben(out / "batch_input.jsonl", jsonl.decode("utf-8"))   # lokal (gitignored) für Provenienz
    if args.dry_run:                                            # offline validierbar (kein Key/Upload)
        print(f"DRY-RUN: {len(subset)} Requests → {out}/batch_input.jsonl (kein Upload).")
        print("erste Zeile:", lines[0][:200])
        return 0
    if args.model not in _modelle(base, api_key):              # Preflight: kein Batch ins Leere
        print(f"Modell '{args.model}' nicht im Account verfügbar.", file=sys.stderr)
        return 2

    file_id = _upload_jsonl(base, api_key, jsonl)
    batch = json.loads(_api("POST", f"{base}/v1/batches", api_key, json_body={
        "input_file_id": file_id, "endpoint": "/v1/chat/completions",
        "completion_window": args.completion_window}))
    bid = batch["id"]
    atomar_schreiben(out / "batch_manifest.json", json.dumps(
        {"batch_id": bid, "input_file_id": file_id, "n": len(subset), "model": args.model,
         "stand": datetime.now(timezone.utc).isoformat(timespec="seconds")},
        ensure_ascii=False, indent=2))
    print(f"BATCH-ID: {bid}  | status: {batch.get('status')} | n: {len(subset)} "
          f"| window: {args.completion_window}")
    print(f"Manifest: {out}/batch_manifest.json")
    print(f"Abruf:    python tools/card_batch.py fetch --batch-id {bid} --out {args.out}")
    return 0


def _merge(out_base: Path, cards_dir: Path, kopf: dict) -> tuple[int, int, int]:
    review, training, n = [], [], 0
    for pfad in sorted(cards_dir.glob("norm_*.yaml")):
        d = yaml.safe_load(pfad.read_text(encoding="utf-8")) or {}
        n += 1
        for kd in d.get("kandidaten", []):
            k = Kandidat(**{f: kd.get(f) for f in Kandidat.__dataclass_fields__})
            if besteht_review_gate(k):
                review.append(kd)
            if besteht_training_gate(k):
                training.append(kd)
    atomar_schreiben(out_base / "cards_review.yaml",
            yaml.safe_dump({**kopf, "n": len(review), "kandidaten": review},
                           allow_unicode=True, sort_keys=False))
    atomar_schreiben(out_base / "cards_training_pairs.yaml",
            yaml.safe_dump({**kopf, "n": len(training), "kandidaten": training},
                           allow_unicode=True, sort_keys=False))
    return n, len(review), len(training)


def fetch(args, base: str, api_key: str) -> int:
    b = json.loads(_api("GET", f"{base}/v1/batches/{args.batch_id}", api_key))
    st = b.get("status")
    print(f"status: {st} | counts: {b.get('request_counts')}")
    if st != "completed":
        print("Noch nicht fertig — später erneut fetchen (Batch läuft async).")
        return 0
    content = _api("GET", f"{base}/v1/files/{b['output_file_id']}/content", api_key,
                   timeout=300).decode("utf-8")

    graph = graph_oeffnen()
    nodes = {n["id"]: n for n in iter_paragraphen(graph)}

    @lru_cache(maxsize=None)
    def ist_im_graph(nid: str) -> bool:
        return bool(graph.aufloesen(nid).get("gefunden"))

    cards_dir = Path(args.out)
    cards_dir.mkdir(parents=True, exist_ok=True)
    fehler: list[dict] = []
    neu = 0
    for line in content.splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        nid = r.get("custom_id")
        node = nodes.get(nid) or _node_from_graph(graph, nid)   # auch Absatz-/Nummer-IDs
        if node is None:
            continue
        resp, err = r.get("response") or {}, r.get("error")
        if err or resp.get("status_code") != 200:
            fehler.append({"id": nid, "fehler": "batch_error",
                           "detail": err or resp.get("status_code")})
            continue
        try:
            card_text = resp["body"]["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            fehler.append({"id": nid, "fehler": "no_content"})
            continue
        karte = verarbeite_karte(json_objekt(card_text), node=node,
                                 verweist_ids=verweist_ids(graph, nid), ist_im_graph=ist_im_graph)
        atomar_schreiben(cards_dir / (nid.replace(":", "_") + ".yaml"),
                yaml.safe_dump(karte.as_dict(), allow_unicode=True, sort_keys=False))
        neu += 1

    kopf = {"stand": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "provider": "openai-batch", "batch_id": args.batch_id}
    n, nrev, ntrain = _merge(cards_dir.parent, cards_dir, kopf)
    if fehler:
        atomar_schreiben(cards_dir.parent / "cards_failures.json", json.dumps(fehler, ensure_ascii=False, indent=2))
    print(f"verarbeitet {neu} | Karten gesamt {n} | Review {nrev} | Training {ntrain}"
          + (f" | AUSFÄLLE {len(fehler)}" if fehler else ""))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Card-Discovery über die OpenAI Batch-API")
    ap.add_argument("cmd", choices=("submit", "fetch"))
    ap.add_argument("--base-url", default="https://api.openai.com")
    ap.add_argument("--api-key-env", default="OPENAI_API_KEY")
    ap.add_argument("--model", default="")
    ap.add_argument("--subset", default="")
    ap.add_argument("--law", default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=8000)
    ap.add_argument("--completion-window", default="24h")
    ap.add_argument("--batch-id", default="")
    ap.add_argument("--out", default="discovery/api/cards")
    ap.add_argument("--dry-run", action="store_true", help="submit: JSONL bauen, NICHT hochladen")
    args = ap.parse_args()

    base = normbase(args.base_url)
    api_key = os.environ.get(args.api_key_env)
    if not api_key and not (args.cmd == "submit" and args.dry_run):
        print(f"Kein API-Key in ${args.api_key_env}.", file=sys.stderr)
        return 2
    if args.cmd == "submit":
        if not args.model:
            print("--model ist Pflicht.", file=sys.stderr)
            return 2
        return submit(args, base, api_key)
    if not args.batch_id:
        print("fetch braucht --batch-id.", file=sys.stderr)
        return 2
    return fetch(args, base, api_key)


if __name__ == "__main__":
    raise SystemExit(main())
