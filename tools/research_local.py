"""Experimental local research CLI; connects to an already running server only."""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import urllib.request


ROOT = Path(__file__).resolve().parents[1]


def _runtime():
    # Keep direct script invocation usable without an editable installation.
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from risk_layer.norm_check import _strict_json
    from risk_layer.research_agent import ResearchAgent, ResearchConfig
    from risk_layer.research_sources import ResearchCorpus
    return ResearchAgent, ResearchConfig, ResearchCorpus, _strict_json


def parser():
    result = argparse.ArgumentParser(description=(
        "Experimentelle lokale Normrecherche mit einem bereits gestarteten Modellserver. "
        "Startet keinen Server und lädt keine Modelle herunter."))
    result.add_argument("input", help="UTF-8-Textdatei oder - für Standardeingabe")
    result.add_argument("--graph", type=Path, required=True, help="NormGraph-Datei")
    result.add_argument("--index", type=Path, required=True, help="Indexverzeichnis mit PassageEvidence")
    result.add_argument("--catalog", type=Path,
                        help="Optionaler Katalog für ungeprüfte Suchhinweise; Hinweise sind keine Quellenbelege")
    result.add_argument("--url", required=True, help="HTTP-Adresse des lokalen Loopback-Servers")
    result.add_argument("--model", required=True, help="Modellalias des laufenden Servers")
    result.add_argument("--retrieval", choices=("bm25", "hybrid"), default="bm25",
                        help="Quellenzugang: BM25 oder CPU-Hybrid mit BGE-M3 und BM25 (bm25)")
    result.add_argument("--encoder-model", type=Path,
                        help="Gepinntes lokales BGE-M3-Modell für Hybrid; sonst vorhandener lokaler HF-Cache")
    result.add_argument("--timeout", type=float, default=180., help="Zeitbudget je Recherche in Sekunden (180)")
    result.add_argument("--reuse-prompt-cache", action="store_true",
                        help="Promptpräfix innerhalb eines Falls wiederverwenden; erster Request ohne Cache")
    result.add_argument("--finish-reserve", type=float, default=0.,
                        help="0 = bisherige Zeitsteuerung; positiv = Reserve in Sekunden je weiterer Runde "
                             "und Abschluss; die Zulassung prüft mehr als 2 * Wert Restzeit")
    result.add_argument("--static", action="store_true", help="Nur Startquellen verwenden; keine weitere Recherche")
    result.add_argument("--json", action="store_true", help="Maschinenlesbares Ergebnis ohne Dialog- und Thinkingtrace")
    result.add_argument("--trace", type=Path, help="Neue Datei für den vollständigen privaten Lauf einschließlich Dialog")
    return result


def read_text(filename):
    if filename == "-":
        stream = getattr(sys.stdin, "buffer", sys.stdin)
        value = stream.read()
    else:
        value = Path(filename).read_bytes()
    text = value.decode("utf-8-sig") if isinstance(value, bytes) else value
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Die Eingabe muss einen nichtleeren UTF-8-Sachverhalt enthalten.")
    return text


def check_trace_path(path):
    if path is None:
        return
    if os.path.lexists(path):
        raise ValueError("Die Trace-Datei existiert bereits; bitte einen neuen Dateipfad verwenden.")
    if not path.parent.is_dir():
        raise ValueError("Das Verzeichnis für die Trace-Datei muss bereits existieren.")


def write_fresh_json(path, value):
    """Publish a complete same-directory file atomically, never replacing a target."""
    check_trace_path(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n",
                                         prefix=".research-", suffix=".tmp", dir=path.parent,
                                         delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        # A hard link installs the complete inode and fails if another writer
        # has created the destination since our preflight. replace() would not.
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def check_server(agent, config, parse_json):
    """Use exactly the Agent's loopback-validated, proxy/redirect-free opener."""
    started = time.perf_counter()
    request = urllib.request.Request(agent.base_url + "/props", method="GET")
    with agent._opener.open(request, timeout=min(config.timeout_seconds, 15.)) as response:
        raw = response.read(2_097_153)
    if len(raw) > 2_097_152:
        raise ValueError("Die Servermetadaten überschreiten das Größenlimit.")
    props = parse_json(raw)
    if not isinstance(props, dict) or props.get("model_alias") != agent.model:
        raise ValueError("Der laufende Server meldet einen anderen Modellalias.")
    generation = props.get("default_generation_settings")
    context = generation.get("n_ctx") if isinstance(generation, dict) else None
    if type(context) is not int or context < config.context_tokens:
        raise ValueError(f"Der Server benötigt mindestens {config.context_tokens} Kontexttokens.")
    if type(props.get("total_slots")) is not int or props["total_slots"] != 1:
        raise ValueError("Der Forschungsserver muss genau einen Slot bereitstellen.")
    return {"props": props, "props_sha256": hashlib.sha256(raw).hexdigest(),
            "elapsed_seconds": time.perf_counter() - started}


def public_result(result):
    # No HTTP requests, assistant reasoning, full dialog or alias ledger in the
    # ordinary output. Source evidence on accepted suggestions remains usable.
    keys = ("status", "model", "prompt_version", "allow_research", "suggestions",
            "unresolved", "errors", "timing", "usage", "source_budget", "source_prepare_seconds",
            "retrieval_metadata", "retrieval_metrics_before_run", "retrieval_metrics", "execution_policy",
            "catalog_metadata")
    return {key: result[key] for key in keys if key in result}


def render_text(result):
    lines = [f"Experimentelle Normrecherche – Status: {result['status']}"]
    for focus in result.get("suggestions", []):
        lines.extend(["", f"Fokus [{focus['start']}:{focus['end']}]: {focus['satz']}"])
        for norm in focus["normen"]:
            label = norm.get("zitat") or norm["id"]
            title = norm.get("titel")
            lines.append(f"  Vorschlag: {label}" + (f" – {title}" if title else ""))
            research = norm.get("research", {})
            if research.get("reason"):
                lines.append(f"    Begründung: {research['reason']}")
            for key, label in (("source_quote", "Sachverhaltszitat"), ("norm_quote", "Normzitat")):
                quote = research.get(key)
                if isinstance(quote, dict):
                    lines.append(f"    {label} [{quote['start']}:{quote['end']}]: „{quote['text']}“")
    if result.get("status") == "complete" and not result.get("suggestions"):
        lines.append("Keine Normvorschläge abgegeben.")
    for question in result.get("unresolved", []):
        lines.append(f"Offen: {question}")
    for error in result.get("errors", []):
        if isinstance(error, dict):
            lines.append(f"Hinweis ({error.get('kind', 'Fehler')}): {error.get('message', '')}")
        else:
            lines.append(f"Hinweis: {error}")
    timing = result.get("timing", {})
    retrieval = result.get("retrieval_metadata", {})
    if retrieval.get("mode"):
        lines.append(f"Quellenzugang: {retrieval['mode']}")
    if isinstance(result.get("catalog_metadata"), dict):
        lines.append("Kataloghinweise: aktiviert (ungeprüfte Suchhinweise; keine Belege)")
    if isinstance(result.get("source_prepare_seconds"), (int, float)):
        lines.append(f"Quellenvorbereitung zusätzlich: {result['source_prepare_seconds']:.2f} s")
    if isinstance(timing.get("elapsed_seconds"), (int, float)):
        lines.append(f"Dauer: {timing['elapsed_seconds']:.2f} s")
    policy = result.get("execution_policy")
    if isinstance(policy, dict):
        cache = "aktiviert" if policy["reuse_prompt_cache"] else "aus"
        reserve = policy["finish_reserve_seconds"]
        time_control = ("bisherige Zeitsteuerung (Abschlussmodus unter 30 s Restzeit)" if reserve == 0 else
                        f"Reserve je weiterer Runde und Abschluss {reserve:g} s "
                        f"(Zulassung prüft mehr als {2 * reserve:g} s Restzeit)")
        lines.append(f"Ausführung: Promptcache-Wiederverwendung {cache}; {time_control}")
    return "\n".join(lines)


def main(argv=None):
    args = parser().parse_args(argv)
    corpus = None
    try:
        check_trace_path(args.trace)
        if args.encoder_model is not None and args.retrieval != "hybrid":
            raise ValueError("--encoder-model benötigt --retrieval hybrid.")
        text = read_text(args.input)
        Agent, Config, Corpus, parse_json = _runtime()
        config = Config(timeout_seconds=args.timeout, reuse_prompt_cache=args.reuse_prompt_cache,
                        finish_reserve_seconds=args.finish_reserve)
        agent = Agent(args.url, args.model, config)
        source_started = time.perf_counter()
        with redirect_stdout(sys.stderr):
            catalog_options = {"catalog_path": args.catalog} if args.catalog is not None else {}
            corpus = Corpus(args.graph, args.index, retrieval=args.retrieval, encoder_model_path=args.encoder_model,
                            **catalog_options)
        identity = corpus.identity_metadata()
        source_prepare_seconds = time.perf_counter() - source_started
        server = check_server(agent, config, parse_json)
        source_started = time.perf_counter()
        with redirect_stdout(sys.stderr):
            corpus.prepare()
        source_prepare_seconds += time.perf_counter() - source_started
        retrieval_metadata = corpus.retrieval_metadata()
        metrics_before = corpus.retrieval_metrics
        result = {**agent.run(text, corpus, allow_research=not args.static),
                  "source_prepare_seconds": source_prepare_seconds, "retrieval_metadata": retrieval_metadata,
                  "retrieval_metrics_before_run": metrics_before, "retrieval_metrics": corpus.retrieval_metrics,
                  "execution_policy": {"reuse_prompt_cache": config.reuse_prompt_cache,
                                       "finish_reserve_seconds": config.finish_reserve_seconds}}
        if args.trace is not None:
            write_fresh_json(args.trace, {"format": "research_local/1", "input": text,
                "input_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                "corpus": identity, "server": server, "result": result})
        output = public_result(result)
        print(json.dumps(output, ensure_ascii=False, allow_nan=False) if args.json else render_text(output))
        return 0 if result.get("status") == "complete" else 1
    except (ValueError, OSError, RuntimeError) as exc:
        output = {"status": "error", "errors": [{"kind": type(exc).__name__, "message": str(exc)}]}
        print(json.dumps(output, ensure_ascii=False, allow_nan=False) if args.json else render_text(output))
        return 1
    finally:
        if corpus is not None:
            corpus.close()


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
