"""Kommandozeilen-Einstieg des Risk-Layer (Stufe 0 / MVP).

Beispiele:
    risk-layer analyse sachverhalt.txt
    risk-layer analyse sachverhalt.txt --html out/report.html
    cat text.txt | risk-layer analyse -
    risk-layer katalog --check
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .berater import BeraterFehler, BeraterStore
from .catalog import KatalogFehler, lade_katalog
from .graph import graph_oeffnen
from .pipeline import Pipeline
from .report import report_html, report_text
from .zitate import parse_entscheidung, parse_normzitat


def _lies_text(quelle: str) -> str:
    if quelle == "-":
        return sys.stdin.read()
    return Path(quelle).read_text(encoding="utf-8")


def _cmd_analyse(args: argparse.Namespace) -> int:
    try:
        katalog = lade_katalog(args.katalog)
    except KatalogFehler as exc:
        print(f"Katalogfehler: {exc}", file=sys.stderr)
        return 2

    berater = None
    if args.berater:
        try:
            berater = BeraterStore(args.berater)
        except BeraterFehler as exc:
            print(f"Berater-Store-Fehler: {exc}", file=sys.stderr)
            return 2

    graph = graph_oeffnen(args.graph) if args.graph else graph_oeffnen()
    from .triggerindex import index_oeffnen
    trigger = index_oeffnen()
    provider = None
    if args.schicht2:
        from .llm import LlamaServerProvider
        provider = LlamaServerProvider(args.llm_url or "http://127.0.0.1:8080")

    text = _lies_text(args.eingabe)
    report = Pipeline(katalog).run(
        text,
        erzeugt_am=args.zeitstempel,
        unknown=args.unknown,
        berater=berater,
        nutzer=args.nutzer,
        graph=graph,
        provider=provider,
        trigger=trigger,
    )

    # Konsolen-Zusammenfassung
    print(report_text(report))

    if args.html:
        html_pfad = Path(args.html)
        html_pfad.parent.mkdir(parents=True, exist_ok=True)
        html_pfad.write_text(
            report_html(report, text, titel=f"Risk-Layer · {Path(args.eingabe).name}"),
            encoding="utf-8",
        )
        print(f"\n→ HTML-Report: {html_pfad}", file=sys.stderr)

    # Exit-Code signalisiert offene Lücken (nützlich für n8n/CI-Eskalation, /Z80/)
    if report.luecken and args.fail_on_luecke:
        return 3
    return 0


def _cmd_katalog(args: argparse.Namespace) -> int:
    try:
        katalog = lade_katalog(args.katalog)
    except KatalogFehler as exc:
        print(f"Katalogfehler: {exc}", file=sys.stderr)
        return 2
    ohne_quelle = [e.id for e in katalog if not e.hat_definitionsquelle]
    print(f"Katalogversion : {katalog.version}")
    print(f"Begriffe       : {len(katalog)}")
    print(f"ohne Quelle    : {len(ohne_quelle)} {ohne_quelle if ohne_quelle else ''}")
    if args.check:
        print("OK – Katalog strukturell valide.")
    return 0


def _cmd_korpus(args: argparse.Namespace) -> int:
    from . import corpus
    katalog = lade_katalog(args.katalog)
    if args.aktion == "init":
        gesetze = corpus.steuer_korpus() if args.voll else corpus.ableiten_aus_katalog(katalog)
        pfad = corpus.schreibe_config(gesetze, args.dir)
        print(f"Config geschrieben: {pfad}")
        for g in gesetze:
            print(f"  {g.law:8} -> {g.url}")
        return 0
    # fetch
    gesetze = corpus.lade_config(args.dir) or corpus.ableiten_aus_katalog(katalog)
    if not corpus.lade_config(args.dir):
        corpus.schreibe_config(gesetze, args.dir)
    print(f"Lade {len(gesetze)} Gesetze von gesetze-im-internet …")
    erg = corpus.hole_alle(gesetze, args.dir)
    print(f"  geholt: {erg.geholt}")
    if erg.fehler:
        print(f"  Fehler: {erg.fehler}", file=sys.stderr)
    return 0 if erg.geholt else 1


def _cmd_graph(args: argparse.Namespace) -> int:
    from . import gii, graph as graphmod
    if args.aktion == "build":
        knoten = gii.parse_verzeichnis(graphmod.Path(args.dir) / "gii")
        if not knoten:
            print(f"Keine XML in {args.dir}/gii – erst 'korpus fetch' ausführen.", file=sys.stderr)
            return 1
        g = graphmod.NormGraph.build(knoten, args.db)
        print(f"Graph gebaut: {args.db}")
        print(f"  {g.stats()}")
        return 0
    g = graphmod.graph_oeffnen(args.db)
    if g is None:
        print(f"Kein Graph unter {args.db} – erst 'graph build'.", file=sys.stderr)
        return 1
    if args.aktion == "stats":
        print(g.stats())
    elif args.aktion == "dangling":
        for d in g.dangling(args.limit):
            print(f"  {d['src']}  →  {d['dst']}")
    elif args.aktion == "aufloesen":
        r = g.aufloesen(args.id)
        if not r.get("gefunden"):
            print(f"nicht gefunden: {args.id}"); return 1
        print(f"{r['verwendet']}  ({r.get('titel','')})")
        print(r["text"][:600])
        print("verweist_auf:", r["verweist_auf"][:10])
    return 0


def _cmd_modell(args: argparse.Namespace) -> int:
    from . import modell as M

    if args.aktion == "engine":
        backend = getattr(args, "backend", "auto")
        muster = M.backend_patterns(backend)
        print(f"Lade llama.cpp-Engine (Backend: {backend} → {muster[0]}) nach ./engine …")
        letzte = [-1]

        def fortschritt(gelesen: int, gesamt: int) -> None:
            pct = int(gelesen * 100 / gesamt) if gesamt else 0
            if pct != letzte[0]:
                letzte[0] = pct
                print(f"\r  {pct:3d}%  {M.menschlich(gelesen)} / {M.menschlich(gesamt)}",
                      end="", flush=True)

        try:
            binp = M.download_engine(backend=backend, fortschritt=fortschritt)
        except Exception as exc:  # noqa: BLE001
            print(f"\nFehler: {exc}", file=sys.stderr)
            return 1
        print(f"\nEngine bereit: {binp or 'Binary nicht gefunden – Archiv prüfen'}")
        return 0 if binp else 1

    if args.aktion == "list":
        print("Verfügbare Modelle (Default: %s):" % M.DEFAULT_MODELL)
        for key, info in M.MODELLE.items():
            mark = "*" if key == M.DEFAULT_MODELL else " "
            print(f" {mark} {key:12} {info.name:14} {info.quant:7} ~{info.groesse_gb:.1f} GB  {info.notiz}")
        return 0

    info = M.MODELLE.get(args.modell or M.DEFAULT_MODELL)
    if info is None:
        print(f"unbekanntes Modell: {args.modell}", file=sys.stderr)
        return 2

    if args.aktion == "download":
        ziel = M.DEFAULT_MODELL_DIR
        print(f"Lade {info.name} ({info.quant}) aus {info.repo} → {ziel}/ …")
        letzte = [-1]

        def fortschritt(gelesen: int, gesamt: int) -> None:
            pct = int(gelesen * 100 / gesamt) if gesamt else 0
            if pct != letzte[0]:
                letzte[0] = pct
                print(f"\r  {pct:3d}%  {M.menschlich(gelesen)} / {M.menschlich(gesamt)}",
                      end="", flush=True)

        pfade = M.download_modell(info, ziel, fortschritt=fortschritt)
        print("\nFertig:", ", ".join(str(p) for p in pfade))
        return 0

    if args.aktion == "status":
        pfad = M.modell_pfad(info)
        binary = M.finde_llama_server(args.llm_bin)
        print(f"Modell        : {info.name} {info.quant}")
        print(f"  Datei       : {pfad or 'noch nicht geladen (risk-layer modell download)'}")
        print(f"  llama-server: {binary or 'nicht gefunden (llama.cpp-Vulkan-Release auf PATH legen)'}")
        from .llm import server_verfuegbar
        print(f"  läuft       : {'ja' if server_verfuegbar(args.llm_url or 'http://127.0.0.1:8080') else 'nein'}")
        return 0

    if args.aktion == "start":
        pfad = M.modell_pfad(info)
        if pfad is None:
            print("Modell nicht geladen. Erst: risk-layer modell download", file=sys.stderr)
            return 1
        mgr = M.LlamaServerManager(pfad, binary=args.llm_bin)
        try:
            mgr.start()
        except FileNotFoundError as exc:
            print(str(exc), file=sys.stderr)
            return 1
        print(f"llama-server gestartet: {' '.join(mgr.kommando())}")
        print("Warte auf Bereitschaft …")
        ok = mgr.warte_bereit()
        print("bereit ✓" if ok else "nicht erreichbar geworden")
        print("(Strg+C beendet den Server.)")
        try:
            mgr._proc.wait()  # type: ignore[union-attr]
        except KeyboardInterrupt:
            mgr.stop()
            print("\nbeendet.")
        return 0

    return 2


def _cmd_definiere(args: argparse.Namespace) -> int:
    from datetime import date

    try:
        store = BeraterStore(args.berater)
        d = store.definiere(
            args.begriff,
            args.text,
            autor=args.autor,
            datum=args.datum or date.today().isoformat(),
            scope=args.scope,
            governance_typ=args.governance,
            schadensintensitaet=args.intensitaet,
            norm_anker=tuple(args.norm or ()),
            konfidenz=args.konfidenz,
        )
    except BeraterFehler as exc:
        print(f"Berater-Fehler: {exc}", file=sys.stderr)
        return 2
    print(f"definiert: {d.id} (scope={d.scope}, autor={d.autor})")
    if d.geankert_an:
        print(f"  geankert an: {', '.join(d.geankert_an)}")
    return 0


def _cmd_trigger(args: argparse.Namespace) -> int:
    from .graph import graph_oeffnen
    from .triggerindex import TriggerIndex, index_oeffnen
    if args.aktion == "build":
        g = graph_oeffnen(args.graph_db)
        if g is None:
            print("Kein Normgraph – erst 'graph build'.", file=sys.stderr)
            return 1
        ti = TriggerIndex.build(g, args.db)
        print(f"Trigger-Index gebaut: {args.db}\n  {ti.stats()}")
        return 0
    ti = index_oeffnen(args.db)
    if ti is None:
        print(f"Kein Trigger-Index unter {args.db} – erst 'trigger build'.", file=sys.stderr)
        return 1
    if args.aktion == "stats":
        print(ti.stats())
    elif args.aktion == "lookup":
        for c in ti.kandidaten(" ".join(args.text or []), k=args.limit):
            mark = "✓" if c["geprueft"] else "·"
            print(f" {mark} {c['zitat']:14} score={c['score']:6}  {c['titel'][:45]:45} "
                  f"via {', '.join(c['trigger'])}")
    return 0


def _cmd_embedding(args: argparse.Namespace) -> int:
    from .graph import graph_oeffnen
    from .embedding import EmbeddingIndex, encoder_oeffnen, index_oeffnen
    if args.aktion == "build":
        g = graph_oeffnen(args.graph_db)
        if g is None:
            print("Kein Normgraph – erst 'graph build'.", file=sys.stderr)
            return 1
        import os
        dev = os.environ.get("RISK_LAYER_EMB_DEVICE", "cuda")
        enc = encoder_oeffnen(args.encoder, device=dev)
        kat = lade_katalog(getattr(args, "katalog", None))
        print(f"Baue Embedding-Index mit Encoder '{enc.name}' (+ Katalog-Paraphrasen) ...")
        idx = EmbeddingIndex.build(g, enc, katalog=kat)
        idx.save(args.dir)
        anker = getattr(idx, "n_paraphrase_anker", 0)
        print(f"Embedding-Index gebaut: {args.dir} | {len(idx.ids)} Vektoren "
              f"(davon {anker} Paraphrasen-Anker), dim {idx.dim}")
        # TOTE ANKER laut melden: Paraphrasen ohne Index-Ziel dämpfen den Recall
        # unsichtbar (z. B. Granularitäts-Mismatch Katalog/Index). Schreien, nicht schlucken.
        tot = getattr(idx, "verlorene_anker", [])
        if tot:
            print(f"\n  ⚠ {len(tot)} TOTE ANKER (Paraphrasen ohne Index-Ziel):", file=sys.stderr)
            for v in tot:
                print(f"     - {v['begriff']}: Ziel '{v['ziel_zitat']}' nicht im Index "
                      f"({v['n_paraphrasen']} Paraphrasen verloren)", file=sys.stderr)
        return 0
    idx = index_oeffnen(args.dir)
    if idx is None:
        print(f"Kein Embedding-Index unter {args.dir} – erst 'embedding build'.", file=sys.stderr)
        return 1
    if args.aktion == "stats":
        print(f"{len(idx.ids)} Normen, dim {idx.dim}, Encoder {idx.encoder_name}")
    elif args.aktion == "suche":
        enc = encoder_oeffnen(idx.encoder_name if idx.encoder_name != "bge-m3" else "bge-m3")
        for h in idx.suche(" ".join(args.text or []), enc, k=args.limit, min_sim=0.0):
            print(f"  {h['zitat']:14} sim={h['sim']:6}  {h['titel'][:50]}")
    return 0


def _cmd_abgleich(args: argparse.Namespace) -> int:
    from . import abgleich as A
    katalog = lade_katalog(args.katalog)
    berater = BeraterStore(args.berater) if args.berater else None
    treffer = A.laufe(katalog, berater=berater, fetch=not args.no_fetch,
                      nur_neue=not args.alle)
    if not treffer:
        print("Keine neuen BFH/BMF-Treffer an getrackten Begriffen.")
        return 0
    for t in treffer:
        print(f"[{t.eintrag.quelle}] {t.eintrag.titel}")
        print(f"   {t.eintrag.link}")
        for b in t.betroffene:
            print(f"   → {b['begriff']}  ({b['quelle']}, via {b['via']})")
    return 0


def _cmd_zitat(args: argparse.Namespace) -> int:
    text = " ".join(args.zitat)
    if args.urteil:
        uid = parse_entscheidung(text)
        if uid is None:
            print("kein Entscheidungszitat erkannt", file=sys.stderr)
            return 1
        print(uid.to_id())
        for a in uid.aliase:
            print(f"  alias: {a}")
        return 0
    anker = parse_normzitat(text)
    for a in anker:
        d = a.as_dict()
        flags = []
        if d["relativ"]:
            flags.append("relativ")
        if d["offener_bereich"]:
            flags.append("ff.")
        print(f"{d['id'] or '(relativ – nicht aufgelöst)':40} "
              f"konfidenz={d['konfidenz']} {' '.join(flags)}".rstrip())
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="risk-layer",
        description="Schlüsselbegriff-Assistenz – deterministische Schicht-1-Auswertung.",
    )
    p.add_argument("--katalog", help="Pfad zur Katalogdatei (Default: catalog/begriffe.yaml)")
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("analyse", help="Text analysieren und Report erzeugen")
    a.add_argument("eingabe", help="Textdatei oder '-' für stdin")
    a.add_argument("--html", help="Pfad für annotierten HTML-Report")
    a.add_argument("--zeitstempel", help="Fester ISO-Zeitstempel (für reproduzierbare Läufe)")
    a.add_argument(
        "--unknown",
        action="store_true",
        help="Unknown-als-Risiko: juristisch aufgeladene Unbekannte eskalieren (offline)",
    )
    a.add_argument("--berater", help="Verzeichnis mit Berater-Definitionen (personal/geteilt)")
    a.add_argument("--nutzer", help="Nutzername (Sichtbarkeit persönlicher Definitionen)")
    a.add_argument("--graph", help="Pfad zur Normgraph-SQLite (Default: corpus/graph.sqlite)")
    a.add_argument("--schicht2", action=argparse.BooleanOptionalAction, default=True,
                   help="LLM-Kandidaten (lokaler llama-server); --no-schicht2 deaktiviert")
    a.add_argument("--llm-url", help="llama-server URL (Default http://127.0.0.1:8080)")
    a.add_argument(
        "--fail-on-luecke",
        action="store_true",
        help="Exit-Code 3, wenn offene Lücken vorhanden sind (Eskalation)",
    )
    a.set_defaults(func=_cmd_analyse)

    k = sub.add_parser("katalog", help="Katalog prüfen/anzeigen")
    k.add_argument("--check", action="store_true", help="strukturelle Validierung")
    k.set_defaults(func=_cmd_katalog)

    tg = sub.add_parser("trigger", help="Maschinellen Trigger-Index (Norm-Findungs-Stütze) bauen/abfragen")
    tg.add_argument("aktion", choices=["build", "stats", "lookup"])
    tg.add_argument("text", nargs="*", help="Text für 'lookup'")
    tg.add_argument("--db", default="corpus/trigger.sqlite", help="Trigger-Index-SQLite")
    tg.add_argument("--graph-db", default="corpus/graph.sqlite", help="Normgraph (für build)")
    tg.add_argument("--limit", type=int, default=8)
    tg.set_defaults(func=_cmd_trigger)

    em = sub.add_parser("embedding", help="Semantischen Embedding-Index (Paraphrasen) bauen/abfragen")
    em.add_argument("aktion", choices=["build", "stats", "suche"])
    em.add_argument("text", nargs="*", help="Text für 'suche'")
    em.add_argument("--dir", default="corpus/embedding", help="Index-Verzeichnis")
    em.add_argument("--graph-db", default="corpus/graph.sqlite", help="Normgraph (für build)")
    em.add_argument("--encoder", default="auto", choices=["auto", "bge-m3", "hash"],
                    help="auto: BGE-M3 falls installiert, sonst Hash")
    em.add_argument("--limit", type=int, default=8)
    em.set_defaults(func=_cmd_embedding)

    ab = sub.add_parser("abgleich", help="Neue BFH/BMF-Veröffentlichungen gegen den Katalog prüfen")
    ab.add_argument("--berater", help="Berater-Definitionen einbeziehen")
    ab.add_argument("--no-fetch", action="store_true", help="nicht ins Netz (nur Zustand)")
    ab.add_argument("--alle", action="store_true", help="auch bereits gesehene Einträge")
    ab.set_defaults(func=_cmd_abgleich)

    z = sub.add_parser("zitat", help="Norm-/Entscheidungszitat → kanonische ID")
    z.add_argument("zitat", nargs="+", help="z. B. '§ 2 Abs. 2 Nr. 2 UStG'")
    z.add_argument("--urteil", action="store_true", help="als Entscheidungszitat parsen")
    z.set_defaults(func=_cmd_zitat)

    ko = sub.add_parser("korpus", help="Normtext-Korpus verwalten (gesetze-im-internet)")
    ko.add_argument("aktion", choices=["init", "fetch"], help="init=Config, fetch=Download")
    ko.add_argument("--voll", action="store_true", help="breiter Steuer-Korpus statt nur Katalog-Gesetze")
    ko.add_argument("--dir", default="corpus", help="Korpus-Verzeichnis")
    ko.set_defaults(func=_cmd_korpus)

    gr = sub.add_parser("graph", help="Normgraph bauen/abfragen")
    gr.add_argument("aktion", choices=["build", "stats", "dangling", "aufloesen"])
    gr.add_argument("id", nargs="?", help="Norm-ID für 'aufloesen'")
    gr.add_argument("--dir", default="corpus", help="Korpus-Verzeichnis (für build)")
    gr.add_argument("--db", default="corpus/graph.sqlite", help="SQLite-Pfad")
    gr.add_argument("--limit", type=int, default=30, help="max. Dangling-Einträge")
    gr.set_defaults(func=_cmd_graph)

    m = sub.add_parser("modell", help="LLM-Modell (GGUF) verwalten und llama-server starten")
    m.add_argument("aktion", choices=["list", "download", "engine", "status", "start"])
    m.add_argument("--modell", help="Modell-Key (Default: qwen3-14b)")
    m.add_argument("--llm-bin", help="Pfad zum llama-server-Binary")
    m.add_argument("--llm-url", help="llama-server URL (Default http://127.0.0.1:8080)")
    m.add_argument("--backend", choices=["auto", "cuda", "hip", "vulkan", "cpu"],
                   default="auto",
                   help="GPU-Backend für 'engine' (auto: NVIDIA→cuda, AMD→hip, sonst Vulkan/CPU)")
    m.set_defaults(func=_cmd_modell)

    d = sub.add_parser("definiere", help="Berater-Definition anlegen (define-on-the-fly)")
    d.add_argument("begriff", help="der zu definierende Begriff")
    d.add_argument("--text", required=True, help="Definitionstext (eigene Worte)")
    d.add_argument("--berater", required=True, help="Zielverzeichnis des Berater-Stores")
    d.add_argument("--autor", required=True, help="Autor (Nutzername)")
    d.add_argument("--scope", choices=["personal", "geteilt"], default="personal")
    d.add_argument("--norm", action="append", help="Norm-Anker (mehrfach möglich)")
    d.add_argument("--governance", choices=["FP", "FF", "IN"], default="IN")
    d.add_argument("--intensitaet", choices=["niedrig", "mittel", "hoch"], default="mittel")
    d.add_argument("--konfidenz", choices=["niedrig", "mittel", "hoch"], default="mittel")
    d.add_argument("--datum", help="Eintragsdatum (Default: heute)")
    d.set_defaults(func=_cmd_definiere)

    return p


def _utf8_konsole() -> None:
    """Stellt UTF-8 für Konsolenausgabe sicher (Windows-Codepages)."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
        except (AttributeError, ValueError):
            pass


def main(argv: list[str] | None = None) -> int:
    _utf8_konsole()
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
