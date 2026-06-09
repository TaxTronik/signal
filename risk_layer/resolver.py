"""Definitionsauflösung /LF150/ /LF160/ /NF30/.

Tragende Regel: Die angezeigte Definition stammt AUSSCHLIESSLICH aus einer
hinterlegten Quelle (Katalog bzw. – Ausbaustufe – Verweisgraph), niemals aus
Modellgewichten. Existiert keine Quelle, wird KEINE Definition erzeugt, sondern
eine Lücke (Rechercheauftrag) ausgegeben.

Diese Trennung ist sicherheitskritisch (/AB40/): Selbst mit leerem Katalog darf
nie eine Definition entstehen – jede erkannte Position wird dann zur Lücke.
"""

from __future__ import annotations

from .catalog import Katalog, KatalogFehler
from .models import Begriffskarte, Status, Treffer
from .zitate import parse_normzitat

RECHERCHE_HINWEIS = "Definition nicht hinterlegt – Rechercheauftrag"


def resolve(treffer: Treffer, katalog: Katalog) -> Begriffskarte:
    """Löst einen Treffer zu einer Begriffskarte auf.

    Greift NUR auf den Katalog zu. Bei fehlender Definitionsquelle ⇒ Status.LUECKE.

    Raises:
        KatalogFehler: wenn die Begriffs-ID des Treffers nicht im Katalog
            existiert (Treffer aus fremder/abweichender Katalogversion) –
            statt eines nackten KeyError ohne Kontext.
    """
    try:
        eintrag = katalog.get(treffer.begriff_id)
    except KeyError as exc:
        raise KatalogFehler(
            f"Begriffs-ID {treffer.begriff_id!r} nicht im Katalog "
            f"(Version {katalog.version!r}) – Treffer und Katalog passen nicht zusammen."
        ) from exc

    if eintrag.hat_definitionsquelle:
        return Begriffskarte(
            treffer=treffer,
            eintrag=eintrag,
            status=Status.TREFFER,
            quellen=list(eintrag.definitionsquelle),
            rechercheauftrag=None,
        )

    # Keine hinterlegte Quelle ⇒ eigenes Risikosignal (Lücke), KEINE Definition (/LF160/)
    return Begriffskarte(
        treffer=treffer,
        eintrag=eintrag,
        status=Status.LUECKE,
        quellen=[],
        rechercheauftrag=(
            f"{RECHERCHE_HINWEIS}: '{eintrag.begriff_kanonisch}' "
            f"(Norm-Anker: {', '.join(eintrag.norm_anker) or 'offen'})"
        ),
    )


def resolve_alle(treffer: list[Treffer], katalog: Katalog) -> list[Begriffskarte]:
    return [resolve(t, katalog) for t in treffer]


def anreichern_mit_berater(karten: list[Begriffskarte], store, *, nutzer: str | None = None) -> list[Begriffskarte]:
    """Hängt sichtbare Berater-Annotationen an die passenden Katalogkarten."""
    if store is None:
        return karten
    for k in karten:
        for d in store.annotationen(k.eintrag.id, nutzer=nutzer):
            k.berater_notizen.append({
                "autor": d.autor, "scope": d.scope, "review_status": d.review_status,
                "konfidenz": d.konfidenz, "text": d.definition_text,
                "governance_typ": d.governance_typ, "norm_anker": list(d.norm_anker),
                "quellen": [q.as_dict() for q in d.definitionsquelle],
            })
    return karten


def anreichern_mit_graph(karten: list[Begriffskarte], graph, *, am: str | None = None) -> list[Begriffskarte]:
    """Hängt den Gesetzeswortlaut aus dem Normgraph an die Karten (/LF170/).

    Quelle ist ausschließlich der Korpus-Text – niemals das Modell. Bei einer
    Lücke bleibt der Status Lücke (der Wortlaut allein löst den unbestimmten
    Begriff nicht auf); der Gesetzestext dient als belegter Kontext.
    """
    if graph is None:
        return karten
    for k in karten:
        gesehen: set[str] = set()
        for zitat in k.eintrag.norm_anker:
            for anker in parse_normzitat(zitat):
                if not anker.norm_id:
                    continue
                nid = anker.norm_id.to_id()
                res = graph.aufloesen(nid, am=am)
                if res.get("gefunden") and res["verwendet"] not in gesehen:
                    gesehen.add(res["verwendet"])
                    k.gesetzestext.append({
                        "id": nid,
                        "verwendet": res["verwendet"],
                        "titel": res.get("titel", ""),
                        "text": res["text"],
                        "law": res.get("law", ""),
                        "gueltig_ab": res.get("gueltig_ab"),
                    })
    return karten
