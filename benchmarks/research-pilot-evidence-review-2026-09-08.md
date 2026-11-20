# Recherchepilot: Belege und Begründungen am 8. September 2026

Dieser Kurzreview betrifft alle zehn ausgegebenen Begründungen des abgeschlossenen
R3-Piloten (`quality-research-moe-pilot-20260908-r3`, V5 mit semantischen
Kataloghinweisen). Die acht Fälle waren bereits bekannte Entwicklungsfälle.
Der Review enthält keine Originalfälle oder internen Modellüberlegungen und
ersetzt keine vollständige fachliche Adjudikation der Normauswahl.

## Treffer und mechanische Herkunft

Alle acht Fälle wurden ohne technischen Fehler abgeschlossen. Vier von vier
positiven Fällen erreichten mindestens eine historische Zielnorm; keiner der
vier negativen Fälle erhielt eine Ausgabe. Sechs der zehn ausgegebenen
Fall-Norm-Paare entsprechen historischen Zielen. Die übrigen vier Vorschläge
in positiven Fällen sind noch nicht adjudiziert und werden hier nicht pauschal
als falsch bewertet. Daraus folgt keine Aussage von 100 % Vorschlagspräzision.

Der unabhängige Herkunftsaudit prüfte 57 zurückgelieferte Seitenbeobachtungen,
236 Normzitatsegmente und acht Sachverhaltsegmente ohne Abweichung von den
gebundenen Originaltexten. Auch die 32 Katalogkarten und acht privaten
Suchvektoren wurden aus YAML, Ankerindex und Suchjournal nachgerechnet.
Diese Prüfungen bestätigen Herkunft und Positionen. Sie prüfen nicht, ob eine
ausgewählte Stelle die zugehörige Begründung tatsächlich trägt.

## Qualitative Einzelbefunde

Die Offsets bezeichnen nullbasierte Originalpositionen; das Ende ist exklusiv.
Ein Satzfragment ist von einer sachlich falschen Belegauswahl zu unterscheiden.

| Fall-ID | Norm und ausgewählter Bereich | Belegpassung und Begründung |
|---|---|---|
| `ao_v1:1` | § 7 AStG, 215–425 | Satzfragment: Die ausgewählte Stelle endet bei „bei dem“. Die behauptete Zurechnungsfolge ist darin nicht vollständig lesbar. |
| `ao_v1:1` | § 8 AStG, 6203–6419 | **Unpassender Beleg:** Der Ausschnitt betrifft Umwandlung und Buchwerte; die Begründung betrifft die Ausnahme einer wesentlichen wirtschaftlichen Tätigkeit. |
| `ao_v1:1` | § 42 AO, 434–650 | Thematisch passendes, aber beschnittenes Satzfragment. Die Begründung verbindet Rechtsfolge und Missbrauchsvoraussetzungen, die im ausgewählten Ausschnitt nur teilweise enthalten sind. |
| `ao_v1:1` | § 39 AO, 218–438 | Die Stelle regelt die Zurechnung eines Wirtschaftsguts. Die Begründung erweitert diese Aussage ohne Beleg im gewählten Ausschnitt auf „Erträge“. |
| `ao_v1:2` | § 42 AO, 650–884 | Die Ausnahme außersteuerlicher Gründe wird passend zitiert. Die Begründung behandelt ihr Fehlen bereits als begründeten Missbrauchstatbestand; weitere Voraussetzungen der Norm sind damit nicht festgestellt. |
| `embedding_eval:positive:01` | § 146 AO, 0–214 | Direkter Beleg für die genannte Pflicht zur täglichen Festhaltung von Kasseneinnahmen und Kassenausgaben. |
| `embedding_eval:positive:01` | § 158 AO, 0–215 | Der Zusammenhang mit §§ 140–148 ist belegt. Der behauptete pauschale Wegfall der zugrunde zu legenden Beweiskraft ist weiter als die normseitige, auf „soweit“ und den Einzelfall begrenzte Ausnahme. |
| `embedding_eval:positive:01` | § 162 AO, 438–658 | Der Satz ist vor dem vollständigen §-158-Verweis abgeschnitten. Zudem nennt die Begründung einen „Ermessensspielraum“, während der gebundene Normtext bei erfüllten Voraussetzungen „hat … zu schätzen“ beziehungsweise „Zu schätzen ist“ vorgibt. |
| `embedding_eval:positive:02` | § 146 AO, 0–214 | Direkter Beleg für vollständige, geordnete Aufzeichnungen und die tägliche Festhaltung der genannten Kassenbewegungen. |
| `embedding_eval:positive:02` | § 158 AO, 0–215 | Passender Regelungszusammenhang; die Begründung formuliert den Wegfall erneut pauschal und übernimmt die Einschränkung „soweit“ nicht. |

## Ursache der falschen AStG8-Belegauswahl

Der originale Abschluss-Toolcall wählte selbst `quote_id=e6q2` für
`norm:AStG:8`. Der Host übernahm genau diesen Ausschnitt, 6203–6419.
Es gab keine Aliasverwechslung oder Verschiebung der Originalpositionen.
Die zur Begründung gehörenden Abschnitte waren in allen vier tatsächlichen
Modellanfragen unter `e6q3` (6419–6631) und `e6q4` (6631–6776) verfügbar.
Der Fehler entstand somit bei der Belegauswahl des Modells, obwohl die
passendere Quelle zugänglich war.

## Bedeutung für eine mögliche Freigabe

Der Pilot zeigt einen Treffergewinn gegenüber R2, aber zugleich konkrete
Fehler in Belegauswahl und Rechtsfolgenbeschreibung. `source_verified=true`
darf deshalb nur als Herkunftsbestätigung verstanden werden. Eine Freigabe
als verlässlich begründete oder abschließende rechtliche Bewertung lässt sich
aus diesem Pilot allein nicht ableiten. Belegpassung und die Reichweite der
Begründungen benötigen eigene Abnahmekriterien neben Ziel-ID-Treffern.
R3 änderte Prompt und Kataloganschluss gemeinsam; der Vergleich isoliert
keinen einzelnen Änderungseffekt.

Rohbericht-SHA256:
`6ebe68065c8e5912e0f44641b25929d9b93a4941d507c3f5d348b03a2b61a89b`

AStG8-Abschlussrequest-SHA256:
`b8cf294ac644bc455a676beadd02756e21ef875f017ebed03c79312ba626d4c7`
