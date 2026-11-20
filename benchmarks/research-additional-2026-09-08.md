# Lokale Recherche: vollständiger Additional-Block vom 8. September 2026

Der Rechercheagent erreicht in diesem Lauf alle 15 historisch positiven Fälle,
während das öffentliche Profil `treffer` 13 erreicht hatte. Zugleich erhält
nur einer von 15 historisch negativen Fällen eine Ausgabe, gegenüber sieben
in der Baseline. Die Abdeckung einzelner hinterlegter Zielnormen sinkt jedoch
von 18 auf 17. Alle 30 Fälle wurden ohne technischen Fehler abgeschlossen.

## Vergleich und Auswahl

| Beobachtung | Gespeicherte GPU-Baseline `treffer` | Rechercheagent |
|---|---:|---:|
| Positive Fälle mit mindestens einer historischen Zielnorm | 13/15 | 15/15 |
| Erreichte historische Fall-Zielnorm-Paare | 18/27 | 17/27 |
| Negative Fälle mit Ausgabe | 7/15 | 1/15 |
| Ausgegebene Normpaare in negativen Fällen | 14 | 1 |
| Vorschläge insgesamt | 63 | 22 |
| Zusätzliche, unadjudizierte Normpaare in positiven Fällen | 31 | 4 |

Der vorab festgelegte Plan umfasst den gesamten bestehenden Additional-Block
in unveränderter Reihenfolge: 15 positive und 15 negative Fälle. Vier davon
waren bereits im acht Fälle umfassenden Recherchepilot enthalten; 26 kommen
gegenüber diesem Pilot hinzu. Die Auswahl erfolgte nicht anhand der neuen
Rechercheergebnisse. Sämtliche Fälle waren aber schon bekannte Entwicklungsdaten:
Dies ist ein vollständiger Teilblock, keine unabhängige Prüfung aller 162 Fälle.

Die positiven Zielmengen sind unvollständig und können verwandte sinnvolle
Normen enthalten. Ihre Abdeckung ist deskriptiv; nicht jede hinterlegte Norm
muss im Einzelfall ausgegeben werden. Zusätzliche Normen sind ohne gesonderte
fachliche Prüfung weder richtig noch falsch adjudiziert. Deshalb lässt sich
aus der Tabelle keine Präzision einzelner Vorschläge ableiten. Die historischen
Negativlabels bleiben unverändert, einschließlich eines bereits vorab als
produktabhängig mehrdeutig erkannten Falls.

## Verfahren und Aufwand

Verwendet wurde Qwen3.6-35B-A3B in Q4_K_M mit Prompt V5, nativen Werkzeugaufrufen,
CPU-Suche aus BM25 und BGE-M3 sowie bis zu vier unverifizierten Kataloghinweisen.
Konfiguration, Modell und Runtime entsprechen dem abgeschlossenen R3-Pilot.
Jeder Fall beginnt mit einer frischen Sitzung und sechs Originalquellen.
Historische Labels und Baselinevorschläge gelangen ausschließlich zum Evaluator.

Feste Grenzen: 16.384 Kontexttokens, 2.048 Ausgabetokens je Modellaufruf,
8.192 insgesamt, 768 Thinking-Budgettokens, zehn Runden, acht Aufrufe von Recherchewerkzeugen
und höchstens fünf verschiedene Normen je Originalsatz. Das Zeitlimit beträgt
600 Sekunden, die konfigurierte Abschlussreserve 180 Sekunden. Promptcache
wird erst nach dem kalten ersten Aufruf innerhalb desselben Falls angefordert.

Die gemessene Falllaufzeit beträgt im Median **81,206 Sekunden**, am
95. Perzentil **138,494 Sekunden**, maximal **147,688 Sekunden**.
Alle Fallzeiten summieren sich auf rund **40,8 Minuten**. Korpusvorbereitung
(4,485 s), Encoderladen und Aufwärmen (zusammen 13,522 s) sowie der zuvor
erfolgte Sprachmodellstart liegen außerhalb dieser Fallzeiten. Dies ist die
gemessene Ausführung dieses Prüfprotokolls, keine allgemeine Serverlatenz.

Es gab 58 Modellaufrufe, 19 Quellenlese- und neun weitere Suchaufrufe,
keine protokollierten Reparaturfehler und ein Ereignis zur Kontextreserve.
Eine zeitbedingte Abschlussreserve wurde nicht ausgelöst. Dieser Vergleich
isoliert weder den Katalogeffekt noch einen Cachegeschwindigkeitsgewinn.

## Herkunft und Belegqualität

Der unabhängige Audit bestätigt alle 30 Journaleinträge, die eingefrorenen
Quellen-, Modell- und Encoderdateien sowie das erneut berechnete Aggregat.
240 zurückgelieferte Seitenbeobachtungen, 922 Normzitatsegmente und
30 Sachverhaltsegmente stimmen exakt mit den Originalen überein. Die
120 Katalogkarten wurden aus YAML, vorhandenen Ankern und 30 gebundenen
Suchvektoren reproduziert; 117 Hinweis-Payloads stimmen damit überein.
Mechanische Herkunftsfehler wurden nicht gefunden.

Die separate Prüfung aller **22 Begründungen** zeigt dennoch Einschränkungen:

- **Ungelesener Regelanfang:** Bei § 8 AStG fehlte der Beginn einer Ausnahme
  in den tatsächlich gelesenen Seiten. Der gewählte Personalausschnitt war echt,
  die erklärte Wirkung aber irreführend. Anders als im
  [Pilotreview](research-pilot-evidence-review-2026-09-08.md) war die entscheidende
  Gegenregel hier noch nicht sichtbar. Eine unvollständige 400-Zeichen-Begründung
  stammte bereits vom Modell; der Host hatte sie nicht gekürzt.
- **Unvollständige Belegwahl:** Unter anderem bei §§ 34 und 173a AO,
  § 2 UStG sowie § 30 HGB wurde ein Satzteil gewählt, obwohl weitere
  notwendige Teile bereits als Nachbarsegmente verfügbar waren. Die Hostzuordnung
  war korrekt; der Einzelbeleg deckte die Begründung nicht vollständig.
- **Zu bestimmte Einordnung:** Manche Begründungen ergänzen einen nicht
  festgestellten steuerlichen Grundkontext oder behandeln einen Anknüpfungspunkt
  schon als erfüllten Tatbestand. Offene Bedingungen sind von bewiesenen
  Tatsachen zu unterscheiden. Bedingt erklärte Zusatznormen werden dabei
  nicht pauschal als falsch bewertet.

Der Block ist stark von AO-Themen und katalognahen Normthemen geprägt.
Kataloghinweise sind Suchhilfen, keine Originalbelege oder Bestätigungen.
Die beobachteten Gewinne begründen daher keine automatische Ablösung des
Produktionsprofils und keinen Anspruch nahezu perfekter fachlicher Treffer.
Auch identische Requests mit festem Seed hatten in Vorläufen unterschiedliche
Modellausgaben; ein einzelner Lauf belegt keine bitgenaue Wiederholbarkeit.

## Nachweisbindung

Die privaten Roharchive enthalten Originaltexte und Modellverläufe und werden
hier nicht veröffentlicht. Dieser Bericht verwendet folgende SHA-256-Bindungen:

| Artefakt | SHA-256 |
|---|---|
| Vorabauswahlplan | `53f40fa2579623816601ef694a5bfd437388ed6e88efe1ad31ca65fe707af5ce` |
| Abgeschlossener Rohbericht | `33854a40563c27097c688390aa579ed3d5f4dfba21a4e426e6f6db52a5f89cdc` |
| Vor Laufbeginn eingefrorenes Protokoll | `1a3a590f20d4710401b2d138d6390244e7979870f8bad02aecac30d04ee33a85` |
| Geprüftes Aggregat | `4a8e8a5290c3419c08399b2933629f696795714c53cc97aabb436dbdcdbe10b2` |
| Unabhängiger Provenienz-/Zählaudit | `316e54717cfb7372207bb938117ce8dcc60401496b3e4da0b7c546c6839dec0b` |
