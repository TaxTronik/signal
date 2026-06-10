# Parser- und Knoten-ID-Spezifikation

> **Eingefroren (Stand 31.05.2026):** Historisches Anforderungsdokument. Der Code
> referenziert die hier definierten IDs (/T…/) und Abschnitte („Parser-Spez. 1–4").
> Inhaltliche Änderungen nur über ein neues ADR — der gelebte Stand steht in
> README/FEATURES/INTEGRATION.

**Companion zum Lastenheft „Schlüsselbegriff-Assistenz"**
**Zielformat der Gesetze:** gii (gesetze-im-internet), DTD-basiert
**Stand:** 31.05.2026

Diese Spezifikation beschreibt die eine Komponente, die Katalog, Gesetze, Verwaltung und Urteile zusammenhält: das **kanonische ID-Schema** und den **Zitat-Parser** (Norm- und Entscheidungszitate). Der Parser ist das universelle Join über alle Korpora.

---

## 1. Kanonisches Knoten-ID-Schema

### 1.1 Norm-Knoten

```
norm:<LAW>:<para>[:abs<n>][:nr<n>][:s<n>][:lit<x>]
```

- `<LAW>`: amtliche Abkürzung aus `<amtabk>` (Fallback `<jurabk>` ohne Jahres-Suffix). Beispiel: „AO 1977" ⇒ `AO`.
- `<para>`: §-Nummer inkl. Buchstaben-Suffix, ohne „§ ". Beispiele: `2`, `2a`, `14`.
- Optionale Tiefenkomponenten in **Zitierreihenfolge** angehängt: `abs`, `nr`, `s` (Satz), `lit` (Buchstabe).

**Beispiele:**

| Zitat | Kanonische ID |
|---|---|
| § 2 Abs. 2 Nr. 2 UStG | `norm:UStG:2:abs2:nr2` |
| § 14 Abs. 1 Nr. 3 Satz 4 KStG | `norm:KStG:14:abs1:nr3:s4` |
| § 9 Nr. 1 Satz 2 GewStG | `norm:GewStG:9:nr1:s2` |
| Art. 11 MwStSystRL | `norm:MwStSystRL:art11` |

> **Hinweis zur Reihenfolge:** Deutsche Zitate sind in der Tiefenreihenfolge nicht einheitlich (Satz-dann-Nr vs. Nr-dann-Satz), weil die Containment-Struktur variiert (ein Satz kann Nummern enthalten, eine Nummer eigene Sätze). Die ID bewahrt die **Quell-Reihenfolge des Zitats**; der Resolver gleicht sie gegen den aus dem XML gebauten Knotenbaum ab. Bei `Art.` (AktG, GG, MwStSystRL, EU-Recht) wird `art<n>` statt `<para>` genutzt.

### 1.2 Verwaltungs-Knoten (UStAE, AEAO, Richtlinien)

```
verw:<QUELLE>:<abschnitt>
```
Beispiele: `verw:UStAE:2.8`, `verw:AEAO:zu§153`, `verw:GewStR:R2.3`.

### 1.3 Urteils-Knoten

```
urteil:<GERICHT>:<aktenzeichen-normalisiert>
```
Beispiele: `urteil:BFH:I-R-37/19`, `urteil:EuGH:C-141/20`, `urteil:BVerfG:2-BvL-19/17`.
Aktenzeichen ist der dauerhafte Schlüssel; Fundstellen (BStBl., DStR) werden als **Aliase** geführt, nicht als ID.

### 1.4 Quellentyp als Pflicht-Property jedes Knotens

```
quellentyp ∈ { norm, unionsrecht, rspr_bfh, rspr_eugh, rspr_sonstige, verwaltung }
```
Trägt die Rangfolge aus dem Artikel: Norm/Unionsrecht = Tatbestand, BFH/EuGH = Anerkennungskorridor, Verwaltung = Streit-/Prozessrisiko. **Jede ausgespielte Definition muss mit ihrem Quellentyp gelabelt werden** (Verwaltungsauffassung ≠ Gerichtsentscheidung). Für ein Haftungswerkzeug nicht optional.

---

## 2. Normzitat-Parser

### 2.1 Grammatik (EBNF, vereinfacht)

```
Zitat        := ParaMarker Para (Tiefe)* LawAbbr?
              | "Art." Num (Tiefe)* LawAbbr?
ParaMarker   := "§§" | "§"
Para         := Num                          (* z. B. 2, 2a, 14 *)
Num          := /\d+[a-z]*/
Tiefe        := Absatz | Satz | Nummer | Buchstabe | Halbsatz | Alternative
Absatz       := ("Abs." | "Absatz") Num
Satz         := ("S." | "Satz") /\d+/
Nummer       := ("Nr." | "Nummer") Num
Buchstabe    := ("Buchst." | "Buchstabe" | "lit.") /[a-z]/
Halbsatz     := ("Halbs." | "Halbsatz") /\d+/
Alternative  := ("Alt." | "Var.") /\d+/
LawAbbr      := <Token aus Alias-Map, Longest-Match>
Connector    := "iVm" | "i.V.m." | "iSd" | "i.S.d." | "iSv" | "nach Maßgabe des"
```

### 2.2 Listen, Bereiche, „ff."

- **Liste:** `§§ 14, 17 KStG` ⇒ zwei Anker `norm:KStG:14`, `norm:KStG:17`.
- **Bereich:** `§§ 28 bis 31 GewStG` ⇒ Anker 28,29,30,31 (auf vorhandene Knoten beschränken).
- **`ff.`** (folgende): als **offener Bereich** markieren, nicht raten. Anker auf die genannte Norm + Flag `offener_bereich=true`; Auflösung optional über die Gliederungsstruktur.
- **Verschachtelte Tiefe:** `§ 14 Abs. 1 Nr. 1 bis 3 KStG` ⇒ drei Anker mit gleichem `abs1`, `nr1..3`.

### 2.3 Relative Verweise — die Hauptfehlerquelle

Zitate **ohne** `LawAbbr` und ohne `§` sind relativ und beziehen sich auf den **aktuellen Kontextknoten**:

- „Satz 1", „Nummer 3", „des vorstehenden Absatzes", „dieser Vorschrift", „Absatz 2".

Der Parser braucht dafür einen **Kontext-Stack** (aktuelles Gesetz / § / Absatz beim Durchlaufen eines Normtexts). Auflösungsregeln:

- `Satz N` ohne weiteren Bezug ⇒ Satz N des aktuellen Absatzes.
- „des vorstehenden Absatzes" ⇒ aktueller Absatz − 1.
- „dieser Vorschrift" / „dieses Gesetzes" ⇒ aktueller § bzw. aktuelles Gesetz.

> Diese Klasse ist explizit zu testen und mit `konfidenz`-Flag zu versehen. Falsch aufgelöste relative Verweise erzeugen **falsche Kanten** — das ist der größte Qualitätsrisikoposten des Graphen.

### 2.4 Law-Alias-Map (automatisch aus dem Korpus)

Jede gii-Datei liefert ihr eigenes `<amtabk>` + `<jurabk>`. Beim Ingest daraus eine Alias-Map bauen:

```
"AO 1977"  -> AO
"AO"       -> AO
"UStG"     -> UStG
"KStG"     -> KStG
"GewStG"   -> GewStG
"EStG"     -> EStG
"AktG"     -> AktG
...
```
Longest-Match beim Parsen (z. B. „GewStG" vor „StG"). Unbekannte Abkürzungen (MwStSystRL, GG, EU-Recht) als Sonderliste pflegen.

### 2.5 Rückgabe

Der Parser liefert je Zitatstring eine **Liste** kanonischer IDs (Listen/Bereiche expandiert) plus pro ID: Quell-Span, `konfidenz`, `relativ?`-Flag.

---

## 3. Entscheidungszitat-Parser

### 3.1 Bestandteile

```
Entscheidung := Gericht "v." Datum "–"? Aktenzeichen Fundstelle*
Gericht      := "BFH" | "BGH" | "EuGH" | "BVerfG" | "FG" <Ort>? | "BVerwG" | ...
Datum        := /\d{1,2}\.\d{1,2}\.\d{4}/ | <ausgeschrieben>
Aktenzeichen := <gerichtsspezifisch, s. u.>
Fundstelle   := <Periodikum> <Jahr> "," <Seite>     (* z. B. BStBl. II 2023, 409 *)
```

### 3.2 Aktenzeichen-Muster (Auswahl)

| Gericht | Muster | Beispiel | Normalisiert |
|---|---|---|---|
| BFH | `<röm. Senat> R <Nr>/<Jahr>` | I R 37/19 | `I-R-37/19` |
| BFH | `XI R 29/22 (XI R 16/18)` | (Parallel-Az.) | Haupt-Az. + Alias |
| EuGH | `C-<Nr>/<Jahr>` | C-141/20 | `C-141/20` |
| BGH | `<röm.> ZR <Nr>/<Jahr>` | IX ZR 285/14 | `IX-ZR-285/14` |
| BVerfG | `<Nr> BvL <Nr>/<Jahr>` | 2 BvL 19/17 | `2-BvL-19/17` |
| FG | `<Nr> K <Nr>/<Jahr>` | 2 K 245/17 | `FGHamburg:2-K-245/17` |

Fundstellen (`BStBl. II 2023, 409`, `DStR 2023, 268`) werden als **Aliase** am Urteilsknoten geführt — nützlich, weil andere Dokumente oft über die Fundstelle statt das Az. zitieren.

### 3.3 Kanten aus Entscheidungszitaten

- **Urteil → Norm:** Normzitat-Parser über den Urteilstext ⇒ `wendet_an`.
- **Norm/Verwaltung → Urteil:** Entscheidungszitat in Annotation/UStAE ⇒ `wird_ausgelegt_durch`.
- **Urteil → Urteil:** Linkphrasen erkennen — „Anschluss an", „Abgrenzung zu", „Fortführung von", „entgegen" + folgendes Entscheidungszitat ⇒ `bezieht_sich_auf` mit Relationstyp.

---

## 4. gii-Knotenextraktion

### 4.1 Norm-Identifikation

- Echte Paragraphen-Norm: `<enbez>` matcht `^§\s*\d`. 
- Rahmen-/Inhaltsnorm (erste `<norm>`, nur `<gliederungseinheit>` / „Inhaltsübersicht") ⇒ überspringen, aber Gliederung für `ff.`-Auflösung behalten.
- `<LAW>` aus `<amtabk>`; `<para>` aus `<enbez>` (ohne „§ ").

### 4.2 Tiefe aus `<textdaten><text format="XML"><Content>`

- **Absatz:** `<P>`-Elemente; Absatz erkannt am führenden `(\d+[a-z]*)`. 
- **Nummer:** `<DL>` mit `<DT>` (z. B. „1.") und `<DD><LA>…`.
- **Buchstabe:** verschachtelte `<DL Type="alpha">`.
- **Satz:** Sentence-Splitter mit **deutscher Legal-Abkürzungs-Schutzliste** (`z. B.`, `Abs.`, `Nr.`, `S.`, `Art.`, `§§`, `i. S. d.`, `i. V. m.`, `ff.`, Dezimalzahlen, Datumsangaben). Satz-Granularität ist heuristisch und die unsicherste Ebene.

> **Empfehlung:** Den Katalog überwiegend auf **Absatz-/Nummern-Ebene** ankern — die kommt zuverlässig aus der Struktur. Auf **Satz-Ebene** nur dort gehen, wo der Katalog es zwingend braucht, und dort den Splitter gegen ein Gold-Set testen.

### 4.3 Metadaten

- Überschrift aus `<titel>`. Stand/Änderungen aus `<standangabe>`/`<standkommentar>` (nur Historie der aktuellen Fassung, **keine** Zeitreise).
- `gueltig_ab` = `<ausfertigung-datum>` bzw. letzter Standkommentar; `gueltig_bis = null` (konsolidierte Fassung).

---

## 5. Validierung & Tests

- /T10/ **Round-Trip:** Zitatstring → ID → rekonstruiertes Zitat. Muss verlustfrei sein (modulo Normalisierung).
- /T20/ **Gold-Set Normzitate:** Hand-annotierte Zitate aus dem DStR-Artikel (er enthält Dutzende: `§ 2 Abs. 2 S. 2 GewStG`, `§ 14 Abs. 1 Nr. 3 KStG`, …). Precision/Recall messen.
- /T30/ **Relative Verweise** als eigene Test-Suite mit Kontext-Stack-Fällen.
- /T40/ **Gold-Set Entscheidungszitate:** ebenfalls aus dem Artikel (`BFH v. 11.7.2024 – III R 41/22`, `EuGH v. 1.12.2022 – C-141/20`, `BVerfG – 2 BvL 19/17`).
- /T50/ **Dangling-Edge-Report:** Verweise auf nicht existierende Knoten ausweisen (Tippfehler im Zitat, fehlendes Gesetz im Korpus, offene `ff.`-Bereiche).

---

## 6. Reihenfolge für die Umsetzung

1. Law-Alias-Map aus dem Korpus harvesten.
2. gii-Ingest → Norm-Knoten (Absatz/Nummer-Granularität zuerst, Satz später).
3. Normzitat-Parser (absolut) + Gold-Set-Test.
4. Kanten `verweist_auf` aus Normtexten.
5. Katalog-Ankopplung (`norm_anker` → Knoten).
6. Verwaltungs-Ingest (UStAE/AEAO) + strukturelle `wird_ausgelegt_durch`-Kanten.
7. Entscheidungszitat-Parser + Urteils-Feed + Frühwarn-Trigger auf Katalogeinträge.
8. Relative Verweise + Satz-Granularität (die unsicheren Teile zuletzt, mit Tests).
