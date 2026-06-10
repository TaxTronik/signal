# Erweiterung: Drei-Stufen-Vertrauensmodell & Trigger als ungeprüfter Vor-Katalog

> **Eingefroren (Stand 31.05.2026):** Historisches Anforderungsdokument. Der Code
> referenziert das hier definierte Vertrauensmodell (geprüft/ungeprüft/ki_vermutung).
> Inhaltliche Änderungen nur über ein neues ADR — der gelebte Stand steht in
> README/FEATURES/INTEGRATION.

**Companion zum Lastenheft „Schlüsselbegriff-Assistenz" / Deterministik-Katalog**
**Stand:** 31.05.2026
**Zweck:** Den Henne-Ei-Start lösen (leerer Katalog = Layer 1 hat nichts zu tun), ohne ungeprüftes Wissen als gesichert auszugeben. Eine vordefinierte Triggerliste liefert Grundabdeckung, trägt aber sichtbar den Status „ungeprüft" und ist die Pipeline, durch die der geprüfte Katalog durch Nutzung wächst.

---

## 1. Kernidee

Es gibt nicht zwei, sondern **drei** Vertrauensstufen für eine Markierung. Sie unterscheiden sich entlang **zwei unabhängiger Achsen**:

- **Reproduzierbarkeit:** deterministisch (gleiche Eingabe → gleiches Ergebnis) vs. probabilistisch (Modell).
- **Fachliche Validierung:** von einem Berufsträger bestätigt vs. (noch) nicht.

| Stufe | Reproduzierbar? | Fachlich bestätigt? | Quelle |
|---|---|---|---|
| **`geprüft`** | ja (deterministisch) | ja | kuratierter Katalog |
| **`ungeprüft`** | ja (deterministisch) | **nein** | vordefinierte Triggerliste |
| **`ki_vermutung`** | **nein** (probabilistisch) | nein | Layer-2-LLM |

**Wichtige Klarstellung:** `ungeprüft` ist **sicherer** als `ki_vermutung` (es beruht auf einem bewusst formulierten, generalisierbaren Muster und ist reproduzierbar), aber **unsicherer** als `geprüft` (die fachliche Richtigkeit ist noch nicht bestätigt). Es sitzt sauber dazwischen.

`ungeprüft ≠ probabilistisch.` Der Trigger ist verlässlich *in dem, was er tut* (deterministisches Matching), nur noch nicht abgesegnet *in dem, was er behauptet* (fachliche Einordnung). Das ist ein anderer Mangel als bei der KI-Vermutung und muss in der Doku/PS-980-Argumentation getrennt gehalten werden.

---

## 2. Anforderungen

### 2.1 Statusführung
- /LF400/ Jede Markierung trägt genau eine der drei Stufen: `geprüft | ungeprüft | ki_vermutung`.
- /LF410/ Die drei Stufen sind in der Anzeige **klar unterscheidbar** (Farbe/Stil/Icon). `ungeprüft` darf NICHT wie `geprüft` aussehen — sonst ist die Verlässlichkeit von Layer 1 entwertet.
- /LF420/ Sortier-/Anzeigepriorität: `geprüft` führt, dann `ungeprüft`, dann `ki_vermutung`. Keine Stufe unterdrückt eine andere (recall-first); Konfidenz nur zum Sortieren, nie als Gate.

### 2.2 Triggerliste (der ungeprüfte Vor-Katalog)
- /LF430/ Trigger sind **klassenbasiert**, nicht textspezifisch. Pflichtfeld `greift_bei_klasse` (welche Fallklasse), NICHT `greift_bei_text`. Das erzwingt Generalisierbarkeit und verhindert Overfitting/„Betrug am System".
- /LF440/ Trigger matchen deterministisch (Muster/Phrase/Lemma), identisch zur Layer-1-Mechanik.
- /LF450/ Trigger erzeugen Markierungen mit Status `ungeprüft`.
- /LF460/ Lackmustest beim Anlegen: Hülfe dieser Trigger auch einem *ungesehenen* Fall derselben Klasse? Nein ⇒ kein Trigger, sondern Spachtelmasse ⇒ verwerfen.

### 2.3 Beförderung (der Wachstumspfad)
- /LF470/ Sieht ein Berufsträger einen `ungeprüft`-Treffer im echten Fall und bestätigt ihn, wird der zugrunde liegende Trigger zum `geprüft`-Katalogeintrag **befördert** (mit Norm-Anker, Definitionsquelle, Governance-Typ ergänzt).
- /LF480/ Beförderung ist ein menschlicher Akt (Berufsträger), kein automatischer. Provenienz wird mitgeführt (wer, wann).
- /LF490/ Die ungeprüfte Liste ist damit kein Müll und kein Dauerzustand, sondern die **Seed-/Wachstums-Pipeline** des geprüften Katalogs (analog zur `unknown_risiko`-Define-on-the-fly-Mechanik).

---

## 3. Datenmodell

### 3.1 Trigger-Eintrag (ungeprüfter Vor-Katalog)

```yaml
- id: trg_unterlassene_gav_buchung
  muster:                              # generalisierbare Erkennungsmuster
    - "wird nicht ausgewiesen"
    - "bleibt aus"
    - "unterbleibt"
    - "nicht gebucht"
  greift_bei_klasse: "Negativbefund: unterlassene Pflichthandlung im GAV-Kontext"
  kontext_hinweis:                     # optionales Gating, erhöht Precision
    norm_naehe: ["§ 14 KStG", "GAV", "Gewinnabführung"]
  begruendung_kurz: "tatsächliche Durchführung GAV (Unterlassung)"
  status: "ungeprüft"
  herkunft: "vordefiniert"             # vordefiniert | aus_ki_vermutung | aus_berater
  generalisierbar: true               # Pflicht-Flag, false ⇒ nicht aufnehmen
  beförderbar_zu: null                # nach Beförderung: ID des Katalogeintrags
```

> **Designhinweis:** Die Unterlassungs-/Negativbefund-Muster sind hier besonders wertvoll, weil sie genau die Layer-2-Schwäche kompensieren (LLMs sind schwach bei Negativwissen — „die Forderung wird nicht ausgewiesen" wird vom Modell als banale Tatsache gelesen, nicht als Tatbestandsverletzung). Ein deterministischer Trigger fängt das zuverlässig.

### 3.2 Markierung (Laufzeit-Ausgabe)

```json
{
  "span_text": "wird nicht ausgewiesen",
  "char_offset": 412,
  "stufe": "ungeprüft",
  "begruendung_kurz": "tatsächliche Durchführung GAV (Unterlassung)",
  "quelle_id": "trg_unterlassene_gav_buchung",
  "beförderbar": true
}
```

---

## 4. Statusübergänge

```
   (vordefiniert)            Berufsträger bestätigt
  ungeprüft ───────────────────────────────────────►  geprüft
      ▲                                                   │
      │ Berufsträger stuft KI-Vermutung als               │ (kuratiert,
      │ generalisierbares Muster hoch                     │  Norm/Quelle/Gov)
      │                                                   ▼
  ki_vermutung ─────────────────────────────────►  (verworfen, wenn unbrauchbar)
```

- `ki_vermutung → ungeprüft`: Berater erkennt in einer KI-Markierung ein generalisierbares Muster ⇒ formuliert es als Trigger (Mensch, nicht automatisch).
- `ungeprüft → geprüft`: Bestätigung + Anreicherung (Norm-Anker, Definitionsquelle, Governance-Typ).
- jede Stufe → `verworfen`: Fehlalarm/unbrauchbar; bei Triggern: Muster war nicht generalisierbar.

---

## 5. Abnahmekriterien

- /AB80/ Die drei Stufen sind in der UI eindeutig unterscheidbar; ein Nutzer ordnet eine Markierung im Blindtest korrekt zu.
- /AB90/ Kein `ungeprüft`/`ki_vermutung`-Treffer wird optisch oder im Datenmodell als `geprüft` ausgegeben.
- /AB100/ Jeder Trigger hat ein nicht-leeres `greift_bei_klasse` und `generalisierbar: true`; textspezifische Trigger werden bei der Aufnahme abgelehnt.
- /AB110/ Beförderung `ungeprüft → geprüft` ist nur durch einen menschlichen Akt auslösbar und schreibt Provenienz.
- /AB120/ Determinismus: Trigger-Matching ist bit-reproduzierbar (gleiche Eingabe + gleiche Triggerlisten-Version → gleiche Treffer).

---

## 6. Abgrenzung / Fallen

- **Nicht** die Overfitting-Falle: Trigger sind klassenbasiert und generalisierbar; der `generalisierbar`-Zwang ist die eingebaute Bremse gegen fallweises Flicken.
- **Nicht** Autoritätsanschein: `ungeprüft` ist sichtbar ungeprüft; es leiht sich nicht die Glaubwürdigkeit von `geprüft`.
- **Nicht** mit dem Gold-Set verwechseln: Die Triggerliste ist *Input* fürs Tool. Das Gold-Set ist *Testmaterial*. Getrennt halten.
- **Reihenfolge:** Diese Erweiterung setzt einen funktionierenden Layer-1-Matcher voraus. Erst der Parser-/Matcher-Test am Gold-Set, dann diese Stufen darauf.
