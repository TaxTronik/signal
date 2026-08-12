# Versionierte Eingaben für Source-Builds

Der von TaxTronik verwaltete Git-/Source-Pfad baut das Festwissen aus den hier
versionierten, im selben Commit geprüften Normquellen. Dadurch hängt ein Deploy
nicht vom momentanen Zugriff des Zielservers auf externe Gesetzesportale ab und
derselbe Signal-Commit erzeugt denselben fachlichen Ausgangsstand.

`release-input/corpus/gii` enthält exakt die vom aktuellen Katalog benötigten
XML-Quellen. Änderungen daran erfolgen ausschließlich über einen bewussten
Festwissen-Release mit anschließendem Parser-, Graph- und Gold-Test. Der normale
Release-Builder bezieht ohne `--corpus-snapshot` weiterhin ausdrücklich aktuelle
Quellen von `gesetze-im-internet.de`; der Managed-Source-Build verwendet dagegen
immer diesen Commit-Snapshot.

Der Snapshot enthält keinen Embedding-Index und keine Mandanten- oder
Beraterdaten.
