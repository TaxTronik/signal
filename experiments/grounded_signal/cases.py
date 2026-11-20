"""Assistant-reviewed synthetic diagnostics, not expert gold or a blind test.

Only ``text`` belongs in a case's model input. ``diagnostic_labels`` and the
family-level prepared package selection are evaluator data. Prepared packages
are deliberately identical for both family members, including empty-hint cases.
The known failure patterns motivated these newly authored synthetic texts.
No model, encoder, graph, or reference-report loading occurs on import.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from typing import Any

DATASET_VERSION = "grounded_signal_cases/1"
ANNOTATION_STATUS = "assistant_reviewed_synthetic"
SOURCE_GRAPH_SHA256 = "3ac8f3d272ca6dc32274ee2c7d9dbb8237bf200ca8abd44603e5cd6128175862"
SPLITS = ("development", "evaluation")

_CASES: tuple[dict[str, Any], ...] = ({'case_id': 'gs-estimation_actor-anchor',
  'family_id': 'estimation_actor',
  'split': 'development',
  'text': 'Das Finanzamt kann die steuerpflichtigen Umsätze der Nordmarkt GmbH anhand der vorgelegten '
          'Aufzeichnungen nicht ermitteln und kündigt eine Schätzung an. Die Geschäftsführerin fragt, ob dabei '
          'beliebige Beträge angesetzt werden dürfen.',
  'prepared_norm_ids': ['norm:AO:162', 'norm:AO:158', 'norm:AO:147'],
  'diagnostic_labels': {'annotation_status': 'assistant_reviewed_synthetic',
                        'variant': 'anchor',
                        'contrast_dimension': 'actor_and_process',
                        'hint_expected': True,
                        'target_norm_ids': ['norm:AO:162'],
                        'allowed_hint_norm_ids': ['norm:AO:158'],
                        'no_hint_norm_ids': ['norm:AO:147'],
                        'insufficient_context_norm_ids': [],
                        'expected_open_conditions': ['Welche konkreten Ermittlungsdefizite bestehen und welche '
                                                     'tatsächlichen Umstände eine sachgerechte Schätzung '
                                                     'tragen, ist nicht beschrieben.'],
                        'prohibited_factual_assertions': ['Das Finanzamt darf ohne Rücksicht auf tatsächliche '
                                                          'Umstände beliebige Besteuerungsgrundlagen '
                                                          'festsetzen.',
                                                          'Eine Steuerhinterziehung oder ein konkreter '
                                                          'schuldhafter Buchführungsmangel steht bereits '
                                                          'fest.'],
                        'source_based_rationale': [{'norm_id': 'norm:AO:162',
                                                    'source_quote': '(1) Soweit die Finanzbehörde die '
                                                                    'Besteuerungsgrundlagen nicht ermitteln '
                                                                    'oder berechnen kann, hat sie sie zu '
                                                                    'schätzen. Dabei sind alle Umstände zu '
                                                                    'berücksichtigen, die für die Schätzung '
                                                                    'von Bedeutung sind.',
                                                    'source_start': 0,
                                                    'source_end': 204,
                                                    'source_text_sha256': '50e1d040b644b9dc89879f414615975dbf86a4700cf40622546b24fec2144090',
                                                    'explanation': 'Schätzung ist an fehlende Ermittelbarkeit '
                                                                   'oder Berechenbarkeit gebunden; relevante '
                                                                   'Umstände müssen berücksichtigt werden. Ein '
                                                                   'Hinweis darf gerade die Grenzen einer '
                                                                   'Schätzung erklären.'},
                                                   {'norm_id': 'norm:AO:158',
                                                    'source_quote': '(1) Die Buchführung und die '
                                                                    'Aufzeichnungen des Steuerpflichtigen, die '
                                                                    'den Vorschriften der §§ 140 bis 148 '
                                                                    'entsprechen, sind der Besteuerung '
                                                                    'zugrunde zu legen.(2) Absatz 1 gilt '
                                                                    'nicht, 1.soweit nach den Umständen des '
                                                                    'Einzelfalls Anlass besteht, die sachliche '
                                                                    'Richtigkeit zu beanstanden oder2.soweit '
                                                                    'die elektronischen Daten nicht nach der '
                                                                    'Vorgabe der einheitlichen digitalen '
                                                                    'Schnittstellen des § 41 Absatz 1 Satz 7 '
                                                                    'des Einkommensteuergesetzes in Verbindung '
                                                                    'mit § 4 Absatz 2a der '
                                                                    'Lohnsteuer-Durchführungsverordnung, des § '
                                                                    '146a oder des § 147b in Verbindung mit '
                                                                    'der jeweiligen Rechtsverordnung zur '
                                                                    'Verfügung gestellt werden.',
                                                    'source_start': 0,
                                                    'source_end': 620,
                                                    'source_text_sha256': '724d21895bb51574b2f469afa47ba2803d08c5703a5272da68cbda9ecf7035dd',
                                                    'explanation': 'Die Beweiskraft kann als verwandte '
                                                                   'Prüffrage vorkommen; ein automatischer '
                                                                   'vollständiger Beweiskraftverlust ist nicht '
                                                                   'vorgegeben.'}]}},
 {'case_id': 'gs-estimation_actor-contrast',
  'family_id': 'estimation_actor',
  'split': 'development',
  'text': 'Die Vertriebsplanung der Nordmarkt GmbH schätzt aus unvollständigen internen Zahlen den Absatz des '
          'nächsten Quartals. Dabei wird kein Steuerbescheid erstellt und keine Finanzbehörde tätig.',
  'prepared_norm_ids': ['norm:AO:162', 'norm:AO:158', 'norm:AO:147'],
  'diagnostic_labels': {'annotation_status': 'assistant_reviewed_synthetic',
                        'variant': 'contrast',
                        'contrast_dimension': 'actor_and_process',
                        'hint_expected': False,
                        'target_norm_ids': [],
                        'allowed_hint_norm_ids': [],
                        'no_hint_norm_ids': ['norm:AO:162', 'norm:AO:158', 'norm:AO:147'],
                        'insufficient_context_norm_ids': [],
                        'expected_open_conditions': [],
                        'prohibited_factual_assertions': ['Die Vertriebsplanung handelt als Finanzbehörde oder '
                                                          'setzt Besteuerungsgrundlagen fest.',
                                                          'Es liegt eine steuerliche Schätzung oder ein '
                                                          'festgestellter steuerlicher Aufbewahrungsmangel '
                                                          'vor.'],
                        'source_based_rationale': [{'norm_id': 'norm:AO:162',
                                                    'source_quote': '(1) Soweit die Finanzbehörde die '
                                                                    'Besteuerungsgrundlagen nicht ermitteln '
                                                                    'oder berechnen kann, hat sie sie zu '
                                                                    'schätzen. Dabei sind alle Umstände zu '
                                                                    'berücksichtigen, die für die Schätzung '
                                                                    'von Bedeutung sind.',
                                                    'source_start': 0,
                                                    'source_end': 204,
                                                    'source_text_sha256': '50e1d040b644b9dc89879f414615975dbf86a4700cf40622546b24fec2144090',
                                                    'explanation': 'Der ausdrücklich beschriebene interne '
                                                                   'Prognosevorgang ist keine Schätzung durch '
                                                                   'die Finanzbehörde nach diesem Absatz.'},
                                                   {'norm_id': 'norm:AO:147',
                                                    'source_quote': '(1) Die folgenden Unterlagen sind '
                                                                    'geordnet aufzubewahren: 1.Bücher und '
                                                                    'Aufzeichnungen, Inventare, '
                                                                    'Jahresabschlüsse, Lageberichte, die '
                                                                    'Eröffnungsbilanz sowie die zu ihrem '
                                                                    'Verständnis erforderlichen '
                                                                    'Arbeitsanweisungen und sonstigen '
                                                                    'Organisationsunterlagen,2.die empfangenen '
                                                                    'Handels- oder '
                                                                    'Geschäftsbriefe,3.Wiedergaben der '
                                                                    'abgesandten Handels- oder '
                                                                    'Geschäftsbriefe,4.Buchungsbelege,4a.Unterlagen '
                                                                    'nach Artikel 15 Absatz 1 und Artikel 163 '
                                                                    'des Zollkodex der Union,5.sonstige '
                                                                    'Unterlagen, soweit sie für die '
                                                                    'Besteuerung von Bedeutung sind.',
                                                    'source_start': 0,
                                                    'source_end': 531,
                                                    'source_text_sha256': '8584cf1334bcc3dd38ad8ac50da660c83cb939bbc3da67f5fc2784e5767871b2',
                                                    'explanation': 'Die Quelle regelt geordnete Aufbewahrung. '
                                                                   'Der Fall enthält keine Aussage über deren '
                                                                   'Verletzung oder über eine '
                                                                   'Aufbewahrungsfrage.'}]}},
 {'case_id': 'gs-cash_exception-anchor',
  'family_id': 'cash_exception',
  'split': 'development',
  'text': 'Mira verkauft in ihrem Laden Waren gegen Barzahlung an viele unbekannte Kundinnen und Kunden. Sie '
          'nutzt eine offene Ladenkasse ohne elektronisches System. Sie fragt, ob eine Ausnahme von '
          'Einzelaufzeichnungen auch das tägliche Festhalten der Kasseneinnahmen entfallen lässt.',
  'prepared_norm_ids': ['norm:AO:146', 'norm:AO:146a', 'norm:AO:147'],
  'diagnostic_labels': {'annotation_status': 'assistant_reviewed_synthetic',
                        'variant': 'anchor',
                        'contrast_dimension': 'exception_and_device',
                        'hint_expected': True,
                        'target_norm_ids': ['norm:AO:146'],
                        'allowed_hint_norm_ids': ['norm:AO:146a'],
                        'no_hint_norm_ids': ['norm:AO:147'],
                        'insufficient_context_norm_ids': [],
                        'expected_open_conditions': ['Die tatsächlichen Voraussetzungen der '
                                                     'Einzelaufzeichnungsausnahme sind für den konkreten '
                                                     'Verkauf zu prüfen; eine Befreiung von täglicher '
                                                     'Aufzeichnung folgt daraus nicht.'],
                        'prohibited_factual_assertions': ['Wegen unbekannter Barzahlender müssen '
                                                          'Kasseneinnahmen überhaupt nicht täglich '
                                                          'festgehalten werden.',
                                                          'Mira benutzt ein elektronisches '
                                                          'Aufzeichnungssystem.',
                                                          'Die im Normtext genannte '
                                                          'Einzelaufzeichnungsausnahme steht in Absatz 1 Satz '
                                                          '2.'],
                        'source_based_rationale': [{'norm_id': 'norm:AO:146',
                                                    'source_quote': '(1) Die Buchungen und die sonst '
                                                                    'erforderlichen Aufzeichnungen sind '
                                                                    'einzeln, vollständig, richtig, '
                                                                    'zeitgerecht und geordnet vorzunehmen. '
                                                                    'Kasseneinnahmen und Kassenausgaben sind '
                                                                    'täglich festzuhalten. Die Pflicht zur '
                                                                    'Einzelaufzeichnung nach Satz 1 besteht '
                                                                    'aus Zumutbarkeitsgründen bei Verkauf von '
                                                                    'Waren an eine Vielzahl von nicht '
                                                                    'bekannten Personen gegen Barzahlung '
                                                                    'nicht. Das gilt nicht, wenn der '
                                                                    'Steuerpflichtige ein elektronisches '
                                                                    'Aufzeichnungssystem im Sinne des § 146a '
                                                                    'verwendet.',
                                                    'source_start': 0,
                                                    'source_end': 481,
                                                    'source_text_sha256': 'b51aef093a85adda5df370d0aa3ac2d422d6b85782f9a6c46133730a8a7bfe58',
                                                    'explanation': 'Die Zumutbarkeitsausnahme betrifft '
                                                                   'Einzelaufzeichnung nach Satz 1, nicht das '
                                                                   'tägliche Festhalten nach Satz 2. Bei einem '
                                                                   'elektronischen System im Sinn des '
                                                                   'Verweises gilt die Ausnahme nicht.'}]}},
 {'case_id': 'gs-cash_exception-contrast',
  'family_id': 'cash_exception',
  'split': 'development',
  'text': 'Mira verkauft in ihrem Laden Waren gegen Barzahlung an viele unbekannte Kundinnen und Kunden. Sie '
          'nutzt ein elektronisches Aufzeichnungssystem mit Kassenfunktion. Sie fragt, ob eine Ausnahme von '
          'Einzelaufzeichnungen auch das tägliche Festhalten der Kasseneinnahmen entfallen lässt.',
  'prepared_norm_ids': ['norm:AO:146', 'norm:AO:146a', 'norm:AO:147'],
  'diagnostic_labels': {'annotation_status': 'assistant_reviewed_synthetic',
                        'variant': 'contrast',
                        'contrast_dimension': 'exception_and_device',
                        'hint_expected': True,
                        'target_norm_ids': ['norm:AO:146', 'norm:AO:146a'],
                        'allowed_hint_norm_ids': [],
                        'no_hint_norm_ids': ['norm:AO:147'],
                        'insufficient_context_norm_ids': [],
                        'expected_open_conditions': ['Ob das konkrete Gerät vollständig unter die verwiesene '
                                                     'Systemregelung fällt und deren technische Anforderungen '
                                                     'erfüllt, ist nicht abschließend belegt.'],
                        'prohibited_factual_assertions': ['Die Ausnahme von Einzelaufzeichnungen gilt '
                                                          'unabhängig vom verwendeten elektronischen System.',
                                                          'Das System ist nachweislich zertifiziert oder '
                                                          'sämtliche technischen Pflichten sind bereits '
                                                          'erfüllt.',
                                                          'Eine Einzelaufzeichnungsausnahme beseitigt die '
                                                          'tägliche Aufzeichnungspflicht.'],
                        'source_based_rationale': [{'norm_id': 'norm:AO:146',
                                                    'source_quote': '(1) Die Buchungen und die sonst '
                                                                    'erforderlichen Aufzeichnungen sind '
                                                                    'einzeln, vollständig, richtig, '
                                                                    'zeitgerecht und geordnet vorzunehmen. '
                                                                    'Kasseneinnahmen und Kassenausgaben sind '
                                                                    'täglich festzuhalten. Die Pflicht zur '
                                                                    'Einzelaufzeichnung nach Satz 1 besteht '
                                                                    'aus Zumutbarkeitsgründen bei Verkauf von '
                                                                    'Waren an eine Vielzahl von nicht '
                                                                    'bekannten Personen gegen Barzahlung '
                                                                    'nicht. Das gilt nicht, wenn der '
                                                                    'Steuerpflichtige ein elektronisches '
                                                                    'Aufzeichnungssystem im Sinne des § 146a '
                                                                    'verwendet.',
                                                    'source_start': 0,
                                                    'source_end': 481,
                                                    'source_text_sha256': 'b51aef093a85adda5df370d0aa3ac2d422d6b85782f9a6c46133730a8a7bfe58',
                                                    'explanation': 'Die Zumutbarkeitsausnahme betrifft '
                                                                   'Einzelaufzeichnung nach Satz 1, nicht das '
                                                                   'tägliche Festhalten nach Satz 2. Bei einem '
                                                                   'elektronischen System im Sinn des '
                                                                   'Verweises gilt die Ausnahme nicht.'},
                                                   {'norm_id': 'norm:AO:146a',
                                                    'source_quote': '(1) Wer aufzeichnungspflichtige '
                                                                    'Geschäftsvorfälle oder andere Vorgänge '
                                                                    'mit Hilfe eines elektronischen '
                                                                    'Aufzeichnungssystems erfasst, hat ein '
                                                                    'elektronisches Aufzeichnungssystem zu '
                                                                    'verwenden, das jeden '
                                                                    'aufzeichnungspflichtigen Geschäftsvorfall '
                                                                    'und anderen Vorgang einzeln, vollständig, '
                                                                    'richtig, zeitgerecht und geordnet '
                                                                    'aufzeichnet. Das elektronische '
                                                                    'Aufzeichnungssystem und die digitalen '
                                                                    'Aufzeichnungen nach Satz 1 sind durch '
                                                                    'eine zertifizierte technische '
                                                                    'Sicherheitseinrichtung zu schützen. Diese '
                                                                    'zertifizierte technische '
                                                                    'Sicherheitseinrichtung muss aus einem '
                                                                    'Sicherheitsmodul, einem Speichermedium '
                                                                    'und einer einheitlichen digitalen '
                                                                    'Schnittstelle bestehen. Die digitalen '
                                                                    'Aufzeichnungen sind auf dem '
                                                                    'Speichermedium zu sichern und für '
                                                                    'Nachschauen sowie Außenprüfungen durch '
                                                                    'elektronische Aufbewahrung verfügbar zu '
                                                                    'halten. Es ist verboten, innerhalb des '
                                                                    'Geltungsbereichs dieses Gesetzes solche '
                                                                    'elektronischen Aufzeichnungssysteme, '
                                                                    'Software für elektronische '
                                                                    'Aufzeichnungssysteme und zertifizierte '
                                                                    'technische Sicherheitseinrichtungen, die '
                                                                    'den in den Sätzen 1 bis 3 beschriebenen '
                                                                    'Anforderungen nicht entsprechen, zur '
                                                                    'Verwendung im Sinne der Sätze 1 bis 3 '
                                                                    'gewerbsmäßig zu bewerben oder '
                                                                    'gewerbsmäßig in den Verkehr zu bringen.',
                                                    'source_start': 0,
                                                    'source_end': 1218,
                                                    'source_text_sha256': '1e33a0c4d6957a6d1dd19c43f4a36de2f262eaa95f7873d645cdeaa4656a2026',
                                                    'explanation': 'Die Norm behandelt elektronische '
                                                                   'Einzelaufzeichnung und technische '
                                                                   'Schutzpflichten; aus dem bloßen '
                                                                   'Gerätegebrauch folgt keine erfüllte '
                                                                   'Zertifizierung.'}]}},
 {'case_id': 'gs-representative_role-anchor',
  'family_id': 'representative_role',
  'split': 'development',
  'text': 'Lea ist gesetzliche Vertreterin der Zedern GmbH und verwaltet deren Bankkonto. Obwohl genügend '
          'Mittel vorhanden sind, lässt sie fällige Steuern offen und bezahlt Lieferanten. Eine persönliche '
          'Haftung wird geprüft.',
  'prepared_norm_ids': ['norm:AO:34', 'norm:AO:69', 'norm:AO:158'],
  'diagnostic_labels': {'annotation_status': 'assistant_reviewed_synthetic',
                        'variant': 'anchor',
                        'contrast_dimension': 'missing_role_and_tax_context',
                        'hint_expected': True,
                        'target_norm_ids': ['norm:AO:34', 'norm:AO:69'],
                        'allowed_hint_norm_ids': [],
                        'no_hint_norm_ids': ['norm:AO:158'],
                        'insufficient_context_norm_ids': [],
                        'expected_open_conditions': ['Pflichtverletzung, Vorsatz oder grobe Fahrlässigkeit und '
                                                     'deren Zusammenhang mit dem Steuerausfall sind zu '
                                                     'prüfen.'],
                        'prohibited_factual_assertions': ['Vorsatz oder grobe Fahrlässigkeit und eine '
                                                          'persönliche Haftung sind allein durch die '
                                                          'Zahlungsreihenfolge abschließend bewiesen.',
                                                          'Die Haftungsnorm nennt als Erstattungstatbestand '
                                                          'lediglich unterbliebene Steuererstattungen.',
                                                          'Die Grundpflicht ist irrelevant, sobald eine '
                                                          'spezielle Haftungsnorm existiert.'],
                        'source_based_rationale': [{'norm_id': 'norm:AO:34',
                                                    'source_quote': '(1) Die gesetzlichen Vertreter '
                                                                    'natürlicher und juristischer Personen '
                                                                    'sowie rechtsfähiger Personenvereinigungen '
                                                                    'und die Geschäftsführer von '
                                                                    'Vermögensmassen haben deren steuerliche '
                                                                    'Pflichten zu erfüllen. Sie haben '
                                                                    'insbesondere dafür zu sorgen, dass die '
                                                                    'Steuern aus den Mitteln entrichtet '
                                                                    'werden, die sie verwalten. Die '
                                                                    'Finanzbehörde kann sich an jeden von '
                                                                    'ihnen wenden.',
                                                    'source_start': 0,
                                                    'source_end': 367,
                                                    'source_text_sha256': '5af1006d8fb49dc1a60594990e45ea88c38e4b0b748a17b584e2bf11d17172b6',
                                                    'explanation': 'Grundpflichten zur Entrichtung aus '
                                                                   'verwalteten Mitteln und die daran '
                                                                   'anknüpfende Haftungsfrage können gemeinsam '
                                                                   'hilfreich sein. Haftung setzt weitere '
                                                                   'Bedingungen voraus.'},
                                                   {'norm_id': 'norm:AO:69',
                                                    'source_quote': 'Die in den §§ 34 und 35 bezeichneten '
                                                                    'Personen haften, soweit Ansprüche aus dem '
                                                                    'Steuerschuldverhältnis (§ 37) infolge '
                                                                    'vorsätzlicher oder grob fahrlässiger '
                                                                    'Verletzung der ihnen auferlegten '
                                                                    'Pflichten nicht oder nicht rechtzeitig '
                                                                    'festgesetzt oder erfüllt oder soweit '
                                                                    'infolgedessen Steuervergütungen oder '
                                                                    'Steuererstattungen ohne rechtlichen Grund '
                                                                    'gezahlt werden. Die Haftung umfasst auch '
                                                                    'die infolge der Pflichtverletzung zu '
                                                                    'zahlenden Säumniszuschläge.',
                                                    'source_start': 0,
                                                    'source_end': 447,
                                                    'source_text_sha256': '2d985f33ec184b4388b9639652fc8f5f0b5c7fe780db88feda2f590480cfbd28',
                                                    'explanation': 'Grundpflichten zur Entrichtung aus '
                                                                   'verwalteten Mitteln und die daran '
                                                                   'anknüpfende Haftungsfrage können gemeinsam '
                                                                   'hilfreich sein. Haftung setzt weitere '
                                                                   'Bedingungen voraus.'}]}},
 {'case_id': 'gs-representative_role-contrast',
  'family_id': 'representative_role',
  'split': 'development',
  'text': 'Lea berichtet, dass verfügbare Mittel zuerst an einige Gläubiger gezahlt wurden. Wer die Zahlungen '
          'veranlasste, um wessen Vermögen es ging und welche Forderung offenblieb, ist unbekannt.',
  'prepared_norm_ids': ['norm:AO:34', 'norm:AO:69', 'norm:AO:158'],
  'diagnostic_labels': {'annotation_status': 'assistant_reviewed_synthetic',
                        'variant': 'contrast',
                        'contrast_dimension': 'missing_role_and_tax_context',
                        'hint_expected': False,
                        'target_norm_ids': [],
                        'allowed_hint_norm_ids': [],
                        'no_hint_norm_ids': ['norm:AO:158'],
                        'insufficient_context_norm_ids': ['norm:AO:34', 'norm:AO:69'],
                        'expected_open_conditions': ['Vertreter- oder Verfügungsrolle, Steuerbezug, betroffene '
                                                     'Forderung und verantwortliche Person fehlen als '
                                                     'Grundkontext; sie dürfen nicht als bloßes '
                                                     'Verschuldensdetail ergänzt werden.'],
                        'prohibited_factual_assertions': ['Lea ist gesetzliche Vertreterin oder verfügt '
                                                          'rechtlich über fremdes Vermögen.',
                                                          'Die offengebliebene Forderung ist eine '
                                                          'Steuerforderung.',
                                                          'Steuerzahlungen wurden vorsätzlich oder grob '
                                                          'fahrlässig vereitelt.'],
                        'source_based_rationale': [{'norm_id': 'norm:AO:34',
                                                    'source_quote': '(1) Die gesetzlichen Vertreter '
                                                                    'natürlicher und juristischer Personen '
                                                                    'sowie rechtsfähiger Personenvereinigungen '
                                                                    'und die Geschäftsführer von '
                                                                    'Vermögensmassen haben deren steuerliche '
                                                                    'Pflichten zu erfüllen. Sie haben '
                                                                    'insbesondere dafür zu sorgen, dass die '
                                                                    'Steuern aus den Mitteln entrichtet '
                                                                    'werden, die sie verwalten. Die '
                                                                    'Finanzbehörde kann sich an jeden von '
                                                                    'ihnen wenden.',
                                                    'source_start': 0,
                                                    'source_end': 367,
                                                    'source_text_sha256': '5af1006d8fb49dc1a60594990e45ea88c38e4b0b748a17b584e2bf11d17172b6',
                                                    'explanation': 'Der Fall belegt die von der Quelle '
                                                                   'vorausgesetzte Rolle und den steuerlichen '
                                                                   'Vorgang nicht.'},
                                                   {'norm_id': 'norm:AO:69',
                                                    'source_quote': 'Die in den §§ 34 und 35 bezeichneten '
                                                                    'Personen haften, soweit Ansprüche aus dem '
                                                                    'Steuerschuldverhältnis (§ 37) infolge '
                                                                    'vorsätzlicher oder grob fahrlässiger '
                                                                    'Verletzung der ihnen auferlegten '
                                                                    'Pflichten nicht oder nicht rechtzeitig '
                                                                    'festgesetzt oder erfüllt oder soweit '
                                                                    'infolgedessen Steuervergütungen oder '
                                                                    'Steuererstattungen ohne rechtlichen Grund '
                                                                    'gezahlt werden. Die Haftung umfasst auch '
                                                                    'die infolge der Pflichtverletzung zu '
                                                                    'zahlenden Säumniszuschläge.',
                                                    'source_start': 0,
                                                    'source_end': 447,
                                                    'source_text_sha256': '2d985f33ec184b4388b9639652fc8f5f0b5c7fe780db88feda2f590480cfbd28',
                                                    'explanation': 'Ohne Grundrolle und Steuerbezug ist keine '
                                                                   'positive Zuordnung zu behaupten; '
                                                                   'insufficient_context ist eine '
                                                                   'ausdrückliche diagnostische Erwartung.'}]}},
 {'case_id': 'gs-ownership_negation-anchor',
  'family_id': 'ownership_negation',
  'split': 'development',
  'text': 'Eine Maschine gehört zivilrechtlich der Tal GmbH. Die Nutzerin Wega GmbH kann die Eigentümerin für '
          'die gesamte gewöhnliche Nutzungsdauer wirtschaftlich von der Einwirkung ausschließen. Gefragt ist '
          'die steuerliche Zurechnung der Maschine.',
  'prepared_norm_ids': ['norm:AO:39', 'norm:AO:34', 'norm:AO:69'],
  'diagnostic_labels': {'annotation_status': 'assistant_reviewed_synthetic',
                        'variant': 'anchor',
                        'contrast_dimension': 'negated_rule_condition',
                        'hint_expected': True,
                        'target_norm_ids': ['norm:AO:39'],
                        'allowed_hint_norm_ids': [],
                        'no_hint_norm_ids': ['norm:AO:34', 'norm:AO:69'],
                        'insufficient_context_norm_ids': [],
                        'expected_open_conditions': ['Die tatsächliche Herrschaft und die Verlässlichkeit der '
                                                     'beschriebenen Ausschlussmöglichkeit sind im konkreten '
                                                     'Verhältnis zu prüfen.'],
                        'prohibited_factual_assertions': ['Mit einer möglichen wirtschaftlichen Zurechnung ist '
                                                          'zugleich das zivilrechtliche Eigentum übertragen.',
                                                          'Es geht um persönliche Vertreterhaftung statt um '
                                                          'Zurechnung eines Wirtschaftsguts.'],
                        'source_based_rationale': [{'norm_id': 'norm:AO:39',
                                                    'source_quote': '(1) Wirtschaftsgüter sind dem Eigentümer '
                                                                    'zuzurechnen.(2) Abweichend von Absatz 1 '
                                                                    'gelten die folgenden Vorschriften: 1.Übt '
                                                                    'ein anderer als der Eigentümer die '
                                                                    'tatsächliche Herrschaft über ein '
                                                                    'Wirtschaftsgut in der Weise aus, dass er '
                                                                    'den Eigentümer im Regelfall für die '
                                                                    'gewöhnliche Nutzungsdauer von der '
                                                                    'Einwirkung auf das Wirtschaftsgut '
                                                                    'wirtschaftlich ausschließen kann, so ist '
                                                                    'ihm das Wirtschaftsgut zuzurechnen. ',
                                                    'source_start': 0,
                                                    'source_end': 412,
                                                    'source_text_sha256': '0871e2da8c9c5e86144344e905ec9469a4557c07212ba09e52686467d360edf5',
                                                    'explanation': 'Die Grundregel knüpft an Eigentum an; die '
                                                                   'alternative wirtschaftliche Zurechnung '
                                                                   'verlangt tatsächliche Herrschaft und '
                                                                   'wirtschaftlichen Ausschluss für die '
                                                                   'gewöhnliche Nutzungsdauer.'}]}},
 {'case_id': 'gs-ownership_negation-contrast',
  'family_id': 'ownership_negation',
  'split': 'development',
  'text': 'Eine Maschine gehört zivilrechtlich der Tal GmbH. Die Nutzerin Wega GmbH kann die Eigentümerin '
          'nicht für die gewöhnliche Nutzungsdauer wirtschaftlich von der Einwirkung ausschließen; die '
          'Eigentümerin darf jederzeit eingreifen. Gefragt ist die steuerliche Zurechnung der Maschine.',
  'prepared_norm_ids': ['norm:AO:39', 'norm:AO:34', 'norm:AO:69'],
  'diagnostic_labels': {'annotation_status': 'assistant_reviewed_synthetic',
                        'variant': 'contrast',
                        'contrast_dimension': 'negated_rule_condition',
                        'hint_expected': True,
                        'target_norm_ids': ['norm:AO:39'],
                        'allowed_hint_norm_ids': [],
                        'no_hint_norm_ids': ['norm:AO:34', 'norm:AO:69'],
                        'insufficient_context_norm_ids': [],
                        'expected_open_conditions': ['Die Zurechnung ist anhand von Grundregel und '
                                                     'gegebenenfalls weiteren belegten Alternativen zu prüfen; '
                                                     'die ausdrücklich negierte Ausschlussmöglichkeit darf '
                                                     'nicht als gegeben behandelt werden.'],
                        'prohibited_factual_assertions': ['Die Nutzerin kann die Eigentümerin für die '
                                                          'gewöhnliche Nutzungsdauer ausschließen.',
                                                          'Die wirtschaftliche Zurechnung an die Nutzerin ist '
                                                          'allein wegen ihrer Nutzung nachgewiesen.',
                                                          'Der Hinweis auf die Zurechnungsnorm ist unnütz, '
                                                          'weil eine einzelne Voraussetzung verneint wird.'],
                        'source_based_rationale': [{'norm_id': 'norm:AO:39',
                                                    'source_quote': '(1) Wirtschaftsgüter sind dem Eigentümer '
                                                                    'zuzurechnen.(2) Abweichend von Absatz 1 '
                                                                    'gelten die folgenden Vorschriften: 1.Übt '
                                                                    'ein anderer als der Eigentümer die '
                                                                    'tatsächliche Herrschaft über ein '
                                                                    'Wirtschaftsgut in der Weise aus, dass er '
                                                                    'den Eigentümer im Regelfall für die '
                                                                    'gewöhnliche Nutzungsdauer von der '
                                                                    'Einwirkung auf das Wirtschaftsgut '
                                                                    'wirtschaftlich ausschließen kann, so ist '
                                                                    'ihm das Wirtschaftsgut zuzurechnen. Bei '
                                                                    'Treuhandverhältnissen sind die '
                                                                    'Wirtschaftsgüter dem Treugeber, beim '
                                                                    'Sicherungseigentum dem Sicherungsgeber '
                                                                    'und beim Eigenbesitz dem Eigenbesitzer '
                                                                    'zuzurechnen.2.Wirtschaftsgüter, die '
                                                                    'mehreren zur gesamten Hand oder einer '
                                                                    'rechtsfähigen Personengesellschaft '
                                                                    'zustehen, werden den Beteiligten oder '
                                                                    'Gesellschaftern anteilig zugerechnet, '
                                                                    'soweit eine getrennte Zurechnung für die '
                                                                    'Besteuerung erforderlich ist. '
                                                                    'Rechtsfähige Personengesellschaften '
                                                                    'gelten für Zwecke der Ertragsbesteuerung '
                                                                    'als Gesamthand und deren Vermögen als '
                                                                    'Gesamthandsvermögen.',
                                                    'source_start': 0,
                                                    'source_end': 953,
                                                    'source_text_sha256': '0871e2da8c9c5e86144344e905ec9469a4557c07212ba09e52686467d360edf5',
                                                    'explanation': 'Die Grundregel knüpft an Eigentum an; die '
                                                                   'alternative wirtschaftliche Zurechnung '
                                                                   'verlangt tatsächliche Herrschaft und '
                                                                   'wirtschaftlichen Ausschluss für die '
                                                                   'gewöhnliche Nutzungsdauer.'}]}},
 {'case_id': 'gs-new_facts_context-anchor',
  'family_id': 'new_facts_context',
  'split': 'evaluation',
  'text': 'Ein bestandskräftiger Einkommensteuerbescheid liegt vor. Dem Finanzamt wird später eine zuvor '
          'unbekannte Tatsache bekannt, die die Steuer erhöhen könnte. Unklar ist, welche Korrekturmöglichkeit '
          'in Betracht kommt.',
  'prepared_norm_ids': ['norm:AO:173', 'norm:AO:129', 'norm:AO:173a'],
  'diagnostic_labels': {'annotation_status': 'assistant_reviewed_synthetic',
                        'variant': 'anchor',
                        'contrast_dimension': 'missing_legal_process',
                        'hint_expected': True,
                        'target_norm_ids': ['norm:AO:173'],
                        'allowed_hint_norm_ids': [],
                        'no_hint_norm_ids': ['norm:AO:129', 'norm:AO:173a'],
                        'insufficient_context_norm_ids': [],
                        'expected_open_conditions': ['Nachträgliches Bekanntwerden, tatsächliche Steuerwirkung '
                                                     'und etwaige besondere Änderungshindernisse sind zu '
                                                     'prüfen.'],
                        'prohibited_factual_assertions': ['Der Sachverhalt belegt einen bloßen Schreib- oder '
                                                          'Rechenfehler.',
                                                          'Grobes Verschulden ist unterschiedslos '
                                                          'Voraussetzung beider Änderungsrichtungen.',
                                                          'Die Änderung ist schon ohne Prüfung der '
                                                          'Steuerwirkung sicher zulässig.'],
                        'source_based_rationale': [{'norm_id': 'norm:AO:173',
                                                    'source_quote': '(1) Steuerbescheide sind aufzuheben oder '
                                                                    'zu ändern, 1.soweit Tatsachen oder '
                                                                    'Beweismittel nachträglich bekannt werden, '
                                                                    'die zu einer höheren Steuer '
                                                                    'führen,2.soweit Tatsachen oder '
                                                                    'Beweismittel nachträglich bekannt werden, '
                                                                    'die zu einer niedrigeren Steuer führen '
                                                                    'und den Steuerpflichtigen kein grobes '
                                                                    'Verschulden daran trifft, dass die '
                                                                    'Tatsachen oder Beweismittel erst '
                                                                    'nachträglich bekannt werden. Das '
                                                                    'Verschulden ist unbeachtlich, wenn die '
                                                                    'Tatsachen oder Beweismittel in einem '
                                                                    'unmittelbaren oder mittelbaren '
                                                                    'Zusammenhang mit Tatsachen oder '
                                                                    'Beweismitteln im Sinne der Nummer 1 '
                                                                    'stehen.(2) Abweichend von Absatz 1 können '
                                                                    'Steuerbescheide, soweit sie auf Grund '
                                                                    'einer Außenprüfung ergangen sind, nur '
                                                                    'aufgehoben oder geändert werden, wenn '
                                                                    'eine Steuerhinterziehung oder eine '
                                                                    'leichtfertige Steuerverkürzung vorliegt. '
                                                                    'Dies gilt auch in den Fällen, in denen '
                                                                    'eine Mitteilung nach § 202 Absatz 1 Satz '
                                                                    '3 ergangen ist.',
                                                    'source_start': 0,
                                                    'source_end': 899,
                                                    'source_text_sha256': '16e15f09402e76f99178815e8b87ad853cf0765a87463d2ae0b3899e77516049',
                                                    'explanation': 'Die Norm trennt höhere und niedrigere '
                                                                   'Steuer sowie weitere Einschränkungen. Ein '
                                                                   'einschlägiger Hinweis ist keine '
                                                                   'abschließende Änderungsentscheidung.'}]}},
 {'case_id': 'gs-new_facts_context-contrast',
  'family_id': 'new_facts_context',
  'split': 'evaluation',
  'text': 'Ein abgeschlossener interner Projektbericht liegt vor. Dem Organisationsteam wird später eine zuvor '
          'unbekannte Tatsache über den Veranstaltungsablauf bekannt. Ein Steuerbescheid oder ein steuerlicher '
          'Vorgang ist damit nicht verbunden.',
  'prepared_norm_ids': ['norm:AO:173', 'norm:AO:129', 'norm:AO:173a'],
  'diagnostic_labels': {'annotation_status': 'assistant_reviewed_synthetic',
                        'variant': 'contrast',
                        'contrast_dimension': 'missing_legal_process',
                        'hint_expected': False,
                        'target_norm_ids': [],
                        'allowed_hint_norm_ids': [],
                        'no_hint_norm_ids': ['norm:AO:173', 'norm:AO:129', 'norm:AO:173a'],
                        'insufficient_context_norm_ids': [],
                        'expected_open_conditions': [],
                        'prohibited_factual_assertions': ['Das Organisationsteam ist die Finanzbehörde oder '
                                                          'ändert einen Steuerbescheid.',
                                                          'Die neu bekannt gewordene Tatsache führt zu einer '
                                                          'höheren oder niedrigeren Steuer.'],
                        'source_based_rationale': [{'norm_id': 'norm:AO:173',
                                                    'source_quote': '(1) Steuerbescheide sind aufzuheben oder '
                                                                    'zu ändern, 1.soweit Tatsachen oder '
                                                                    'Beweismittel nachträglich bekannt werden, '
                                                                    'die zu einer höheren Steuer '
                                                                    'führen,2.soweit Tatsachen oder '
                                                                    'Beweismittel nachträglich bekannt werden, '
                                                                    'die zu einer niedrigeren Steuer führen '
                                                                    'und den Steuerpflichtigen kein grobes '
                                                                    'Verschulden daran trifft, dass die '
                                                                    'Tatsachen oder Beweismittel erst '
                                                                    'nachträglich bekannt werden. Das '
                                                                    'Verschulden ist unbeachtlich, wenn die '
                                                                    'Tatsachen oder Beweismittel in einem '
                                                                    'unmittelbaren oder mittelbaren '
                                                                    'Zusammenhang mit Tatsachen oder '
                                                                    'Beweismitteln im Sinne der Nummer 1 '
                                                                    'stehen.',
                                                    'source_start': 0,
                                                    'source_end': 579,
                                                    'source_text_sha256': '16e15f09402e76f99178815e8b87ad853cf0765a87463d2ae0b3899e77516049',
                                                    'explanation': 'Allgemeines späteres Wissen über einen '
                                                                   'ausdrücklich nichtsteuerlichen Bericht '
                                                                   'belegt den steuerlichen Grundvorgang '
                                                                   'nicht.'},
                                                   {'norm_id': 'norm:AO:129',
                                                    'source_quote': 'Die Finanzbehörde kann Schreibfehler, '
                                                                    'Rechenfehler und ähnliche offenbare '
                                                                    'Unrichtigkeiten, die beim Erlass eines '
                                                                    'Verwaltungsakts unterlaufen sind, '
                                                                    'jederzeit berichtigen. Bei berechtigtem '
                                                                    'Interesse des Beteiligten ist zu '
                                                                    'berichtigen. Wird zu einem schriftlich '
                                                                    'ergangenen Verwaltungsakt die '
                                                                    'Berichtigung begehrt, ist die '
                                                                    'Finanzbehörde berechtigt, die Vorlage des '
                                                                    'Schriftstücks zu verlangen, das '
                                                                    'berichtigt werden soll.',
                                                    'source_start': 0,
                                                    'source_end': 416,
                                                    'source_text_sha256': '31099d1788c3a21475460eb1a2d914d3159cb1275c54e974780167eba266a669',
                                                    'explanation': 'Die Regel betrifft Fehler beim Erlass '
                                                                   'eines Verwaltungsakts, nicht bloß interne '
                                                                   'Projektberichte.'}]}},
 {'case_id': 'gs-correction_origin-anchor',
  'family_id': 'correction_origin',
  'split': 'evaluation',
  'text': 'Die Finanzbehörde hat beim Erlass eines Steuerbescheids selbst eine Ziffer vertauscht. Die '
          'abgegebene Steuererklärung war insoweit richtig. Es geht um die Korrektur dieses Schreibfehlers im '
          'Bescheid.',
  'prepared_norm_ids': ['norm:AO:129', 'norm:AO:173a', 'norm:AO:173'],
  'diagnostic_labels': {'annotation_status': 'assistant_reviewed_synthetic',
                        'variant': 'anchor',
                        'contrast_dimension': 'error_origin',
                        'hint_expected': True,
                        'target_norm_ids': ['norm:AO:129'],
                        'allowed_hint_norm_ids': [],
                        'no_hint_norm_ids': ['norm:AO:173a', 'norm:AO:173'],
                        'insufficient_context_norm_ids': [],
                        'expected_open_conditions': ['Ob die Unrichtigkeit im konkreten Fall offenbar ist und '
                                                     'tatsächlich beim Erlass unterlief, bleibt zu prüfen.'],
                        'prohibited_factual_assertions': ['Die steuerpflichtige Person hat die falsche Ziffer '
                                                          'schon bei Erstellung ihrer Erklärung eingetragen.',
                                                          'Es sind nachträglich neue Tatsachen statt eines '
                                                          'beschriebenen Schreibfehlers bekannt geworden.'],
                        'source_based_rationale': [{'norm_id': 'norm:AO:129',
                                                    'source_quote': 'Die Finanzbehörde kann Schreibfehler, '
                                                                    'Rechenfehler und ähnliche offenbare '
                                                                    'Unrichtigkeiten, die beim Erlass eines '
                                                                    'Verwaltungsakts unterlaufen sind, '
                                                                    'jederzeit berichtigen. Bei berechtigtem '
                                                                    'Interesse des Beteiligten ist zu '
                                                                    'berichtigen. Wird zu einem schriftlich '
                                                                    'ergangenen Verwaltungsakt die '
                                                                    'Berichtigung begehrt, ist die '
                                                                    'Finanzbehörde berechtigt, die Vorlage des '
                                                                    'Schriftstücks zu verlangen, das '
                                                                    'berichtigt werden soll.',
                                                    'source_start': 0,
                                                    'source_end': 416,
                                                    'source_text_sha256': '31099d1788c3a21475460eb1a2d914d3159cb1275c54e974780167eba266a669',
                                                    'explanation': 'Der ausdrücklich geschilderte '
                                                                   'Fehlerursprung ist der Erlass des '
                                                                   'Verwaltungsakts.'},
                                                   {'norm_id': 'norm:AO:173a',
                                                    'source_quote': 'Steuerbescheide sind aufzuheben oder zu '
                                                                    'ändern, soweit dem Steuerpflichtigen bei '
                                                                    'Erstellung seiner Steuererklärung '
                                                                    'Schreib- oder Rechenfehler unterlaufen '
                                                                    'sind und er deshalb der Finanzbehörde '
                                                                    'bestimmte, nach den Verhältnissen zum '
                                                                    'Zeitpunkt des Erlasses des '
                                                                    'Steuerbescheids rechtserhebliche '
                                                                    'Tatsachen unzutreffend mitgeteilt hat.',
                                                    'source_start': 0,
                                                    'source_end': 328,
                                                    'source_text_sha256': '262d03e180947c0398a3e93138891c21f4b04421352ac74ce5b20f20bc0c9f5b',
                                                    'explanation': 'Der alternative Erklärungstatbestand passt '
                                                                   'nicht zur ausdrücklich insoweit richtigen '
                                                                   'Erklärung.'}]}},
 {'case_id': 'gs-correction_origin-contrast',
  'family_id': 'correction_origin',
  'split': 'evaluation',
  'text': 'Die steuerpflichtige Person hat bei Erstellung ihrer Steuererklärung selbst eine Ziffer vertauscht '
          'und dadurch eine rechtserhebliche Tatsache falsch mitgeteilt. Der darauf beruhende Steuerbescheid '
          'liegt vor. Es geht um die Korrektur dieses Erklärungsfehlers.',
  'prepared_norm_ids': ['norm:AO:129', 'norm:AO:173a', 'norm:AO:173'],
  'diagnostic_labels': {'annotation_status': 'assistant_reviewed_synthetic',
                        'variant': 'contrast',
                        'contrast_dimension': 'error_origin',
                        'hint_expected': True,
                        'target_norm_ids': ['norm:AO:173a'],
                        'allowed_hint_norm_ids': ['norm:AO:129'],
                        'no_hint_norm_ids': ['norm:AO:173'],
                        'insufficient_context_norm_ids': [],
                        'expected_open_conditions': ['Zu prüfen sind Zusammenhang des Schreibfehlers mit der '
                                                     'unzutreffenden Tatsachenmitteilung und deren '
                                                     'Rechtserheblichkeit beim Bescheiderlass.',
                                                     'Ob zusätzlich eine offenbare Unrichtigkeit beim Erlass '
                                                     'vorliegt, folgt nicht allein aus der beschriebenen '
                                                     'Fehlerübernahme.'],
                        'prohibited_factual_assertions': ['Die Finanzbehörde hat die Ziffer selbst erstmals '
                                                          'beim Erlass vertauscht.',
                                                          'Ein Schreibfehler und seine Rechtserheblichkeit '
                                                          'führen ohne weitere Prüfung automatisch zur '
                                                          'erfolgreichen Korrektur.'],
                        'source_based_rationale': [{'norm_id': 'norm:AO:173a',
                                                    'source_quote': 'Steuerbescheide sind aufzuheben oder zu '
                                                                    'ändern, soweit dem Steuerpflichtigen bei '
                                                                    'Erstellung seiner Steuererklärung '
                                                                    'Schreib- oder Rechenfehler unterlaufen '
                                                                    'sind und er deshalb der Finanzbehörde '
                                                                    'bestimmte, nach den Verhältnissen zum '
                                                                    'Zeitpunkt des Erlasses des '
                                                                    'Steuerbescheids rechtserhebliche '
                                                                    'Tatsachen unzutreffend mitgeteilt hat.',
                                                    'source_start': 0,
                                                    'source_end': 328,
                                                    'source_text_sha256': '262d03e180947c0398a3e93138891c21f4b04421352ac74ce5b20f20bc0c9f5b',
                                                    'explanation': 'Der Ursprung in der Erklärung ist '
                                                                   'ausdrücklich genannt. Die Erlasskorrektur '
                                                                   'wird nicht pauschal als ausgeschlossen '
                                                                   'gelabelt, weil ihr Verhältnis zusätzliche '
                                                                   'Prüfung verlangen kann.'},
                                                   {'norm_id': 'norm:AO:129',
                                                    'source_quote': 'Die Finanzbehörde kann Schreibfehler, '
                                                                    'Rechenfehler und ähnliche offenbare '
                                                                    'Unrichtigkeiten, die beim Erlass eines '
                                                                    'Verwaltungsakts unterlaufen sind, '
                                                                    'jederzeit berichtigen. Bei berechtigtem '
                                                                    'Interesse des Beteiligten ist zu '
                                                                    'berichtigen. Wird zu einem schriftlich '
                                                                    'ergangenen Verwaltungsakt die '
                                                                    'Berichtigung begehrt, ist die '
                                                                    'Finanzbehörde berechtigt, die Vorlage des '
                                                                    'Schriftstücks zu verlangen, das '
                                                                    'berichtigt werden soll.',
                                                    'source_start': 0,
                                                    'source_end': 416,
                                                    'source_text_sha256': '31099d1788c3a21475460eb1a2d914d3159cb1275c54e974780167eba266a669',
                                                    'explanation': 'Ein zusätzlicher Hinweis ist nur mit '
                                                                   'offenem Erlass-/Offenbarkeitsbezug '
                                                                   'erlaubt; die Fehlerquelle darf dabei nicht '
                                                                   'umgeschrieben werden.'}]}},
 {'case_id': 'gs-vat_group_scope-anchor',
  'family_id': 'vat_group_scope',
  'split': 'evaluation',
  'text': 'Für die Umsatzsteuer soll die Stellung einer Tochter-AG geprüft werden. Sie ist nach der '
          'Beschreibung finanziell, wirtschaftlich und organisatorisch in das Unternehmen der '
          'Muttergesellschaft eingegliedert. Ein Gewinnabführungsvertrag wird nicht behauptet.',
  'prepared_norm_ids': ['norm:UStG:2', 'norm:KStG:14', 'norm:AO:39'],
  'diagnostic_labels': {'annotation_status': 'assistant_reviewed_synthetic',
                        'variant': 'anchor',
                        'contrast_dimension': 'entity_and_tax_scope',
                        'hint_expected': True,
                        'target_norm_ids': ['norm:UStG:2'],
                        'allowed_hint_norm_ids': [],
                        'no_hint_norm_ids': ['norm:KStG:14', 'norm:AO:39'],
                        'insufficient_context_norm_ids': [],
                        'expected_open_conditions': ['Die tatsächliche Eingliederung und der Umfang der auf '
                                                     'inländische Unternehmensteile begrenzten Wirkungen sind '
                                                     'zu prüfen.'],
                        'prohibited_factual_assertions': ['Ein Gewinnabführungsvertrag oder eine '
                                                          'körperschaftsteuerliche Organschaft ist '
                                                          'nachgewiesen.',
                                                          'Jede organisatorische Zusammenarbeit ist bereits '
                                                          'eine abschließend festgestellte umsatzsteuerliche '
                                                          'Organschaft.'],
                        'source_based_rationale': [{'norm_id': 'norm:UStG:2',
                                                    'source_quote': '(2) Die gewerbliche oder berufliche '
                                                                    'Tätigkeit wird nicht selbständig '
                                                                    'ausgeübt, 1.soweit natürliche Personen, '
                                                                    'einzeln oder zusammengeschlossen, einem '
                                                                    'Unternehmen so eingegliedert sind, dass '
                                                                    'sie den Weisungen des Unternehmers zu '
                                                                    'folgen verpflichtet sind,2.wenn eine '
                                                                    'juristische Person nach dem Gesamtbild '
                                                                    'der tatsächlichen Verhältnisse '
                                                                    'finanziell, wirtschaftlich und '
                                                                    'organisatorisch in das Unternehmen des '
                                                                    'Organträgers eingegliedert ist '
                                                                    '(Organschaft). Die Wirkungen der '
                                                                    'Organschaft sind auf Innenleistungen '
                                                                    'zwischen den im Inland gelegenen '
                                                                    'Unternehmensteilen beschränkt. Diese '
                                                                    'Unternehmensteile sind als ein '
                                                                    'Unternehmen zu behandeln. Hat der '
                                                                    'Organträger seine Geschäftsleitung im '
                                                                    'Ausland, gilt der wirtschaftlich '
                                                                    'bedeutendste Unternehmensteil im Inland '
                                                                    'als der Unternehmer.',
                                                    'source_start': 457,
                                                    'source_end': 1228,
                                                    'source_text_sha256': '794f8ac9e6990836ff2eeda319ed7602955150f6d99c08c5d1dca5d67b9faedb',
                                                    'explanation': 'Die umsatzsteuerliche Organschaft knüpft '
                                                                   'an eine juristische Person und '
                                                                   'finanzielle, wirtschaftliche sowie '
                                                                   'organisatorische Eingliederung nach den '
                                                                   'tatsächlichen Verhältnissen an.'},
                                                   {'norm_id': 'norm:KStG:14',
                                                    'source_quote': '(1) 1Verpflichtet sich eine Europäische '
                                                                    'Gesellschaft, Aktiengesellschaft oder '
                                                                    'Kommanditgesellschaft auf Aktien mit '
                                                                    'Geschäftsleitung im Inland und Sitz in '
                                                                    'einem Mitgliedstaat der Europäischen '
                                                                    'Union oder in einem Vertragsstaat des '
                                                                    'EWR-Abkommens (Organgesellschaft) durch '
                                                                    'einen Gewinnabführungsvertrag im Sinne '
                                                                    'des § 291 Abs. 1 des Aktiengesetzes, '
                                                                    'ihren ganzen Gewinn an ein einziges '
                                                                    'anderes gewerbliches Unternehmen '
                                                                    'abzuführen, ist das Einkommen der '
                                                                    'Organgesellschaft, soweit sich aus § 16 '
                                                                    'nichts anderes ergibt, dem Träger des '
                                                                    'Unternehmens (Organträger) zuzurechnen, '
                                                                    'wenn die folgenden Voraussetzungen '
                                                                    'erfüllt sind: ',
                                                    'source_start': 0,
                                                    'source_end': 615,
                                                    'source_text_sha256': '43af895c1fcc59dc5e6ae260ae339b0c2a1cfb1250925ec609ee1efb75a5388e',
                                                    'explanation': 'Die andere Organschaftsregel setzt unter '
                                                                   'anderem einen Gewinnabführungsvertrag '
                                                                   'voraus; die explizite Frage betrifft hier '
                                                                   'Umsatzsteuer.'}]}},
 {'case_id': 'gs-vat_group_scope-contrast',
  'family_id': 'vat_group_scope',
  'split': 'evaluation',
  'text': 'Für eine interne Projektplanung werden drei Abteilungen desselben Unternehmens finanziell, '
          'wirtschaftlich und organisatorisch aufeinander abgestimmt. Es geht weder um selbständige '
          'Unternehmen noch um deren steuerliche Behandlung.',
  'prepared_norm_ids': ['norm:UStG:2', 'norm:KStG:14', 'norm:AO:39'],
  'diagnostic_labels': {'annotation_status': 'assistant_reviewed_synthetic',
                        'variant': 'contrast',
                        'contrast_dimension': 'entity_and_tax_scope',
                        'hint_expected': False,
                        'target_norm_ids': [],
                        'allowed_hint_norm_ids': [],
                        'no_hint_norm_ids': ['norm:UStG:2', 'norm:KStG:14', 'norm:AO:39'],
                        'insufficient_context_norm_ids': [],
                        'expected_open_conditions': [],
                        'prohibited_factual_assertions': ['Die Abteilungen sind eigenständige juristische '
                                                          'Personen oder Mutter- und Tochtergesellschaften.',
                                                          'Es liegt eine steuerliche Organschaft oder ein '
                                                          'Gewinnabführungsvertrag vor.'],
                        'source_based_rationale': [{'norm_id': 'norm:UStG:2',
                                                    'source_quote': '(2) Die gewerbliche oder berufliche '
                                                                    'Tätigkeit wird nicht selbständig '
                                                                    'ausgeübt, 1.soweit natürliche Personen, '
                                                                    'einzeln oder zusammengeschlossen, einem '
                                                                    'Unternehmen so eingegliedert sind, dass '
                                                                    'sie den Weisungen des Unternehmers zu '
                                                                    'folgen verpflichtet sind,2.wenn eine '
                                                                    'juristische Person nach dem Gesamtbild '
                                                                    'der tatsächlichen Verhältnisse '
                                                                    'finanziell, wirtschaftlich und '
                                                                    'organisatorisch in das Unternehmen des '
                                                                    'Organträgers eingegliedert ist '
                                                                    '(Organschaft). Die Wirkungen der '
                                                                    'Organschaft sind auf Innenleistungen '
                                                                    'zwischen den im Inland gelegenen '
                                                                    'Unternehmensteilen beschränkt. Diese '
                                                                    'Unternehmensteile sind als ein '
                                                                    'Unternehmen zu behandeln. Hat der '
                                                                    'Organträger seine Geschäftsleitung im '
                                                                    'Ausland, gilt der wirtschaftlich '
                                                                    'bedeutendste Unternehmensteil im Inland '
                                                                    'als der Unternehmer.',
                                                    'source_start': 457,
                                                    'source_end': 1228,
                                                    'source_text_sha256': '794f8ac9e6990836ff2eeda319ed7602955150f6d99c08c5d1dca5d67b9faedb',
                                                    'explanation': 'Die ausdrücklich interne '
                                                                   'Abteilungsorganisation stellt die für '
                                                                   'diesen Hinweis nötige rechtlich '
                                                                   'eigenständige Person und steuerliche Frage '
                                                                   'nicht her.'},
                                                   {'norm_id': 'norm:KStG:14',
                                                    'source_quote': '(1) 1Verpflichtet sich eine Europäische '
                                                                    'Gesellschaft, Aktiengesellschaft oder '
                                                                    'Kommanditgesellschaft auf Aktien mit '
                                                                    'Geschäftsleitung im Inland und Sitz in '
                                                                    'einem Mitgliedstaat der Europäischen '
                                                                    'Union oder in einem Vertragsstaat des '
                                                                    'EWR-Abkommens (Organgesellschaft) durch '
                                                                    'einen Gewinnabführungsvertrag im Sinne '
                                                                    'des § 291 Abs. 1 des Aktiengesetzes, '
                                                                    'ihren ganzen Gewinn an ein einziges '
                                                                    'anderes gewerbliches Unternehmen '
                                                                    'abzuführen, ist das Einkommen der '
                                                                    'Organgesellschaft, soweit sich aus § 16 '
                                                                    'nichts anderes ergibt, dem Träger des '
                                                                    'Unternehmens (Organträger) zuzurechnen, '
                                                                    'wenn die folgenden Voraussetzungen '
                                                                    'erfüllt sind: ',
                                                    'source_start': 0,
                                                    'source_end': 615,
                                                    'source_text_sha256': '43af895c1fcc59dc5e6ae260ae339b0c2a1cfb1250925ec609ee1efb75a5388e',
                                                    'explanation': 'Interne Zusammenarbeit ist kein belegter '
                                                                   'Gewinnabführungsvertrag zwischen den hier '
                                                                   'vorausgesetzten Rechtsträgern.'}]}},
 {'case_id': 'gs-foreign_substance_scope-anchor',
  'family_id': 'foreign_substance_scope',
  'split': 'evaluation',
  'text': 'Eine in Deutschland unbeschränkt steuerpflichtige Muttergesellschaft beherrscht eine Gesellschaft '
          'mit Sitz und Geschäftsleitung in einem anderen EU-Mitgliedstaat. Diese hat eigene Räume und '
          'qualifiziertes Personal. Es wird geprüft, ob ihr Tätigkeitsnachweis eine Hinzurechnung ausschließen '
          'kann; Art und Besteuerung der Einkünfte sind noch ungeklärt.',
  'prepared_norm_ids': ['norm:AStG:8', 'norm:AStG:7', 'norm:AO:42'],
  'diagnostic_labels': {'annotation_status': 'assistant_reviewed_synthetic',
                        'variant': 'anchor',
                        'contrast_dimension': 'geographic_exception_scope',
                        'hint_expected': True,
                        'target_norm_ids': ['norm:AStG:8'],
                        'allowed_hint_norm_ids': ['norm:AStG:7'],
                        'no_hint_norm_ids': ['norm:AO:42'],
                        'insufficient_context_norm_ids': [],
                        'expected_open_conditions': ['Niedrige Besteuerung, Art der Einkünfte, tatsächliche '
                                                     'wesentliche wirtschaftliche Tätigkeit, '
                                                     'Einkünftezuordnung sowie die Auskunftsvoraussetzung sind '
                                                     'nicht vollständig nachgewiesen.'],
                        'prohibited_factual_assertions': ['Eigene Räume und Personal schließen die '
                                                          'Hinzurechnung automatisch für alle Einkünfte aus.',
                                                          'Eine niedrige Besteuerung oder ein '
                                                          'Gestaltungsmissbrauch ist bereits festgestellt.',
                                                          'Die Tätigkeitsregel des Absatzes 2 ist bloß eine '
                                                          'positive Definition passiver Einkünfte und enthält '
                                                          'keinen Ausnahmemechanismus.'],
                        'source_based_rationale': [{'norm_id': 'norm:AStG:8',
                                                    'source_quote': '(2) Ungeachtet des Absatzes 1 ist eine '
                                                                    'ausländische Gesellschaft nicht '
                                                                    'Zwischengesellschaft für Einkünfte, für '
                                                                    'die nachgewiesen wird, dass die '
                                                                    'Gesellschaft in dem Staat, in dem sie '
                                                                    'ihren Sitz oder ihre Geschäftsleitung '
                                                                    'hat, insoweit einer wesentlichen '
                                                                    'wirtschaftlichen Tätigkeit nachgeht. Dies '
                                                                    'setzt insbesondere den Einsatz der für '
                                                                    'die Ausübung der Tätigkeit erforderlichen '
                                                                    'sachlichen und personellen Ausstattung in '
                                                                    'diesem Staat voraus. Die Tätigkeit muss '
                                                                    'durch hinreichend qualifiziertes Personal '
                                                                    'selbständig und eigenverantwortlich '
                                                                    'ausgeübt werden. Der wesentlichen '
                                                                    'wirtschaftlichen Tätigkeit der '
                                                                    'Gesellschaft sind nur Einkünfte der '
                                                                    'Gesellschaft zuzuordnen, die durch diese '
                                                                    'Tätigkeit erzielt werden und dies nur '
                                                                    'insoweit, als der '
                                                                    'Fremdvergleichsgrundsatz (§ 1) beachtet '
                                                                    'worden ist. Die Sätze 1 bis 3 gelten '
                                                                    'nicht, wenn die Gesellschaft ihre '
                                                                    'wesentliche wirtschaftliche Tätigkeit '
                                                                    'überwiegend durch Dritte besorgen '
                                                                    'lässt.(3) Absatz 2 gilt nur, wenn die '
                                                                    'ausländische Gesellschaft ihren Sitz oder '
                                                                    'ihre Geschäftsleitung in einem '
                                                                    'Mitgliedstaat der Europäischen Union oder '
                                                                    'einem Vertragsstaat des EWR-Abkommens '
                                                                    'hat.(4) Die Absätze 2 und 3 gelten nicht, '
                                                                    'wenn der Staat, in dem die Gesellschaft '
                                                                    'ihren Sitz oder ihre Geschäftsleitung '
                                                                    'hat, im Wege des zwischenstaatlichen '
                                                                    'Informationsaustausches keine Auskünfte '
                                                                    'erteilt, die zur Durchführung der '
                                                                    'Besteuerung erforderlich sind.',
                                                    'source_start': 6488,
                                                    'source_end': 7856,
                                                    'source_text_sha256': '817b0f4e4be475bb3b2b56623bc7851c87dfc23f0c148234326fbb8d1d7684e3',
                                                    'explanation': 'Der Tätigkeitsnachweis ist ein '
                                                                   'Ausnahmemechanismus zu Absatz 1. Absatz 3 '
                                                                   'begrenzt ihn auf Sitz oder '
                                                                   'Geschäftsleitung in EU/EWR; Absatz 4 '
                                                                   'enthält eine weitere Auskunftsgrenze.'},
                                                   {'norm_id': 'norm:AStG:7',
                                                    'source_quote': '(1) Beherrscht ein unbeschränkt '
                                                                    'Steuerpflichtiger eine Körperschaft, '
                                                                    'Personenvereinigung oder Vermögensmasse '
                                                                    'im Sinne des Körperschaftsteuergesetzes, '
                                                                    'die weder Geschäftsleitung noch Sitz im '
                                                                    'Inland hat und die nicht gemäß § 3 Absatz '
                                                                    '1 des Körperschaftsteuergesetzes von der '
                                                                    'Körperschaftsteuerpflicht ausgenommen ist '
                                                                    '(ausländische Gesellschaft), sind die '
                                                                    'Einkünfte, für die diese Gesellschaft '
                                                                    'Zwischengesellschaft ist, bei dem '
                                                                    'unbeschränkt Steuerpflichtigen '
                                                                    'entsprechend seiner unmittelbaren und '
                                                                    'mittelbaren Beteiligung am Nennkapital '
                                                                    'steuerpflichtig. ',
                                                    'source_start': 0,
                                                    'source_end': 550,
                                                    'source_text_sha256': '76dcb64bd6d8bb7c5f73412b7c5baf8ba65c59d8b15f0880e88ca40d993b7a7c',
                                                    'explanation': 'Die Beherrschung und Auslandsrolle sind '
                                                                   'ausdrücklich genannt; die weitere '
                                                                   'Zurechnung setzt die rechtliche Einordnung '
                                                                   'der Einkünfte voraus.'}]}},
 {'case_id': 'gs-foreign_substance_scope-contrast',
  'family_id': 'foreign_substance_scope',
  'split': 'evaluation',
  'text': 'Eine in Deutschland unbeschränkt steuerpflichtige Muttergesellschaft beherrscht eine Gesellschaft '
          'mit Sitz und Geschäftsleitung in einem Staat außerhalb von EU und EWR. Diese hat eigene Räume und '
          'qualifiziertes Personal. Es wird geprüft, ob ihr Tätigkeitsnachweis eine Hinzurechnung ausschließen '
          'kann; Art und Besteuerung der Einkünfte sind noch ungeklärt.',
  'prepared_norm_ids': ['norm:AStG:8', 'norm:AStG:7', 'norm:AO:42'],
  'diagnostic_labels': {'annotation_status': 'assistant_reviewed_synthetic',
                        'variant': 'contrast',
                        'contrast_dimension': 'geographic_exception_scope',
                        'hint_expected': True,
                        'target_norm_ids': ['norm:AStG:8'],
                        'allowed_hint_norm_ids': ['norm:AStG:7'],
                        'no_hint_norm_ids': ['norm:AO:42'],
                        'insufficient_context_norm_ids': [],
                        'expected_open_conditions': ['Die Einordnung nach Absatz 1 und die Besteuerung der '
                                                     'Einkünfte bleiben zu prüfen; eine Hinzurechnung ist '
                                                     'nicht allein aus dem Drittstaatensitz ableitbar.',
                                                     'Die für Absatz 2 ausdrücklich geregelte EU-/EWR-Grenze '
                                                     'ist eine entgegenstehende Vorgabe, keine noch ungeklärte '
                                                     'Sitzinformation.'],
                        'prohibited_factual_assertions': ['Die Gesellschaft hat Sitz oder Geschäftsleitung in '
                                                          'einem EU- oder EWR-Staat.',
                                                          'Der Tätigkeitsnachweis nach Absatz 2 ist nach '
                                                          'diesem Wortlaut unabhängig von der EU-/EWR-Grenze '
                                                          'anwendbar.',
                                                          'Alle Einkünfte sind allein wegen des '
                                                          'Drittstaatensitzes sicher hinzuzurechnen.'],
                        'source_based_rationale': [{'norm_id': 'norm:AStG:8',
                                                    'source_quote': '(2) Ungeachtet des Absatzes 1 ist eine '
                                                                    'ausländische Gesellschaft nicht '
                                                                    'Zwischengesellschaft für Einkünfte, für '
                                                                    'die nachgewiesen wird, dass die '
                                                                    'Gesellschaft in dem Staat, in dem sie '
                                                                    'ihren Sitz oder ihre Geschäftsleitung '
                                                                    'hat, insoweit einer wesentlichen '
                                                                    'wirtschaftlichen Tätigkeit nachgeht. Dies '
                                                                    'setzt insbesondere den Einsatz der für '
                                                                    'die Ausübung der Tätigkeit erforderlichen '
                                                                    'sachlichen und personellen Ausstattung in '
                                                                    'diesem Staat voraus. Die Tätigkeit muss '
                                                                    'durch hinreichend qualifiziertes Personal '
                                                                    'selbständig und eigenverantwortlich '
                                                                    'ausgeübt werden. Der wesentlichen '
                                                                    'wirtschaftlichen Tätigkeit der '
                                                                    'Gesellschaft sind nur Einkünfte der '
                                                                    'Gesellschaft zuzuordnen, die durch diese '
                                                                    'Tätigkeit erzielt werden und dies nur '
                                                                    'insoweit, als der '
                                                                    'Fremdvergleichsgrundsatz (§ 1) beachtet '
                                                                    'worden ist. Die Sätze 1 bis 3 gelten '
                                                                    'nicht, wenn die Gesellschaft ihre '
                                                                    'wesentliche wirtschaftliche Tätigkeit '
                                                                    'überwiegend durch Dritte besorgen '
                                                                    'lässt.(3) Absatz 2 gilt nur, wenn die '
                                                                    'ausländische Gesellschaft ihren Sitz oder '
                                                                    'ihre Geschäftsleitung in einem '
                                                                    'Mitgliedstaat der Europäischen Union oder '
                                                                    'einem Vertragsstaat des EWR-Abkommens '
                                                                    'hat.(4) Die Absätze 2 und 3 gelten nicht, '
                                                                    'wenn der Staat, in dem die Gesellschaft '
                                                                    'ihren Sitz oder ihre Geschäftsleitung '
                                                                    'hat, im Wege des zwischenstaatlichen '
                                                                    'Informationsaustausches keine Auskünfte '
                                                                    'erteilt, die zur Durchführung der '
                                                                    'Besteuerung erforderlich sind.',
                                                    'source_start': 6488,
                                                    'source_end': 7856,
                                                    'source_text_sha256': '817b0f4e4be475bb3b2b56623bc7851c87dfc23f0c148234326fbb8d1d7684e3',
                                                    'explanation': 'Der Tätigkeitsnachweis ist ein '
                                                                   'Ausnahmemechanismus zu Absatz 1. Absatz 3 '
                                                                   'begrenzt ihn auf Sitz oder '
                                                                   'Geschäftsleitung in EU/EWR; Absatz 4 '
                                                                   'enthält eine weitere Auskunftsgrenze.'},
                                                   {'norm_id': 'norm:AStG:7',
                                                    'source_quote': '(1) Beherrscht ein unbeschränkt '
                                                                    'Steuerpflichtiger eine Körperschaft, '
                                                                    'Personenvereinigung oder Vermögensmasse '
                                                                    'im Sinne des Körperschaftsteuergesetzes, '
                                                                    'die weder Geschäftsleitung noch Sitz im '
                                                                    'Inland hat und die nicht gemäß § 3 Absatz '
                                                                    '1 des Körperschaftsteuergesetzes von der '
                                                                    'Körperschaftsteuerpflicht ausgenommen ist '
                                                                    '(ausländische Gesellschaft), sind die '
                                                                    'Einkünfte, für die diese Gesellschaft '
                                                                    'Zwischengesellschaft ist, bei dem '
                                                                    'unbeschränkt Steuerpflichtigen '
                                                                    'entsprechend seiner unmittelbaren und '
                                                                    'mittelbaren Beteiligung am Nennkapital '
                                                                    'steuerpflichtig. ',
                                                    'source_start': 0,
                                                    'source_end': 550,
                                                    'source_text_sha256': '76dcb64bd6d8bb7c5f73412b7c5baf8ba65c59d8b15f0880e88ca40d993b7a7c',
                                                    'explanation': 'Der Grundvorgang bleibt relevant; aus '
                                                                   'fehlender Reichweite einer Ausnahme folgt '
                                                                   'kein schon erfüllter '
                                                                   'Zurechnungstatbestand.'}]}})


def load_cases(split: str | None = None) -> list[dict[str, Any]]:
    """Return independent evaluator records, optionally filtered by whole-family split."""
    if split is not None and split not in SPLITS:
        raise ValueError("split must be development, evaluation, or None")
    return deepcopy([case for case in _CASES if split is None or case["split"] == split])


def dataset_manifest() -> dict[str, Any]:
    """Describe fixed provenance and limitations without implying legal adjudication."""
    encoded = json.dumps(_CASES, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {
        "format": DATASET_VERSION,
        "annotation_status": ANNOTATION_STATUS,
        "synthetic": True,
        "expert_adjudicated": False,
        "blind_evaluation": False,
        "known_failure_patterns": True,
        "source_graph_sha256": SOURCE_GRAPH_SHA256,
        "source_graph_path": "out/gpu-ab-20260907-final/artifacts/graph.sqlite",
        "dataset_sha256": hashlib.sha256(encoded).hexdigest(),
        "case_count": len(_CASES),
        "family_count": len({case["family_id"] for case in _CASES}),
        "family_splits": {
            split: sorted({case["family_id"] for case in _CASES if case["split"] == split})
            for split in SPLITS
        },
        "case_counts": {split: sum(case["split"] == split for case in _CASES) for split in SPLITS},
        "limits": [
            "Assistant-authored diagnostic annotations are not expert legal gold.",
            "The failure patterns and norm families were known before this synthetic set was written.",
            "Family separation prevents sibling examples crossing splits, not prior topic exposure.",
            "Permitted additional hints are not negatives; insufficient context is separately declared.",
            "Prose checklists require qualitative review and are not automatic legal correctness scores.",
            "Prepared package selection is an evaluator-controlled source-availability arm, not a retrieval result.",
        ],
    }
