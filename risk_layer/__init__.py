"""Risk-Layer – Schlüsselbegriff-Assistenz für die steuerliche Risikosteuerung.

Leitidee: Das System markiert unbestimmte Rechtsbegriffe, an denen subsumiert
werden muss. Es lenkt Aufmerksamkeit, es übernimmt keine Subsumtion.

Schicht 1 (dieser MVP) ist vollständig deterministisch und ohne LLM/GPU lauffähig.
"""

__version__ = "0.1.0"

from .pipeline import analyse, Pipeline  # noqa: E402,F401
