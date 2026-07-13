from __future__ import annotations

import json
from pathlib import Path
import shutil

from risk_layer.cli import main


def _mini_korpus(tmp_path: Path) -> Path:
    corpus = tmp_path / "corpus"
    gii = corpus / "gii"
    gii.mkdir(parents=True)
    shutil.copyfile(Path("tests/fixtures/mini_gii.xml"), gii / "EStG.xml")
    return corpus


def test_graph_norm_manifest_mit_source_date_epoch_byteidentisch(
    tmp_path: Path,
    monkeypatch,
) -> None:
    corpus = _mini_korpus(tmp_path)
    database = corpus / "graph.sqlite"
    monkeypatch.setenv("SOURCE_DATE_EPOCH", "1700000000")

    assert main(["graph", "build", "--dir", str(corpus), "--db", str(database)]) == 0
    manifest = corpus / "norm_manifest.json"
    first = manifest.read_bytes()

    assert main(["graph", "build", "--dir", str(corpus), "--db", str(database)]) == 0
    second = manifest.read_bytes()

    assert second == first
    assert json.loads(second)["erstellt_am"] == "2023-11-14T22:13:20Z"


def test_graph_build_verwirft_ungueltiges_source_date_epoch(
    tmp_path: Path,
    monkeypatch,
) -> None:
    corpus = _mini_korpus(tmp_path)
    database = corpus / "graph.sqlite"
    monkeypatch.setenv("SOURCE_DATE_EPOCH", " 1700000000")

    assert main(["graph", "build", "--dir", str(corpus), "--db", str(database)]) == 2
    assert not database.exists()
