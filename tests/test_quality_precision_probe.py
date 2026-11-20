"""The experimental GPU cache must preserve task identity and output validity."""
import json
from types import SimpleNamespace

import numpy as np
import pytest

from tools.quality_precision_probe import CachedEncoder, CachedScores


def test_distinct_instructions_and_budgets_never_share_cached_answers(tmp_path):
    calls = []

    def score(pairs, instruction=None, max_length=None):
        calls.append((pairs, instruction, max_length))
        return [.6] * len(pairs)

    model = SimpleNamespace(name="qwen", revision="pinned", score=score,
                            score_with_instruction=score)
    cached = CachedScores(model, tmp_path / "scores.jsonl")
    pairs = [("fact", "source"), ("fact", "source")]
    assert cached.score(pairs) == [.6, .6]
    assert cached.score(pairs) == [.6, .6]
    cached.score_with_instruction(pairs, "support")
    cached.score_with_instruction(pairs, "contradiction")
    cached.score_with_instruction(pairs, "support", max_length=4096)
    assert len(calls) == 4 and all(len(call[0]) == 1 for call in calls)
    assert len((tmp_path / "scores.jsonl").read_text().splitlines()) == 4


@pytest.mark.parametrize("value", [True, "0.9", float("nan"), float("inf"), -1., 1.1])
def test_invalid_check_outputs_cannot_be_sanitized_by_cache(tmp_path, value):
    model = SimpleNamespace(name="qwen", revision="pinned",
                            score_with_instruction=lambda *a, **kw: [value])
    cached = CachedScores(model, tmp_path / "scores.jsonl")
    with pytest.raises(ValueError, match="output"):
        cached.score_with_instruction([("q", "d")], "support")
    assert cached.cache == {}
    assert (tmp_path / "scores.jsonl").read_text() == ""


def test_query_vectors_are_archived_in_exact_text_order(tmp_path):
    model = SimpleNamespace(name="encoder", revision="pinned", encode=lambda texts:
                            np.array([[len(t), 2.] for t in texts], dtype=np.float32))
    cached = CachedEncoder(model, tmp_path)
    cached.encode(["first", "two"])
    actual = cached.encode(["two", "third query", "first"])
    assert actual.tolist() == [[3., 2.], [11., 2.], [5., 2.]]
    assert json.loads((tmp_path / "queries.texts.json").read_text()) == ["first", "two", "third query"]
    assert np.load(tmp_path / "queries.npy").tolist() == [[5., 2.], [3., 2.], [11., 2.]]
