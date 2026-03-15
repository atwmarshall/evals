from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from evals.core import DatasetScorer, ScorerContext
from evals.scorers.context_sufficiency import ContextSufficiencyScorer
from evals.scorers.faithfulness import FaithfulnessScorer


def _ctx(context=None, input_text="What is the capital of Australia?", metadata=None):
    meta = dict(metadata or {})
    if context is not None:
        meta["context"] = context
    return ScorerContext(input=input_text, metadata=meta)


def _mock_response(content: str):
    msg = MagicMock()
    msg.content = content
    resp = MagicMock()
    resp.message = msg
    return resp


class TestFaithfulnessScorer:
    @pytest.fixture
    def scorer(self):
        return FaithfulnessScorer(scale=5)

    def test_returns_none_when_no_context_in_metadata(self, scorer):
        ctx = _ctx()  # no context key
        result = scorer("Sydney is the capital.", "Canberra", ctx)
        assert result is None

    def test_returns_none_when_context_is_empty_list(self, scorer):
        ctx = _ctx(context=[])
        result = scorer("Sydney is the capital.", "Canberra", ctx)
        assert result is None

    def test_api_error_returns_none(self, scorer):
        ctx = _ctx(context=["Australia has many cities."])
        scorer._client = MagicMock()
        scorer._client.chat.side_effect = RuntimeError("connection refused")
        result = scorer("Sydney.", "Canberra", ctx)
        assert result is None

    def test_score_normalised_correctly(self, scorer):
        ctx = _ctx(context=["The capital of Australia is Canberra."])
        scorer._client = MagicMock()

        scorer._client.chat.return_value = _mock_response('{"score": 5, "reasoning": "fully supported"}')
        assert scorer("Canberra.", "Canberra", ctx) == pytest.approx(1.0)

        scorer._client.chat.return_value = _mock_response('{"score": 1, "reasoning": "no support"}')
        assert scorer("Sydney.", "Canberra", ctx) == pytest.approx(0.0)

        scorer._client.chat.return_value = _mock_response('{"score": 3, "reasoning": "partial"}')
        assert scorer("Maybe Canberra.", "Canberra", ctx) == pytest.approx(0.5)

    def test_context_joined_as_newline_in_prompt(self, scorer):
        chunks = ["Chunk one.", "Chunk two."]
        ctx = _ctx(context=chunks)
        scorer._client = MagicMock()
        scorer._client.chat.return_value = _mock_response('{"score": 4, "reasoning": "ok"}')
        scorer("Some answer.", "expected", ctx)

        call_args = scorer._client.chat.call_args
        prompt = call_args[1]["messages"][0]["content"]
        assert "Chunk one.\nChunk two." in prompt

    def test_expected_is_ignored(self, scorer):
        ctx1 = _ctx(context=["The sky is blue."])
        ctx2 = _ctx(context=["The sky is blue."])
        scorer._client = MagicMock()
        scorer._client.chat.return_value = _mock_response('{"score": 5, "reasoning": "ok"}')

        scorer("The sky is blue.", "expected A", ctx1)
        call1 = scorer._client.chat.call_args[1]["messages"][0]["content"]

        scorer("The sky is blue.", "expected B", ctx2)
        call2 = scorer._client.chat.call_args[1]["messages"][0]["content"]

        assert call1 == call2

    def test_format_status_written_to_metadata_out(self, scorer):
        ctx = _ctx(context=["Some context."])
        scorer._client = MagicMock()
        scorer._client.chat.return_value = _mock_response('{"score": 4, "reasoning": "ok"}')
        scorer("answer", "expected", ctx)
        assert ctx.metadata_out.get("faithfulness_format_status") == "clean"

    def test_parse_failure_returns_none(self, scorer):
        ctx = _ctx(context=["Some context."])
        scorer._client = MagicMock()
        scorer._client.chat.return_value = _mock_response("I think it's a 4 out of 5.")
        result = scorer("answer", "expected", ctx)
        assert result is None


def _make_embed_response(embeddings: list[list[float]]):
    resp = MagicMock()
    resp.embeddings = embeddings
    return resp


class TestContextSufficiencyScorer:
    @pytest.fixture
    def scorer(self):
        return ContextSufficiencyScorer()

    def test_is_dataset_scorer(self, scorer):
        assert isinstance(scorer, DatasetScorer)

    def test_no_completion_arg(self, scorer):
        # The 2-arg signature is structural — completion must not exist
        import inspect
        sig = inspect.signature(scorer.__call__)
        assert list(sig.parameters) == ["expected", "ctx"]

    def test_returns_0_when_no_context(self, scorer):
        ctx = _ctx()  # no context key
        assert scorer("expected", ctx) == 0.0

    def test_returns_0_when_context_is_empty_list(self, scorer):
        ctx = _ctx(context=[])
        assert scorer("expected", ctx) == 0.0

    def test_high_similarity_when_answer_in_context(self, scorer):
        # expected and chunk embeddings point in same direction → cosine ~1.0
        scorer._client = MagicMock()
        scorer._client.embed.return_value = _make_embed_response([
            [1.0, 0.0],   # expected
            [0.9, 0.1],   # chunk — nearly parallel
        ])
        ctx = _ctx(context=["some relevant chunk"])
        score = scorer("Paris is France's capital", ctx)
        assert score > 0.8

    def test_low_similarity_when_answer_absent(self, scorer):
        # embeddings are orthogonal → cosine ≈ 0
        scorer._client = MagicMock()
        scorer._client.embed.return_value = _make_embed_response([
            [1.0, 0.0],   # expected
            [0.0, 1.0],   # chunk — orthogonal
        ])
        ctx = _ctx(context=["unrelated context"])
        score = scorer("Canberra", ctx)
        assert score < 0.1

    def test_returns_max_across_chunks(self, scorer):
        # Best chunk should win
        scorer._client = MagicMock()
        scorer._client.embed.return_value = _make_embed_response([
            [1.0, 0.0],   # expected
            [0.0, 1.0],   # chunk 1 — orthogonal (bad)
            [1.0, 0.0],   # chunk 2 — identical (perfect)
        ])
        ctx = _ctx(context=["bad chunk", "perfect chunk"])
        score = scorer("answer", ctx)
        assert score == pytest.approx(1.0)

    def test_context_as_string_not_list(self, scorer):
        scorer._client = MagicMock()
        scorer._client.embed.return_value = _make_embed_response([
            [1.0, 0.0],
            [1.0, 0.0],
        ])
        ctx = _ctx(context="The capital of France is Paris.")
        score = scorer("Paris", ctx)
        assert score == pytest.approx(1.0)

    def test_embed_called_with_expected_plus_chunks(self, scorer):
        scorer._client = MagicMock()
        scorer._client.embed.return_value = _make_embed_response([
            [1.0, 0.0],
            [0.5, 0.5],
        ])
        ctx = _ctx(context=["chunk one"])
        scorer("my expected", ctx)
        call_input = scorer._client.embed.call_args[1]["input"]
        assert call_input == ["my expected", "chunk one"]
