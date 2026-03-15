from __future__ import annotations

import os

import numpy as np
import ollama

from evals.core import DatasetScorer, ScorerContext

_EMBED_MODEL = os.environ.get("EMBED_MODEL", "nomic-embed-text")


def _embed(client: ollama.Client, texts: list[str]) -> np.ndarray:
    response = client.embed(model=_EMBED_MODEL, input=texts)
    return np.array(response.embeddings)


class ContextSufficiencyScorer(DatasetScorer):
    """Embedding-based dataset-quality scorer.

    Measures semantic similarity between the expected answer and the most
    relevant context chunk via cosine similarity of Ollama embeddings
    (default model: nomic-embed-text, override with EMBED_MODEL env var).

    Returns the max cosine similarity across all chunks (0.0–1.0), or 0.0
    if context is absent or empty.

    Completion is structurally absent from the signature — this scorer measures
    whether the dataset's context is sufficient to support the expected answer,
    not whether the model produced a good answer.
    """

    def __init__(self) -> None:
        self._client = ollama.Client()

    def __call__(self, expected: str, ctx: ScorerContext) -> float:
        context = ctx.metadata.get("context")
        if not context:
            return 0.0

        chunks = [context] if isinstance(context, str) else list(context)
        if not chunks:
            return 0.0

        all_embs = _embed(self._client, [expected] + chunks)
        expected_emb = all_embs[0]
        chunk_embs = all_embs[1:]

        norms = np.linalg.norm(chunk_embs, axis=1) * np.linalg.norm(expected_emb)
        similarities = np.dot(chunk_embs, expected_emb) / norms

        return float(np.max(similarities))
