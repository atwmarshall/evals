from __future__ import annotations

import logging
import os

import ollama

from evals.core import Dataset, Sample, ScorerCallable, ScorerContext

logger = logging.getLogger(__name__)

VARIATION_PROMPTS: dict[str, str] = {
    "synonym_swap": "Rewrite this question replacing key verbs and nouns with synonyms. Keep the exact meaning.",
    "rephrase":     "Rephrase this question completely differently. Same meaning, different sentence structure.",
    "add_noise":    "Add one irrelevant sentence to this question. The added sentence should be unrelated to the task.",
    "formal":       "Rewrite this question in a more formal register.",
    "concise":      "Rewrite this question more concisely. Remove all unnecessary words.",
}

_VARIATION_PROMPT = """\
{instruction}

Text:
{input}

Return only the rewritten text. Do not add any explanation or preamble.\
"""


class VariationGenerator:
    """Generates semantically equivalent dataset variations by rewriting inputs via LLM.

    Each variation type applies a different rewriting instruction (synonym swap,
    rephrase, noise injection, etc.) while preserving sample.id, sample.expected,
    and sample.metadata unchanged.

    A dedicated VARIATION_MODEL env var is provided so the variation model can
    differ from both the evaluated model (DEFAULT_MODEL) and the judge (JUDGE_MODEL).
    Using the same model to generate and score variations would measure
    self-consistency rather than scorer reliability.
    """

    def __init__(self, model: str | None = None) -> None:
        self.model = model or os.environ.get("VARIATION_MODEL") or os.environ.get("DEFAULT_MODEL", "llama3.2:3b")
        host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        self._client = ollama.Client(host=host)

    def generate(
        self,
        dataset: Dataset,
        variations: list[str] | None = None,
    ) -> dict[str, Dataset]:
        """Generate varied datasets for each variation type.

        Args:
            dataset: Original dataset to vary.
            variations: Subset of VARIATION_PROMPTS keys to apply. Defaults to all.

        Returns:
            Dict mapping variation name to Dataset. Always includes "baseline"
            with the original dataset unchanged.
        """
        if variations is None:
            variations = list(VARIATION_PROMPTS.keys())

        unknown = [v for v in variations if v not in VARIATION_PROMPTS]
        if unknown:
            raise ValueError(f"unknown variation types: {unknown!r}. Valid: {list(VARIATION_PROMPTS)}")

        result: dict[str, Dataset] = {"baseline": dataset}

        for variation_name in variations:
            instruction = VARIATION_PROMPTS[variation_name]
            varied_samples: list[Sample] = []
            for sample in dataset:
                try:
                    varied_input = self._vary_sample(sample.input, instruction)
                except Exception as e:
                    logger.warning(
                        "variation failed for sample %s (%s): %s",
                        sample.id, variation_name, e,
                    )
                    varied_input = sample.input
                varied_samples.append(Sample(
                    id=sample.id,
                    input=varied_input,
                    expected=sample.expected,
                    metadata=sample.metadata,
                ))
            result[variation_name] = Dataset(samples=varied_samples)

        return result

    def validate_variations(
        self,
        variations: dict[str, Dataset],
        scorer: ScorerCallable,
        threshold: float = 0.8,
    ) -> dict[str, Dataset]:
        """Return a filtered copy of `variations` where per-sample validity is confirmed.

        For each variation type (excluding "baseline"), scores the original expected
        answer against the varied input using the provided scorer. Any (sample, variation)
        pair where the scorer returns None or a score below `threshold` is discarded.

        Note: with pure scorers (exact_match, normalised_match, regex) this is a no-op —
        they do not read ctx.input, so the expected answer always scores 1.0 against itself.
        This step is only meaningfully discriminating with context-aware scorers
        (LLMJudgeScorer, CascadeScorer).

        The "baseline" key is passed through unchanged without scoring.
        Empty variation Datasets (all samples filtered) are kept as keys with samples=[]
        so downstream reporters see them rather than silently missing them.
        """
        validated: dict[str, Dataset] = {}

        for variation_name, dataset in variations.items():
            if variation_name == "baseline":
                validated["baseline"] = dataset
                continue

            valid_samples: list[Sample] = []

            for varied_sample in dataset:
                ctx = ScorerContext(
                    input=varied_sample.input,
                    metadata={**varied_sample.metadata, "id": varied_sample.id},
                    metadata_out={},
                )
                try:
                    score = scorer(varied_sample.expected, varied_sample.expected, ctx)
                except Exception as exc:
                    logger.warning(
                        "validate_variations: scorer raised for sample %s variation %r: %s — discarding",
                        varied_sample.id, variation_name, exc,
                    )
                    continue

                if score is None:
                    logger.warning(
                        "validate_variations: discarded sample %s from variation %r — scorer_returned_none (threshold=%.3f)",
                        varied_sample.id, variation_name, threshold,
                    )
                    continue

                if score < threshold:
                    logger.warning(
                        "validate_variations: discarded sample %s from variation %r — score_below_threshold (score=%.3f, threshold=%.3f)",
                        varied_sample.id, variation_name, score, threshold,
                    )
                    continue

                valid_samples.append(varied_sample)

            if not valid_samples:
                logger.warning(
                    "validate_variations: variation %r has zero valid samples after filtering",
                    variation_name,
                )

            validated[variation_name] = Dataset(samples=valid_samples)

        return validated

    def _vary_sample(self, input_text: str, instruction: str) -> str:
        """Call the LLM to rewrite input_text according to instruction.

        Raises on Ollama error — caller catches and falls back to original input.
        """
        prompt = _VARIATION_PROMPT.format(instruction=instruction, input=input_text)
        response = self._client.chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.0},
        )
        return response.message.content.strip()
