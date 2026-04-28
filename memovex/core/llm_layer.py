"""
memovex — Optional LLM Answer Generation Layer.

Thin wrapper that takes retrieved memories and generates a final answer
using any OpenAI-compatible API (GPT-4o, GPT-4o-mini, local models, etc.).

This layer is what puts MemoVex on equal footing with Mem0, Zep,
and Letta when comparing on generation benchmarks like LoCoMo and MuSiQue.

Usage:
    from memovex.core.llm_layer import LLMLayer

    llm = LLMLayer(model="gpt-4o-mini")   # or gpt-4o, claude-3-5-haiku, etc.

    # Multiple choice (LoCoMo)
    choice = llm.answer_mc(question, context, choices)

    # Open QA (MuSiQue)
    answer = llm.answer_open(question, context)

Environment variables:
    OPENAI_API_KEY   — required for OpenAI models
    OPENAI_BASE_URL  — override for local/compatible endpoints (optional)
"""

from __future__ import annotations

import logging
import os
import time
from typing import List, Optional

logger = logging.getLogger(__name__)


class LLMLayer:
    """
    Minimal LLM generation layer over retrieved context.

    Deliberately thin: no agent loop, no tool use, no chain-of-thought.
    Just: (question + retrieved_context) → answer.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        max_retries: int = 3,
        retry_delay: float = 2.0,
        temperature: float = 0.0,
    ):
        self.model = model
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.temperature = temperature
        self._total_tokens = 0
        self._call_count = 0

        key = api_key or os.getenv("OPENAI_API_KEY", "")
        url = base_url or os.getenv("OPENAI_BASE_URL")

        try:
            from openai import OpenAI
            kwargs = {"api_key": key}
            if url:
                kwargs["base_url"] = url
            self._client = OpenAI(**kwargs)
            self._available = True
        except ImportError:
            logger.warning("openai package not installed — LLMLayer disabled")
            self._client = None
            self._available = False
        except Exception as e:
            logger.warning("LLMLayer init failed: %s", e)
            self._client = None
            self._available = False

    @property
    def available(self) -> bool:
        return self._available and bool(os.getenv("OPENAI_API_KEY", ""))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def answer_mc(
        self,
        question: str,
        context: str,
        choices: List[str],
        system: str = "",
    ) -> Optional[int]:
        """
        Multiple-choice answering.

        Returns the 0-based index of the chosen answer, or None on failure.
        Matches the evaluation protocol of LoCoMo-mc10.
        """
        choices_text = "\n".join(f"  {i+1}. {c}" for i, c in enumerate(choices))
        prompt = (
            f"You are given excerpts from a long conversation. "
            f"Answer the question by selecting exactly ONE of the numbered options.\n\n"
            f"CONTEXT:\n{context}\n\n"
            f"QUESTION: {question}\n\n"
            f"OPTIONS:\n{choices_text}\n\n"
            f"Reply with ONLY the number of the correct option (1–{len(choices)}). "
            f"No explanation."
        )

        raw = self._call(prompt, system=system, max_tokens=8)
        if raw is None:
            return None

        # Parse "3" or "3." or "Option 3" etc.
        import re
        m = re.search(r"\d+", raw.strip())
        if m:
            idx = int(m.group()) - 1
            if 0 <= idx < len(choices):
                return idx
        return None

    def answer_open(
        self,
        question: str,
        context: str,
        system: str = "",
        max_tokens: int = 64,
    ) -> Optional[str]:
        """
        Open-ended QA.

        Returns a short answer string, or None on failure.
        Matches the evaluation protocol of MuSiQue.
        """
        prompt = (
            f"You are given passages from a knowledge base. "
            f"Answer the question using ONLY information from the passages. "
            f"Be concise — answer in as few words as possible.\n\n"
            f"PASSAGES:\n{context}\n\n"
            f"QUESTION: {question}\n\n"
            f"ANSWER:"
        )
        return self._call(prompt, system=system, max_tokens=max_tokens)

    def stats(self) -> dict:
        return {
            "model": self.model,
            "calls": self._call_count,
            "total_tokens": self._total_tokens,
            "estimated_cost_usd": self._estimate_cost(),
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _call(self, prompt: str, system: str = "",
              max_tokens: int = 128) -> Optional[str]:
        if not self.available:
            return None

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        for attempt in range(self.max_retries):
            try:
                resp = self._client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=max_tokens,
                )
                self._call_count += 1
                usage = resp.usage
                if usage:
                    self._total_tokens += (usage.prompt_tokens or 0) + (usage.completion_tokens or 0)
                return resp.choices[0].message.content.strip()

            except Exception as e:
                err = str(e)
                if "rate_limit" in err.lower() or "429" in err:
                    wait = self.retry_delay * (attempt + 1)
                    logger.debug("Rate limit — waiting %.1fs", wait)
                    time.sleep(wait)
                elif attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                else:
                    logger.warning("LLM call failed after %d attempts: %s", self.max_retries, e)
                    return None
        return None

    def _estimate_cost(self) -> float:
        # Approximate pricing per 1M tokens (input+output blended)
        pricing = {
            "gpt-4o":        5.00,
            "gpt-4o-mini":   0.30,
            "gpt-4-turbo":  10.00,
            "gpt-3.5-turbo": 0.50,
        }
        rate = pricing.get(self.model, 1.0)
        return round(self._total_tokens / 1_000_000 * rate, 4)
