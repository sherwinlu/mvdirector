# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Retry policy: decide how the next attempt differs from the ones that failed.

Escalation ladder (configurable): a new seed, then a prompt rewritten from the QC
failure reasons, then adjusted workflow parameters. Reviewer feedback always
triggers a rewrite on the next attempt.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict

from mvdirector.config import RetryConfig
from mvdirector.errors import LLMError
from mvdirector.llm import LLMClient, text_block
from mvdirector.models import AssetKind, AssetState, RetryStrategy, Shot
from mvdirector.planning import prompts

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class AttemptPlan:
    number: int
    strategy: RetryStrategy
    seed: int


class _RewriteReply(BaseModel):
    model_config = ConfigDict(extra="ignore")
    prompt: str


_PROMPT_FAILURE_CHECKS = frozenset({"vision_spec", "no_people", "face_framing"})
_REPAIRABLE_STILL_CHECKS = frozenset({"vision_spec", "vision_artifacts", "no_people"})
# Knuth's 32-bit multiplicative hashing stride keeps sequential retry seeds well-dispersed inside
# 32-bit unsigned/signed bounds (video models such as MiniMax H3 truncate 64-bit seeds mod 2**32;
# inspired by JavaDerek/music-video-pipeline workflow_graph.py).
_KNUTH_SEED_STRIDE = 2654435761
_SEED_MODULUS = 2**31 - 1


def knuth_seed(base_seed: int, step: int) -> int:
    """Derive a well-dispersed positive 31-bit seed using Knuth's multiplicative stride."""
    return (int(base_seed) + int(step) * _KNUTH_SEED_STRIDE) % _SEED_MODULUS


class RetryPolicy:
    def __init__(self, config: RetryConfig, project_seed: int, llm: LLMClient | None) -> None:
        self._config = config
        self._project_seed = project_seed
        self._llm = llm

    def prepare(self, shot: Shot, kind: AssetKind, *, can_repair: bool = True) -> AttemptPlan:
        """Choose the next attempt's strategy and apply it to the asset's overrides."""
        state = shot.asset(kind)
        number = len(state.attempts) + 1
        after_fallback = kind is AssetKind.VIDEO and shot.fallback_from is not None
        if (
            kind is AssetKind.STILL
            and can_repair
            and not state.director_note
            and self.can_repair_still(state)
        ):
            return AttemptPlan(
                number=number,
                strategy=RetryStrategy.REPAIR_STILL,
                seed=self.seed_for(shot, kind, number),
            )
        strategy = self._strategy_for(state, after_fallback=after_fallback)
        if strategy is RetryStrategy.REWRITE_PROMPT:
            state.prompt_override = self._rewrite(shot, kind, state, after_fallback=after_fallback)
            state.director_note = None
        elif strategy is RetryStrategy.ADJUST_PARAMS:
            state.param_overrides.update(self._config.adjust_params.get(kind.value, {}))
        return AttemptPlan(number=number, strategy=strategy, seed=self.seed_for(shot, kind, number))

    def seed_for(self, shot: Shot, kind: AssetKind, number: int) -> int:
        """Deterministic 31-bit seed per attempt using Knuth stride dispersion."""
        key = f"{self._project_seed}:{shot.id}:{kind.value}"
        base = random.Random(key).randrange(_SEED_MODULUS)
        return knuth_seed(base, number)

    def can_repair_still(self, state: AssetState) -> bool:
        """True when ``in_place_repair`` is enabled and the latest failed still is repairable.

        Inspired by ``afj176/open-cinema-studio`` (``studio.py:_repair_frame``).
        """
        if not self._config.in_place_repair or not state.attempts:
            return False
        last = state.attempts[-1]
        if last.path is None or last.qc is None or last.qc.passed:
            return False
        failed_names = {c.name for c in last.qc.failed_checks}
        if not failed_names or not failed_names.issubset(_REPAIRABLE_STILL_CHECKS):
            return False
        # Do not repair twice in a row; fall back to full regeneration if repair failed.
        return last.strategy is not RetryStrategy.REPAIR_STILL

    def _strategy_for(self, state: AssetState, *, after_fallback: bool = False) -> RetryStrategy:
        if state.director_note:
            return RetryStrategy.REWRITE_PROMPT
        prior = state.pass_attempts(after_fallback=after_fallback)
        if not prior:
            return RetryStrategy.INITIAL
        ladder = self._config.strategies
        if all(a.error is not None for a in prior):
            # Every earlier attempt was an infrastructure error; nothing to learn from yet.
            return RetryStrategy.NEW_SEED
        used = {a.strategy for a in prior if a.error is None}
        last_qc = next((a.qc for a in reversed(prior) if a.qc is not None), None)
        if (
            last_qc is not None
            and RetryStrategy.REWRITE_PROMPT in ladder
            and RetryStrategy.REWRITE_PROMPT not in used
            and any(c.name in _PROMPT_FAILURE_CHECKS for c in last_qc.failed_checks)
        ):
            return RetryStrategy.REWRITE_PROMPT
        if RetryStrategy.REWRITE_PROMPT in used:
            for candidate in ladder:
                if candidate not in used:
                    return candidate
        return ladder[min(len(prior) - 1, len(ladder) - 1)]

    def _rewrite(
        self, shot: Shot, kind: AssetKind, state: AssetState, *, after_fallback: bool = False
    ) -> str:
        base = shot.still_prompt if kind is AssetKind.STILL else shot.motion_prompt
        current = state.prompt_override or base
        prior = state.pass_attempts(after_fallback=after_fallback)
        last_report = next((a.qc for a in reversed(prior) if a.qc is not None), None)
        issues = last_report.issues if last_report else []
        hints = last_report.fix_hints if last_report else []
        if state.director_note:
            hints = [state.director_note, *hints]

        if self._llm is not None:
            request = (
                f"ASSET: {kind.value} for a {shot.type.value} shot, framing {shot.framing.value}\n"
                f"CURRENT PROMPT:\n{current}\n\nPROBLEMS FOUND:\n"
                + "\n".join(f"- {issue}" for issue in issues or ["(none recorded)"])
                + "\n\nSUGGESTED FIXES:\n"
                + "\n".join(f"- {hint}" for hint in hints or ["(none)"])
            )
            try:
                reply = self._llm.complete_json(
                    system=prompts.REWRITE_SYSTEM,
                    content=[text_block(request)],
                    schema=_RewriteReply,
                )
                if reply.prompt.strip():
                    return reply.prompt.strip()
            except LLMError as exc:
                log.warning("Prompt rewrite failed for %s (%s); appending fix hints", shot.id, exc)

        additions = [hint for hint in dict.fromkeys(hints) if hint not in current]
        return " ".join([current, *additions]).strip()
