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


class RetryPolicy:
    def __init__(self, config: RetryConfig, project_seed: int, llm: LLMClient | None) -> None:
        self._config = config
        self._project_seed = project_seed
        self._llm = llm

    def prepare(self, shot: Shot, kind: AssetKind) -> AttemptPlan:
        """Choose the next attempt's strategy and apply it to the asset's overrides."""
        state = shot.asset(kind)
        number = len(state.attempts) + 1
        after_fallback = kind is AssetKind.VIDEO and shot.fallback_from is not None
        strategy = self._strategy_for(state, after_fallback=after_fallback)
        if strategy is RetryStrategy.REWRITE_PROMPT:
            state.prompt_override = self._rewrite(shot, kind, state, after_fallback=after_fallback)
            state.director_note = None
        elif strategy is RetryStrategy.ADJUST_PARAMS:
            state.param_overrides.update(self._config.adjust_params.get(kind.value, {}))
        return AttemptPlan(number=number, strategy=strategy, seed=self.seed_for(shot, kind, number))

    def seed_for(self, shot: Shot, kind: AssetKind, number: int) -> int:
        """Deterministic per attempt, so any result can be reproduced from the manifest."""
        key = f"{self._project_seed}:{shot.id}:{kind.value}:{number}"
        return random.Random(key).randrange(2**31)

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
