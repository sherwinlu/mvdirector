# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Mutable physical world-state ledger and aesthetic-clause conflict stripping.

Generalizes monotonic wardrobe tracking to arbitrary story props, lighting states, and irreversible
environmental changes (for example, a lantern switching from ``unlit`` to ``lit`` or a mirror
becoming ``shattered``). Provides ``filter_look_for_state`` to strip contradicting global lighting
or aesthetic clauses whenever a tracked object's state contradicts them.

Credits & Attribution:
    - Mutable vs. immutable physical state ledger and ``look_for_state`` conflict stripping
      inspired by ``afj176/open-cinema-studio``
      (https://github.com/afj176/open-cinema-studio, ``director.py``).
    - Event-sourced irreversible ``WorldState`` single-write-chokepoint validation inspired by
      ``JavaDerek/music-video-pipeline``
      (https://github.com/JavaDerek/music-video-pipeline — ``authoring/worldstate.py``).
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from mvdirector.errors import ConfigError

# Negative/inactive state descriptors and the positive aesthetic keywords they contradict.
_INACTIVE_STATE_CONTRADICTIONS: dict[frozenset[str], tuple[str, ...]] = {
    frozenset({"unlit", "off", "extinguished", "dark", "dead", "powered_off", "dim"}): (
        "glowing",
        "glow",
        "lit",
        "illuminated",
        "illuminating",
        "shining",
        "flaming",
        "burning",
        "radiant",
        "flickering flame",
        "warm flame",
    ),
    frozenset({"dry", "clear", "sunny", "cloudless"}): (
        "rain-soaked",
        "rain",
        "raining",
        "downpour",
        "drenched",
        "wet puddles",
        "stormy",
    ),
    frozenset({"intact", "unbroken", "whole", "pristine"}): (
        "shattered",
        "broken",
        "cracked",
        "smashed",
        "splintered",
    ),
    frozenset({"closed", "shut", "locked", "sealed"}): (
        "wide open",
        "open doorway",
        "flung open",
    ),
}


class IrreversibleFactViolation(ConfigError):
    """Raised when a shot attempts to reverse an irreversible world-state fact."""


@dataclass(frozen=True)
class StateInterval:
    """One interval in the append-only world-state ledger."""

    entity: str
    attribute: str
    value: str
    start_shot: int
    irreversible: bool = False
    event_id: str = ""


class WorldStateLedger:
    """Append-only interval-versioned world state with a single write choke point."""

    def __init__(self) -> None:
        self._history: dict[tuple[str, str], list[StateInterval]] = {}

    def set_fact(
        self,
        entity: str,
        attribute: str,
        value: str,
        *,
        shot_index: int,
        irreversible: bool = False,
        event_id: str = "",
    ) -> StateInterval:
        """Record a transition starting at ``shot_index``; rejects irreversible regressions."""
        norm_entity = entity.strip().lower()
        norm_attr = attribute.strip().lower()
        norm_val = value.strip()
        if not norm_entity or not norm_attr or not norm_val:
            raise ConfigError("World-state entity, attribute, and value must be non-empty.")
        key = (norm_entity, norm_attr)
        entries = self._history.setdefault(key, [])
        if entries:
            last = entries[-1]
            if shot_index < last.start_shot:
                raise ConfigError(
                    f"World-state update for {norm_entity}.{norm_attr} at shot {shot_index} "
                    f"precedes earlier transition at shot {last.start_shot}."
                )
            if last.irreversible and last.value.casefold() != norm_val.casefold():
                cause = f" (opened by {last.event_id})" if last.event_id else ""
                raise IrreversibleFactViolation(
                    f"Cannot change irreversible fact '{norm_entity}.{norm_attr}' from "
                    f"{last.value!r} to {norm_val!r} at shot {shot_index}{cause}."
                )
            if last.value.casefold() == norm_val.casefold():
                if irreversible and not last.irreversible:
                    promoted = StateInterval(
                        entity=last.entity,
                        attribute=last.attribute,
                        value=last.value,
                        start_shot=last.start_shot,
                        irreversible=True,
                        event_id=event_id or last.event_id,
                    )
                    entries[-1] = promoted
                    return promoted
                return last
        interval = StateInterval(
            entity=norm_entity,
            attribute=norm_attr,
            value=norm_val,
            start_shot=shot_index,
            irreversible=irreversible,
            event_id=event_id,
        )
        entries.append(interval)
        return interval

    def state_at(self, shot_index: int) -> dict[str, str]:
        """Return active ``{'entity.attribute': value}`` facts at ``shot_index``."""
        active: dict[str, str] = {}
        for (entity, attr), entries in self._history.items():
            current: StateInterval | None = None
            for item in entries:
                if item.start_shot <= shot_index:
                    current = item
                else:
                    break
            if current is not None:
                active[f"{entity}.{attr}"] = current.value
        return active


def _contradicted_terms(active_states: Mapping[str, str]) -> list[tuple[str, tuple[str, ...]]]:
    """Return ``(entity_token, forbidden_adjectives)`` pairs for inactive/contradicted states."""
    pairs: list[tuple[str, tuple[str, ...]]] = []
    for key, raw_val in active_states.items():
        val_tokens = set(re.findall(r"[a-z0-9_]+", raw_val.lower()))
        entity_part = key.rsplit(".", 1)[0] if "." in key else key
        entity_token = entity_part.rsplit(".", 1)[-1].strip().lower().replace("_", " ")
        for triggers, forbidden in _INACTIVE_STATE_CONTRADICTIONS.items():
            if val_tokens & triggers:
                pairs.append((entity_token, forbidden))
    return pairs


def filter_look_for_state(look_text: str, active_states: Mapping[str, str]) -> str:
    """Strip clauses from ``look_text`` that contradict any active world state.

    For example, if ``active_states`` contains ``{'lantern.illumination': 'unlit'}``, a comma-,
    semicolon-, or period-delimited clause like ``'warm glowing lantern light'`` is removed while
    unrelated global style/color clauses remain intact.
    """
    if not look_text.strip() or not active_states:
        return look_text
    contradictions = _contradicted_terms(active_states)
    if not contradictions:
        return look_text

    clauses = re.split(r"([,;]|(?<!\d)\.(?=\s|$))", look_text)
    kept: list[str] = []
    i = 0
    while i < len(clauses):
        part = clauses[i]
        sep = clauses[i + 1] if i + 1 < len(clauses) else ""
        i += 2
        low = part.lower()
        blocked = False
        for entity_token, forbidden in contradictions:
            entity_ok = (
                not entity_token
                or entity_token in ("weather", "sky")
                or bool(re.search(rf"\b{re.escape(entity_token)}\b", low))
            )
            if entity_ok and any(re.search(rf"\b{re.escape(word)}\b", low) for word in forbidden):
                blocked = True
                break
        if not blocked and part.strip():
            kept.append(part.strip() + (sep if sep else ""))
    cleaned = " ".join(kept).strip().rstrip(",;").strip()
    return cleaned


def build_ledger_from_facts(facts: Iterable[tuple[str, str, str, int, bool]]) -> WorldStateLedger:
    """Build a ledger from ``(entity, attribute, value, start_shot, irreversible)`` tuples."""
    ledger = WorldStateLedger()
    for entity, attr, val, shot_index, irreversible in sorted(facts, key=lambda row: row[3]):
        ledger.set_fact(
            entity,
            attr,
            val,
            shot_index=shot_index,
            irreversible=irreversible,
        )
    return ledger
