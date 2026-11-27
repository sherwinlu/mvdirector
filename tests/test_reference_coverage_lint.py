# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Tests for pre-spend reference coverage linting and ``mvd assets --lint``."""

from __future__ import annotations

import shlex

import pytest
from typer.testing import CliRunner

from mvdirector import api
from mvdirector.cli import app
from mvdirector.errors import ConfigError
from mvdirector.generation.preflight import CoverageFinding, lint_reference_coverage
from mvdirector.models import (
    AudioSlice,
    Character,
    Manifest,
    PropDesign,
    ReferenceSet,
    SetDesign,
    Shot,
    ShotType,
    VisualBible,
)


def _shot(
    index: int,
    *,
    section: str = "Verse 1",
    still_prompt: str = "",
    motion_prompt: str = "",
    setting: str = "",
    props: list[str] | None = None,
) -> Shot:
    return Shot(
        id=f"s{index:03d}",
        index=index - 1,
        start=float((index - 1) * 4),
        end=float(index * 4),
        type=ShotType.BROLL,
        section=section,
        setting=setting,
        props=props or [],
        still_prompt=still_prompt,
        motion_prompt=motion_prompt,
        audio_slice=AudioSlice(
            path="a.wav",
            start=float((index - 1) * 4),
            end=float(index * 4),
            head_handle=0.0,
        ),
        num_frames=97,
    )


def test_lint_flags_recurring_undeclared_prop_across_three_shots_and_singularizes() -> None:
    manifest = Manifest(
        title="t",
        duration=16.0,
        fps=24,
        shots=[
            _shot(1, section="Verse 1", still_prompt="Close-up of a brass lantern on wet stone."),
            _shot(
                2, section="Verse 1", still_prompt="The brass lantern swings gently in the wind."
            ),
            _shot(
                3,
                section="Verse 1",
                still_prompt="Several brass lanterns glow beside the iron gate.",
            ),
            _shot(4, section="Verse 1", still_prompt="A single violin rests in shadow."),
        ],
    )
    findings = lint_reference_coverage(manifest)
    by_noun = {f.noun: f for f in findings}
    assert "brass lantern" in by_noun
    # Suppresses redundant single-word "lantern" when "brass lantern" covers the same shots.
    assert "lantern" not in by_noun
    finding = by_noun["brass lantern"]
    assert isinstance(finding, CoverageFinding)
    assert finding.count == 3
    assert finding.shot_ids == ["s001", "s002", "s003"]
    assert finding.sections == ["Verse 1"]
    assert finding.entity_kind == "prop"
    assert finding.declared is False
    assert finding.locked is False
    assert finding.suggested_command == "mvd props generate --prop brass_lantern"
    assert "violin" not in by_noun
    assert finding.to_dict()["noun"] == "brass lantern"


def test_lint_flags_recurring_noun_across_two_sections_and_classifies_sets() -> None:
    manifest = Manifest(
        title="t",
        duration=8.0,
        fps=24,
        shots=[
            _shot(1, section="Intro", still_prompt="Wide view inside the gothic cathedral."),
            _shot(2, section="Chorus", still_prompt="Light pours across the cathedral arches."),
        ],
    )
    findings = lint_reference_coverage(manifest)
    by_noun = {f.noun: f for f in findings}
    assert "cathedral" in by_noun
    cathedral = by_noun["cathedral"]
    assert cathedral.count == 2
    assert cathedral.sections == ["Intro", "Chorus"]
    assert cathedral.entity_kind == "set"
    assert cathedral.suggested_command == "mvd plates generate --set cathedral"


def test_lint_ignores_locked_entities_and_cinematography_stopwords() -> None:
    manifest = Manifest(
        title="t",
        duration=12.0,
        fps=24,
        character=Character(
            name="Aria",
            appearance="Tall woman in a velvet coat",
            references={"base": ReferenceSet(sheet="assets/reference/aria.png")},
        ),
        visual_bible=VisualBible(
            settings=[
                SetDesign(
                    name="Neon Alley",
                    description="Wet asphalt alley with iron fire escapes",
                    plate="assets/plates/neon_alley.png",
                )
            ],
            props=[
                PropDesign(
                    id="silver_locket",
                    name="Silver Locket",
                    description="Engraved oval locket on a thin chain",
                    plate="assets/props/locket.png",
                )
            ],
        ),
        shots=[
            _shot(
                1,
                section="Verse 1",
                setting="Neon Alley",
                still_prompt=(
                    "Close-up camera shot of Aria holding the silver locket in neon shadow."
                ),
                motion_prompt="Slow push-in as rain falls across the alley.",
            ),
            _shot(
                2,
                section="Chorus",
                setting="Neon Alley",
                still_prompt="Medium camera shot of her velvet coat and the engraved locket.",
                motion_prompt="Gentle handheld drift with soft glowing lights.",
            ),
            _shot(
                3,
                section="Bridge",
                setting="Neon Alley",
                still_prompt="Wide camera shot down the empty asphalt alley in mist.",
                motion_prompt="Slow dolly forward in silhouette.",
            ),
        ],
    )
    assert lint_reference_coverage(manifest) == []


def test_lint_flags_declared_but_unlocked_props_and_sets() -> None:
    manifest = Manifest(
        title="t",
        duration=12.0,
        fps=24,
        visual_bible=VisualBible(
            settings=[SetDesign(name="Harbor Pier", description="Weathered wooden dock")],
            props=[
                PropDesign(id="pocket_watch", name="Pocket Watch", description="Tarnished brass")
            ],
        ),
        shots=[
            _shot(
                1,
                section="Verse 1",
                setting="Harbor Pier",
                props=["pocket_watch"],
                still_prompt="The pocket watch ticks on a crate.",
            ),
            _shot(
                2,
                section="Chorus",
                setting="Harbor Pier",
                still_prompt="Waves slap the pier beneath the pocket watch.",
            ),
        ],
    )
    findings = lint_reference_coverage(manifest)
    by_noun = {f.noun: f for f in findings}
    assert by_noun["Harbor Pier"].declared is True
    assert by_noun["Harbor Pier"].entity_kind == "set"
    assert by_noun["Harbor Pier"].suggested_command == "mvd plates generate --set 'Harbor Pier'"
    assert by_noun["Pocket Watch"].declared is True
    assert by_noun["Pocket Watch"].entity_kind == "prop"
    assert by_noun["Pocket Watch"].suggested_command == "mvd props generate --prop pocket_watch"


def test_lint_accepts_raw_mapping_inputs_and_validates_thresholds() -> None:
    raw_manifest = {
        "visual_bible": {"settings": [], "props": []},
        "shots": [
            {"id": "s1", "section": "Verse", "description": "A cracked hourglass sits on oak."},
            {
                "id": "s2",
                "section": "Chorus",
                "continuity_note": "Sand falls inside the hourglass.",
            },
        ],
    }
    findings = lint_reference_coverage(raw_manifest)
    assert [f.noun for f in findings] == ["hourglass"]

    with pytest.raises(ConfigError):
        lint_reference_coverage(raw_manifest, min_shots=0)
    with pytest.raises(ConfigError):
        lint_reference_coverage(raw_manifest, min_sections=True)


def test_api_lint_assets_and_cli_assets_lint(project) -> None:
    manifest = Manifest(
        title="t",
        duration=12.0,
        fps=24,
        shots=[
            _shot(1, section="Verse 1", still_prompt="A glowing compass spins on the desk."),
            _shot(2, section="Chorus", still_prompt="Close-up of the compass needle."),
            _shot(3, section="Outro", still_prompt="The compass rests in snow."),
        ],
    )
    project.save_manifest(manifest)

    findings = api.lint_assets(project.root)
    assert [f.noun for f in findings] == ["compass"]

    runner = CliRunner()
    res = runner.invoke(app, ["assets", "--lint", "-p", str(project.root)])
    assert res.exit_code == 0
    assert "Reference coverage lint" in res.output
    assert "compass" in res.output
    assert "mvd props generate --prop compass" in res.output

    # Clean project reports zero findings cleanly
    manifest.shots = [_shot(1, section="Verse 1", still_prompt="A single lantern glows.")]
    project.save_manifest(manifest)
    clean_res = runner.invoke(app, ["assets", "--lint", "-p", str(project.root)])
    assert clean_res.exit_code == 0
    assert "No recurring undeclared or unlocked references found." in clean_res.output


def test_lint_shell_quotes_suggested_commands() -> None:
    malicious_set = 'Alley"; rm -rf / #'
    malicious_prop = 'Bad"$(id)'
    manifest = Manifest(
        title="t",
        duration=12.0,
        fps=24,
        visual_bible=VisualBible(settings=[SetDesign(name=malicious_set, description="Dark")]),
        shots=[
            _shot(1, section="Intro", setting=malicious_set, still_prompt="Rain falls."),
            _shot(2, section="Chorus", setting=malicious_set, still_prompt="Thunder rolls."),
        ],
    )
    findings = lint_reference_coverage(manifest)
    assert len(findings) == 1
    assert findings[0].suggested_command == (
        f"mvd plates generate --set {shlex.quote(malicious_set)}"
    )
    assert shlex.split(findings[0].suggested_command) == [
        "mvd",
        "plates",
        "generate",
        "--set",
        malicious_set,
    ]

    raw_manifest = {
        "visual_bible": {
            "settings": [{"name": malicious_set, "description": ""}],
            "props": [{"id": malicious_prop, "name": "Bad Prop", "description": ""}],
        },
        "shots": [
            {
                "id": "s1",
                "section": "Intro",
                "setting": malicious_set,
                "props": [malicious_prop],
            },
            {
                "id": "s2",
                "section": "Chorus",
                "setting": malicious_set,
                "props": [malicious_prop],
            },
        ],
    }
    raw_findings = {f.entity_kind: f for f in lint_reference_coverage(raw_manifest)}
    assert shlex.split(raw_findings["set"].suggested_command)[-1] == malicious_set
    assert shlex.split(raw_findings["prop"].suggested_command)[-1] == malicious_prop


def test_lint_alias_matching_uses_whole_tokens_not_substrings() -> None:
    manifest = Manifest(
        title="t",
        duration=16.0,
        fps=24,
        visual_bible=VisualBible(
            settings=[SetDesign(name="Bar", description="Dim dive bar")],
            props=[
                PropDesign(id="ring", name="Ring", description="Gold signet ring"),
                PropDesign(id="star", name="Star", description="Silver star pendant"),
                PropDesign(id="pocket_watch", name="Pocket Watch", description="Brass watch"),
            ],
        ),
        shots=[
            _shot(1, section="Verse 1", still_prompt="Aria is wearing a dark coat, staring out."),
            _shot(2, section="Chorus", still_prompt="Flickering neon, bare floor, starting now."),
            _shot(3, section="Bridge", still_prompt="A heavy barrel rolls past the doorway."),
        ],
    )
    assert lint_reference_coverage(manifest) == []

    # Whole-word & multi-word mentions still match unlocked entities.
    matched_manifest = Manifest(
        title="t",
        duration=8.0,
        fps=24,
        visual_bible=manifest.visual_bible,
        shots=[
            _shot(
                1,
                section="Verse 1",
                still_prompt="She holds a gold ring as the pocket watch ticks.",
            ),
            _shot(
                2,
                section="Chorus",
                still_prompt="The ring glints beside the pocket watch.",
            ),
        ],
    )
    by_noun = {f.noun: f for f in lint_reference_coverage(matched_manifest)}
    assert set(by_noun) == {"Ring", "Pocket Watch"}
    assert by_noun["Ring"].shot_ids == ["s001", "s002"]
    assert by_noun["Pocket Watch"].shot_ids == ["s001", "s002"]


def test_lint_chunk_re_skips_articles_after_prepositions_and_detects_three_letter_props() -> None:
    manifest = Manifest(
        title="t",
        duration=24.0,
        fps=24,
        shots=[
            _shot(
                1,
                section="Verse 1",
                still_prompt="Aria holds a brass compass in dim red light.",
            ),
            _shot(
                2,
                section="Verse 1",
                still_prompt="Close-up of the brass compass in low fog.",
            ),
            _shot(
                3,
                section="Verse 1",
                still_prompt="The brass compass spins under open sky.",
            ),
            _shot(
                4,
                section="Verse 1",
                still_prompt="Slow pan to a tarnished key beside a hat and map.",
            ),
            _shot(
                5,
                section="Verse 1",
                still_prompt="Her eye tracks the key near the folded map and hat.",
            ),
            _shot(
                6,
                section="Verse 1",
                still_prompt="Cut to the brass key resting on a crumpled map and hat.",
            ),
        ],
    )
    by_noun = {f.noun: f for f in lint_reference_coverage(manifest)}
    assert "brass compass" in by_noun
    assert by_noun["brass compass"].count == 3
    assert "compass" not in by_noun
    assert {"key", "hat", "map"}.issubset(by_noun)
    for noise in ("pan", "eye", "sky", "red", "dim", "low", "cut"):
        assert noise not in by_noun


def test_lint_does_not_flag_verbs_and_camera_directions_as_props() -> None:
    """Regression: every recurring content word used to become a prop finding.

    On real 50-150 shot manifests the bare-token pass reported ~140 findings per project,
    led by words like "forward", "remain" and "maintain".
    """
    prompts = [
        "The camera remains fixed as she leans forward into the light.",
        "The camera maintains a steady push forward while the lantern swings gently.",
        "The camera remains still; she walks forward beside a brass lantern.",
        "Slow push forward at eye-level, the camera holds as the lantern sways.",
    ]
    sections = ["Verse 1", "Chorus", "Verse 2", "Bridge"]
    manifest = Manifest(
        title="t",
        duration=16.0,
        fps=24,
        shots=[
            _shot(i + 1, section=sections[i], still_prompt=text) for i, text in enumerate(prompts)
        ],
    )
    nouns = {f.noun for f in lint_reference_coverage(manifest)}
    for noise in ("forward", "remain", "maintain", "swing", "walk", "push", "eye-level"):
        assert noise not in nouns
    # The lantern is established as a noun once ("beside a brass lantern."), so its other,
    # verb-adjacent mentions still count toward the recurrence threshold.
    assert "lantern" in nouns


def test_cli_lint_tells_you_to_declare_undeclared_entities_before_generating(project) -> None:
    """Regression: the suggested generate command fails for a prop the bible does not declare."""
    manifest = Manifest(
        title="t",
        duration=12.0,
        fps=24,
        visual_bible=VisualBible(settings=[SetDesign(name="Neon Alley", description="wet alley")]),
        shots=[
            _shot(
                1, section="Verse 1", still_prompt="A compass on the desk.", setting="Neon Alley"
            ),
            _shot(
                2, section="Chorus", still_prompt="Close-up of the compass.", setting="Neon Alley"
            ),
            _shot(3, section="Outro", still_prompt="She holds the compass.", setting="Neon Alley"),
        ],
    )
    project.save_manifest(manifest)
    res = CliRunner().invoke(app, ["assets", "--lint", "-p", str(project.root)])
    assert res.exit_code == 0, res.output
    assert (
        "suggest: declare it under visual_bible.props in manifest.json, then: "
        "mvd props generate --prop compass"
    ) in res.output
    # A declared-but-unlocked set is directly actionable, so no declare step is suggested.
    assert "suggest: mvd plates generate --set 'Neon Alley'" in res.output
