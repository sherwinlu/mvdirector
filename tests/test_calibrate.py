# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Threshold suggestions from labelled attempts (``mvd calibrate``)."""

from __future__ import annotations

import json

import pytest

from mvdirector.calibrate import (
    Label,
    collect_samples,
    labels_from_reviews,
    load_labels,
    render,
    suggest,
)
from mvdirector.errors import ConfigError
from mvdirector.models import (
    AssetKind,
    AssetStatus,
    Attempt,
    AudioSlice,
    CheckResult,
    CheckStatus,
    Manifest,
    QCReport,
    RetryStrategy,
    ReviewDecision,
    Shot,
    ShotType,
)


def _manifest(correlations: list[float]) -> Manifest:
    """One lip-sync shot per correlation value, each with one selected video attempt."""
    shots = []
    for index, value in enumerate(correlations, start=1):
        shot = Shot(
            id=f"s{index:03d}",
            index=index - 1,
            start=0.0,
            end=4.0,
            type=ShotType.LIPSYNC,
            audio_slice=AudioSlice(path="a.wav", start=0.0, end=4.0, head_handle=0.0),
            num_frames=97,
        )
        shot.video.attempts = [
            Attempt(
                number=1,
                strategy=RetryStrategy.INITIAL,
                seed=1,
                prompt="p",
                path=f"assets/videos/{shot.id}_a01.mp4",
                qc=QCReport(
                    passed=True,
                    score=1.0,
                    checks=[
                        CheckResult(
                            name="lipsync",
                            status=CheckStatus.PASSED,
                            metrics={"correlation": value, "lag_seconds": 0.0},
                        ),
                        CheckResult(
                            name="dimensions",
                            status=CheckStatus.PASSED,
                            metrics={"width": 1280, "height": 720},
                        ),
                    ],
                ),
            )
        ]
        shot.video.status, shot.video.selected_attempt = AssetStatus.PASSED, 1
        shots.append(shot)
    return Manifest(title="t", duration=4.0 * len(shots), fps=24, shots=shots)


def test_labels_load_from_csv_and_json(tmp_path):
    csv_path = tmp_path / "labels.csv"
    csv_path.write_text(
        "shot,kind,attempt,label\ns001,video,1,good\ns002,video,,BAD\ns003,still,2,1\n"
    )
    labels = load_labels(csv_path)
    assert labels == [
        Label("s001", AssetKind.VIDEO, 1, True),
        Label("s002", AssetKind.VIDEO, None, False),
        Label("s003", AssetKind.STILL, 2, True),
    ]

    json_path = tmp_path / "labels.json"
    json_path.write_text(json.dumps([{"shot": "s001", "label": "fail"}]))
    assert load_labels(json_path) == [Label("s001", AssetKind.VIDEO, None, False)]

    bad = tmp_path / "bad.csv"
    bad.write_text("shot,kind,attempt,label\ns001,video,1,maybe\n")
    with pytest.raises(ConfigError, match="row 1"):
        load_labels(bad)
    with pytest.raises(ConfigError, match="not found"):
        load_labels(tmp_path / "missing.csv")


def test_review_decisions_become_video_labels():
    manifest = _manifest([0.7, 0.2, 0.5])
    manifest.shots[0].review.decision = ReviewDecision.APPROVED
    manifest.shots[1].review.decision = ReviewDecision.REJECTED
    assert labels_from_reviews(manifest) == [
        Label("s001", AssetKind.VIDEO, 1, True),
        Label("s002", AssetKind.VIDEO, 1, False),
    ]


def test_suggestion_separates_good_from_bad_and_names_the_config_key():
    manifest = _manifest([0.72, 0.65, 0.58, 0.31, 0.22, 0.15])
    labels = [
        Label(f"s{i:03d}", AssetKind.VIDEO, None, good)
        for i, good in enumerate([True, True, True, False, False, False], start=1)
    ]
    samples = collect_samples(manifest, labels)
    assert set(samples) == {"correlation"}  # width, height and lag are not thresholds

    (suggestion,) = suggest(samples)
    assert suggestion.config_key == "qc.video.lipsync_min"
    assert suggestion.higher_is_better
    assert (suggestion.good_count, suggestion.bad_count) == (3, 3)
    assert 0.31 < suggestion.threshold < 0.58
    assert suggestion.balanced_accuracy == 1.0

    text = render([suggestion], len(labels))
    assert "qc.video.lipsync_min" in text and ">=" in text and "6 labelled" in text


def test_overlapping_labels_report_low_balanced_accuracy():
    manifest = _manifest([0.5, 0.5, 0.5, 0.5])
    labels = [
        Label("s001", AssetKind.VIDEO, None, True),
        Label("s002", AssetKind.VIDEO, None, False),
        Label("s003", AssetKind.VIDEO, None, True),
        Label("s004", AssetKind.VIDEO, None, False),
    ]
    (suggestion,) = suggest(collect_samples(manifest, labels))
    assert suggestion.balanced_accuracy == 0.5
    assert "does not separate" in render([suggestion], 4)


def test_labels_without_qc_reports_or_shots_are_rejected():
    manifest = _manifest([0.5])
    with pytest.raises(ConfigError, match="unknown shot"):
        collect_samples(manifest, [Label("s999", AssetKind.VIDEO, None, True)])
    with pytest.raises(ConfigError, match="no QC report"):
        collect_samples(manifest, [Label("s001", AssetKind.STILL, None, True)])
    assert "no metric had both" in render([], 1)
