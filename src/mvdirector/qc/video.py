# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Quality gate for video clips: technical, temporal, lip-sync, identity and vision checks."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import soundfile as sf
from PIL import Image

from mvdirector import media
from mvdirector.config import QCConfig, VideoConfig
from mvdirector.models import (
    AssetKind,
    CheckResult,
    CheckStatus,
    Manifest,
    QCReport,
    Shot,
    ShotType,
)
from mvdirector.qc import lipsync
from mvdirector.qc.common import dimensions_check, identity_check, vision_checks
from mvdirector.qc.faces import FaceAnalyzer, IdentityReference, MouthTracker
from mvdirector.qc.scoring import aggregate, linear_score, skipped
from mvdirector.qc.vision import VisionJudge

log = logging.getLogger(__name__)

_MOTION_SIDE = 64


def frame_differences(frames: list[np.ndarray]) -> np.ndarray:
    """Mean absolute difference (0-1) between consecutive frames, on tiny greyscale copies."""
    small = [
        np.asarray(
            Image.fromarray(f).convert("L").resize((_MOTION_SIDE, _MOTION_SIDE)), dtype=np.float64
        )
        for f in frames
    ]
    if len(small) < 2:
        return np.zeros(0)
    return np.array([np.abs(b - a).mean() / 255.0 for a, b in zip(small[:-1], small[1:])])


class VideoQC:
    def __init__(
        self,
        qc: QCConfig,
        video: VideoConfig,
        face_analyzer: FaceAnalyzer | None,
        mouth_tracker: MouthTracker | None,
        vision: VisionJudge | None,
    ) -> None:
        self._qc = qc
        self._cfg = qc.video
        self._video = video
        self._faces = face_analyzer
        self._mouth = mouth_tracker
        self._vision = vision

    def evaluate(
        self,
        path: Path,
        shot: Shot,
        manifest: Manifest,
        audio_slice: Path,
        reference: IdentityReference | None,
        *,
        reference_image: Path | None = None,
        extra_references: list[tuple[str, Path]] | None = None,
        expected_size: tuple[int, int] | None = None,
    ) -> QCReport:
        info = media.probe(path)
        technical = self._technical_checks(info, shot, expected_size=expected_size)
        if any(c.status is CheckStatus.FAILED for c in technical):
            # A broken file makes every later measurement meaningless.
            return aggregate(technical, self._qc.weights, self._qc.pass_score)

        frames = media.extract_frames(path, self._cfg.analysis_width)
        fps = info.fps or float(self._video.fps)
        checks = [*technical, *self._temporal_checks(frames, allow_cuts=shot.allow_cuts)]
        if shot.type is ShotType.LIPSYNC:
            checks += self._lipsync_checks(frames, fps, shot, manifest, audio_slice)
        if shot.type.shows_singer:
            checks.append(self._identity_check(frames, fps, reference))
        checks += vision_checks(
            self._vision,
            self._sample(frames, self._cfg.vision_frames),
            shot,
            manifest,
            AssetKind.VIDEO,
            self._cfg.vision_hard_min,
            self._cfg.vision_character_hard_min,
            reference_image,
            extra_references=extra_references,
        )
        return aggregate(checks, self._qc.weights, self._qc.pass_score)

    # ---------------------------------------------------------------- checks

    def _technical_checks(
        self,
        info: media.MediaInfo,
        shot: Shot,
        expected_size: tuple[int, int] | None = None,
    ) -> list[CheckResult]:
        if info.width is None or info.height is None:
            return [
                CheckResult(
                    name="dimensions",
                    status=CheckStatus.FAILED,
                    hard=True,
                    detail="file has no video stream",
                )
            ]
        needed = shot.audio_slice.head_handle + shot.duration
        long_enough = info.duration >= needed - self._cfg.duration_tolerance
        target_w, target_h = expected_size or (self._video.width, self._video.height)
        return [
            dimensions_check(
                info.width,
                info.height,
                target_w,
                target_h,
                self._cfg.aspect_tolerance,
                self._cfg.min_resolution_scale,
            ),
            CheckResult(
                name="duration",
                status=CheckStatus.PASSED if long_enough else CheckStatus.FAILED,
                hard=True,
                detail=""
                if long_enough
                else f"clip is {info.duration:.2f}s, shot needs {needed:.2f}s",
                metrics={"duration": round(info.duration, 3)},
            ),
        ]

    def _temporal_checks(
        self, frames: list[np.ndarray], *, allow_cuts: bool = False
    ) -> list[CheckResult]:
        differences = frame_differences(frames)
        if differences.size == 0:
            return [skipped("temporal", "clip has a single frame")]
        median, peak = float(np.median(differences)), float(differences.max())
        cut_mask = differences > self._cfg.jump_cut_threshold
        cut_count = int(np.sum(cut_mask))
        non_cut = differences[~cut_mask]
        flicker = float(np.percentile(non_cut if non_cut.size else differences, 95))
        metrics: dict[str, float] = {
            "median_motion": round(median, 5),
            "peak_motion": round(peak, 4),
            "p95_motion": round(flicker, 4),
        }
        if allow_cuts:
            metrics["cut_count"] = float(cut_count)
        if median < self._cfg.static_max_motion:
            return [
                CheckResult(
                    name="temporal",
                    status=CheckStatus.FAILED,
                    hard=True,
                    score=0.0,
                    detail="clip is almost static",
                    metrics=metrics,
                    fix_hint="Describe clear, continuous motion of the singer and the camera.",
                )
            ]
        max_montage_cuts = max(3, differences.size // 4)
        if cut_count > 0 and (not allow_cuts or cut_count > max_montage_cuts or not non_cut.size):
            detail = (
                f"too many abrupt scene changes ({cut_count}); looks like flicker or strobe"
                if allow_cuts
                else "clip contains an abrupt scene change"
            )
            hint = (
                "Limit cuts to a few deliberate angle changes per clip."
                if allow_cuts
                else "Describe one continuous take in a single location, with no cuts."
            )
            return [
                CheckResult(
                    name="temporal",
                    status=CheckStatus.FAILED,
                    hard=True,
                    score=0.0,
                    detail=detail,
                    metrics=metrics,
                    fix_hint=hint,
                )
            ]
        score = linear_score(flicker, self._cfg.flicker_bad, self._cfg.flicker_good)
        unstable = flicker > self._cfg.flicker_bad
        detail = (
            "heavy flicker or erratic motion"
            if unstable
            else (f"multi-cut montage ({cut_count} cut(s))" if allow_cuts and cut_count else "")
        )
        return [
            CheckResult(
                name="temporal",
                status=CheckStatus.FAILED if unstable else CheckStatus.PASSED,
                score=score,
                detail=detail,
                fix_hint="Ask for smooth, stable motion and a steady camera." if unstable else None,
                metrics=metrics,
            )
        ]

    def _lipsync_checks(
        self,
        frames: list[np.ndarray],
        fps: float,
        shot: Shot,
        manifest: Manifest,
        audio_slice: Path,
    ) -> list[CheckResult]:
        names = ("face_presence", "lipsync")
        if shot.framing not in self._cfg.lipsync_framings:
            return [
                skipped(n, f"mouth too small to measure in a {shot.framing.value} shot")
                for n in names
            ]
        if self._mouth is None:
            return [skipped(n, "mouth tracking unavailable") for n in names]

        samples, sample_rate = sf.read(str(audio_slice), always_2d=True)
        envelope = lipsync.envelope_at_fps(samples.mean(axis=1), sample_rate, fps, len(frames))
        if lipsync.is_silent(envelope):
            return [skipped(n, "no vocal in this audio slice") for n in names]

        max_lag = max(1, round(self._cfg.lipsync_max_lag_seconds * fps))
        visible_count = len(manifest.visible_characters(shot))
        track_all = getattr(self._mouth, "track_all", None)
        if visible_count > 1 and callable(track_all):
            series_list = track_all(frames, max_faces=min(visible_count, 4))
        else:
            series_list = [self._mouth.track(frames)]
        measurements = [lipsync.measure_sync(series, envelope, max_lag) for series in series_list]
        present_measurements = [
            m for m in measurements if m.face_presence >= self._cfg.face_presence_min
        ]
        if present_measurements:
            result = max(present_measurements, key=lambda m: (m.correlation, m.face_presence))
        else:
            result = max(measurements, key=lambda m: (m.face_presence, m.correlation))
        presence_ok = result.face_presence >= self._cfg.face_presence_min
        presence = CheckResult(
            name="face_presence",
            status=CheckStatus.PASSED if presence_ok else CheckStatus.FAILED,
            hard=True,
            detail=""
            if presence_ok
            else f"face tracked in only {result.face_presence:.0%} of frames",
            fix_hint=None
            if presence_ok
            else "Keep the singer facing the camera for the whole clip; no turning away.",
            metrics={"face_presence": round(result.face_presence, 3)},
        )
        in_sync = result.correlation >= self._cfg.lipsync_min
        sync = CheckResult(
            name="lipsync",
            status=CheckStatus.PASSED if in_sync else CheckStatus.FAILED,
            hard=True,
            score=linear_score(result.correlation, self._cfg.lipsync_bad, self._cfg.lipsync_good),
            detail=""
            if in_sync
            else f"mouth movement does not follow the vocal (correlation {result.correlation:.2f})",
            fix_hint=None
            if in_sync
            else (
                "Keep the face frontal with the mouth unobstructed, and simplify body and "
                "camera movement so the mouth can follow the vocal."
            ),
            metrics={
                "correlation": round(result.correlation, 4),
                "lag_seconds": round(result.lag_frames / fps, 3),
            },
        )
        return [presence, sync]

    def _identity_check(
        self, frames: list[np.ndarray], fps: float, reference: IdentityReference | None
    ) -> CheckResult:
        if self._faces is None or reference is None:
            return skipped("identity", "face analysis or reference portrait unavailable")
        step = max(1, round(self._cfg.identity_sample_seconds * fps))
        similarities = []
        for frame in frames[::step]:
            faces = self._faces.faces(frame)
            if faces:
                sims = [s for f in faces if (s := reference.similarity(f)) is not None]
                if sims:
                    similarities.append(max(sims))
        # The median tolerates a few turned-away frames but catches real identity drift.
        similarity = float(np.median(similarities)) if similarities else None
        return identity_check(similarity, self._cfg.identity_hard_min, self._cfg.identity_good)

    @staticmethod
    def _sample(frames: list[np.ndarray], count: int) -> list[Path | np.ndarray]:
        indices = np.linspace(0, len(frames) - 1, num=min(count, len(frames))).round().astype(int)
        return [frames[i] for i in indices]
