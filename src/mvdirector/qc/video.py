# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Quality gate for video clips: technical, temporal, lip-sync, identity and vision checks."""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import soundfile as sf
from PIL import Image

from mvdirector import media, plugins
from mvdirector.config import QCConfig, VideoConfig
from mvdirector.errors import MediaToolError
from mvdirector.models import (
    DEFAULT_OPENING_SECONDS,
    AssetKind,
    CheckResult,
    CheckStatus,
    Manifest,
    QCReport,
    Shot,
    ShotType,
)
from mvdirector.qc import continuity as continuity_mod
from mvdirector.qc import lipsync
from mvdirector.qc.common import (
    dimensions_check,
    identity_check,
    vision_checks,
    vision_identity_check,
)
from mvdirector.qc.faces import Face, FaceAnalyzer, IdentityReference, MouthTracker
from mvdirector.qc.scoring import aggregate, floor_passing_score, linear_score, skipped
from mvdirector.qc.vision import VisionJudge

log = logging.getLogger(__name__)

_MOTION_SIDE = 64
_STATIC_DETAIL = "clip is almost static"


def _waive_static_for_passing_lipsync(checks: list[CheckResult]) -> list[CheckResult]:
    """Accept a near-static singing clip when its lip-sync check passed.

    In a lip-sync close-up the mouth is most of the motion, so whole-frame motion sits right at
    ``static_max_motion``; clips with strong lip-sync (correlation 0.8) were retried as "almost
    static". A truly frozen clip still fails, because a frozen mouth fails the lip-sync check.
    """
    sync = next((c for c in checks if c.name == "lipsync"), None)
    if sync is None or sync.status is not CheckStatus.PASSED:
        return checks
    waived: list[CheckResult] = []
    for check in checks:
        if (
            check.name == "temporal"
            and check.status is CheckStatus.FAILED
            and check.detail == _STATIC_DETAIL
        ):
            waived.append(
                CheckResult(
                    name="temporal",
                    status=CheckStatus.PASSED,
                    hard=False,
                    score=None,
                    detail="near-static framing accepted: the mouth movement passed lip-sync",
                    metrics={**check.metrics, "static_waived": 1.0},
                )
            )
        else:
            waived.append(check)
    return waived


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
        lipsync_scorer: lipsync.LipSyncScorer | None = None,
        qc_check_plugins: Sequence[str] = (),
        *,
        opening_seconds: float = DEFAULT_OPENING_SECONDS,
    ) -> None:
        self._qc = qc
        self._cfg = qc.video
        self._video = video
        self._faces = face_analyzer
        self._mouth = mouth_tracker
        self._vision = vision
        self._lipsync_scorer = lipsync_scorer
        self._qc_check_plugins = tuple(dict.fromkeys(qc_check_plugins))
        self._opening_seconds = opening_seconds

    def _opening_margin(self, shot: Shot) -> float:
        margin = self._qc.opening_threshold_margin
        if margin > 0.0 and shot.is_opening_at(self._opening_seconds):
            return margin
        return 0.0

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
        prev_tail_frames: list[np.ndarray] | None = None,
        beat_times: list[float] | None = None,
    ) -> QCReport:
        margin = self._opening_margin(shot)
        effective_pass = min(1.0, self._qc.pass_score + margin)
        info = media.probe(path)
        technical = self._technical_checks(info, shot, expected_size=expected_size)
        if any(c.status is CheckStatus.FAILED for c in technical):
            # A broken file makes every later measurement meaningless.
            return aggregate(technical, self._qc.weights, effective_pass)

        frames = media.extract_frames(path, self._cfg.analysis_width)
        fps = info.fps or float(self._video.fps)
        checks = [*technical, *self._temporal_checks(frames, allow_cuts=shot.allow_cuts)]
        if prev_tail_frames:
            checks.append(self._join_continuity_check(prev_tail_frames, frames))
        if beat_times and len(beat_times) >= 2:
            checks.append(self._beat_motion_check(frames, fps, beat_times))
        if shot.type is ShotType.LIPSYNC:
            checks += self._lipsync_checks(
                frames, fps, shot, manifest, audio_slice, pass_score=effective_pass
            )
            checks = _waive_static_for_passing_lipsync(checks)
        if shot.type.shows_singer:
            checks.append(
                self._identity_check(
                    frames, fps, reference, margin=margin, pass_score=effective_pass
                )
            )
        checks += vision_checks(
            self._vision,
            self._sample(frames, self._cfg.vision_frames),
            shot,
            manifest,
            AssetKind.VIDEO,
            min(1.0, self._cfg.vision_hard_min + margin),
            min(1.0, self._cfg.vision_character_hard_min + margin),
            reference_image,
            extra_references=extra_references,
            spec_hard_min=min(1.0, self._cfg.vision_spec_hard_min + margin),
        )
        checks += self._plugin_checks(frames, fps, shot, manifest, audio_slice)
        return aggregate(checks, self._qc.weights, effective_pass)

    def _plugin_checks(
        self,
        frames: list[np.ndarray],
        fps: float,
        shot: Shot,
        manifest: Manifest,
        audio_slice: Path,
    ) -> list[CheckResult]:
        if not self._qc_check_plugins:
            return []
        fail_hard = self._qc.on_missing_dependency == "fail"
        results: list[CheckResult] = []
        for name in self._qc_check_plugins:
            try:
                fn = plugins.load_plugin(plugins.QC_CHECKS_GROUP, name)
                out = fn(list(frames), fps, shot, manifest, audio_slice)
                if not isinstance(out, list) or not all(isinstance(c, CheckResult) for c in out):
                    raise TypeError(f"plugin {name!r} must return list[CheckResult]")
                sanitized: list[CheckResult] = []
                for check in out:
                    updates: dict[str, float] = {}
                    if check.score is not None:
                        if not math.isfinite(check.score):
                            raise ValueError(
                                f"plugin {name!r} returned non-finite score for {check.name!r}"
                            )
                        clamped_score = min(1.0, max(0.0, float(check.score)))
                        if clamped_score != check.score:
                            updates["score"] = clamped_score
                    if not math.isfinite(check.weight):
                        raise ValueError(
                            f"plugin {name!r} returned non-finite weight for {check.name!r}"
                        )
                    if check.weight < 0.0:
                        updates["weight"] = 0.0
                    for metric_key, metric_val in check.metrics.items():
                        if (
                            isinstance(metric_val, bool)
                            or not isinstance(metric_val, (int, float))
                            or not math.isfinite(float(metric_val))
                        ):
                            raise ValueError(
                                f"plugin {name!r} returned invalid metric {metric_key!r} "
                                f"for {check.name!r}"
                            )
                    sanitized.append(check.model_copy(update=updates) if updates else check)
                results.extend(sanitized)
            except Exception as exc:  # noqa: BLE001
                log.warning("QC plugin %r failed on %s: %s", name, shot.id, exc)
                if fail_hard:
                    results.append(
                        CheckResult(
                            name=name,
                            status=CheckStatus.FAILED,
                            hard=True,
                            score=0.0,
                            detail=f"plugin error: {exc}",
                        )
                    )
                else:
                    results.append(skipped(name, f"plugin error: {exc}"))
        return results

    @staticmethod
    def _join_continuity_check(
        prev_tail_frames: list[np.ndarray], frames: list[np.ndarray]
    ) -> CheckResult:
        """Advisory camera-motion continuity audit across a CONTINUE shot join."""
        report = continuity_mod.measure_join_continuity(prev_tail_frames, frames[:4])
        return CheckResult(
            name="join_continuity",
            status=CheckStatus.PASSED if report.smooth else CheckStatus.FAILED,
            score=None,
            weight=0.0,
            hard=False,
            detail="; ".join(report.issues),
            fix_hint=(
                "Avoid sudden camera velocity lurches or stalled first frames at the handoff seam."
                if report.issues
                else None
            ),
            metrics={
                "velocity_jump": report.velocity_jump,
                "speed_dip_ratio": report.speed_dip_ratio,
                "duplicate_boundary_frame": 1.0 if report.duplicate_boundary_frame else 0.0,
            },
        )

    @staticmethod
    def _beat_motion_check(
        frames: list[np.ndarray], fps: float, beat_times: list[float]
    ) -> CheckResult:
        """Advisory beat-to-visual-motion cross-correlation telemetry."""
        report = continuity_mod.measure_beat_motion_sync(frames, fps, beat_times, permutations=40)
        return CheckResult(
            name="beat_motion",
            status=CheckStatus.PASSED,
            score=None,
            weight=0.0,
            hard=False,
            metrics={
                "beat_correlation": report.correlation,
                "best_lag_seconds": report.best_lag_seconds,
                "chance_baseline": report.chance_baseline,
                "z_score": report.z_score,
            },
        )

    # ---------------------------------------------------------------- checks

    def _technical_checks(
        self,
        info: media.MediaInfo,
        shot: Shot,
        expected_size: tuple[int, int] | None = None,
    ) -> list[CheckResult]:
        if info.width is None or info.height is None or info.width <= 0 or info.height <= 0:
            return [
                CheckResult(
                    name="dimensions",
                    status=CheckStatus.FAILED,
                    hard=True,
                    detail="file has no valid video stream",
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
        flicker_excess = max(0.0, flicker - median)
        metrics: dict[str, float] = {
            "median_motion": round(median, 5),
            "peak_motion": round(peak, 4),
            "p95_motion": round(flicker, 4),
            "flicker_excess": round(flicker_excess, 4),
        }
        if allow_cuts:
            metrics["cut_count"] = float(cut_count)
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
        unstable = flicker > self._cfg.flicker_bad
        if unstable:
            return [
                CheckResult(
                    name="temporal",
                    status=CheckStatus.FAILED,
                    score=0.0,
                    detail="heavy flicker or erratic motion",
                    fix_hint="Ask for smooth, stable motion and a steady camera.",
                    metrics=metrics,
                )
            ]
        if median < self._cfg.static_max_motion:
            return [
                CheckResult(
                    name="temporal",
                    status=CheckStatus.FAILED,
                    hard=True,
                    score=0.0,
                    detail=_STATIC_DETAIL,
                    metrics=metrics,
                    fix_hint="Describe clear, continuous motion of the singer and the camera.",
                )
            ]
        # Measure erratic spikes above steady continuous motion (p95 - median) so smooth camera
        # moves or steady environmental motion (water, embers, snow) are not penalized as flicker,
        # while raw p95 > flicker_bad is already rejected above.
        score = linear_score(flicker_excess, self._cfg.flicker_bad, self._cfg.flicker_good)
        detail = f"multi-cut montage ({cut_count} cut(s))" if allow_cuts and cut_count else ""
        return [
            CheckResult(
                name="temporal",
                status=CheckStatus.PASSED,
                score=score,
                detail=detail,
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
        *,
        pass_score: float | None = None,
    ) -> list[CheckResult]:
        names = ("face_presence", "lipsync")
        if shot.framing not in self._cfg.lipsync_framings:
            return [
                skipped(n, f"mouth too small to measure in a {shot.framing.value} shot")
                for n in names
            ]
        if self._lipsync_scorer is None and self._mouth is None:
            return [skipped(n, "lip-sync scorer unavailable") for n in names]

        try:
            samples, sample_rate = sf.read(str(audio_slice), always_2d=True)
        except (sf.LibsndfileError, OSError) as exc:
            raise MediaToolError(f"Could not read audio slice {audio_slice}: {exc}") from exc
        if samples.size == 0 or sample_rate <= 0:
            return [skipped(n, "no vocal in this audio slice") for n in names]
        mono = samples.mean(axis=1)
        envelope = lipsync.envelope_at_fps(mono, sample_rate, fps, len(frames))
        if lipsync.is_silent(envelope):
            return [skipped(n, "no vocal in this audio slice") for n in names]

        singing_count = len(manifest.singing_characters(shot))
        needed_singers = min(singing_count, 2) if singing_count >= 2 else 1
        visible_count = max(len(manifest.visible_characters(shot)), needed_singers)
        max_faces = min(max(visible_count, 1), 4)
        multi_capable = False
        if self._lipsync_scorer is not None:
            score_all = getattr(self._lipsync_scorer, "score_all", None)
            if visible_count > 1 and callable(score_all):
                measurements = score_all(
                    frames,
                    fps,
                    mono,
                    sample_rate,
                    self._cfg.lipsync_max_lag_seconds,
                    max_faces=max_faces,
                )
                multi_capable = True
            else:
                measurements = [
                    self._lipsync_scorer.score(
                        frames,
                        fps,
                        mono,
                        sample_rate,
                        self._cfg.lipsync_max_lag_seconds,
                        max_faces=max_faces,
                    )
                ]
        else:
            assert self._mouth is not None
            max_lag = max(1, round(self._cfg.lipsync_max_lag_seconds * fps))
            track_all = getattr(self._mouth, "track_all", None)
            if visible_count > 1 and callable(track_all):
                series_list = track_all(frames, max_faces=max_faces)
                multi_capable = True
            else:
                series_list = [self._mouth.track(frames)]
            measurements = [
                lipsync.measure_sync(series, envelope, max_lag) for series in series_list
            ]

        if needed_singers >= 2 and multi_capable:
            if len(measurements) < needed_singers:
                result = lipsync.SyncMeasurement(correlation=0.0, lag_frames=0, face_presence=0.0)
            else:
                ordered = sorted(
                    measurements,
                    key=lambda m: (
                        m.face_presence >= self._cfg.face_presence_min,
                        m.correlation,
                        m.face_presence,
                    ),
                    reverse=True,
                )
                result = ordered[needed_singers - 1]
        else:
            present_measurements = [
                m for m in measurements if m.face_presence >= self._cfg.face_presence_min
            ]
            if present_measurements:
                result = max(present_measurements, key=lambda m: (m.correlation, m.face_presence))
            elif measurements:
                result = max(measurements, key=lambda m: (m.face_presence, m.correlation))
            else:
                result = lipsync.SyncMeasurement(correlation=0.0, lag_frames=0, face_presence=0.0)

        presence_ok = result.face_presence >= self._cfg.face_presence_min
        tracked_singers = sum(
            1 for m in measurements if m.face_presence >= self._cfg.face_presence_min
        )
        if presence_ok:
            presence_detail, presence_hint = "", None
        elif needed_singers >= 2 and multi_capable:
            presence_detail = (
                f"both singers must stay tracked (tracked {tracked_singers} of "
                f"{needed_singers} singing faces)"
            )
            presence_hint = (
                "Keep both singers facing the camera in a medium close-up two-shot for the whole "
                "clip; neither singer may turn away."
            )
        else:
            presence_detail = f"face tracked in only {result.face_presence:.0%} of frames"
            presence_hint = "Keep the singer facing the camera for the whole clip; no turning away."
        presence = CheckResult(
            name="face_presence",
            status=CheckStatus.PASSED if presence_ok else CheckStatus.FAILED,
            hard=True,
            detail=presence_detail,
            fix_hint=presence_hint,
            metrics={"face_presence": round(result.face_presence, 3)},
        )
        in_sync = result.correlation >= self._cfg.lipsync_min
        blocking = self._cfg.lipsync_blocking
        lag_fps = result.eval_fps or fps
        lag_seconds = round(result.lag_frames / lag_fps, 3)
        sync_metrics: dict[str, float] = {
            "correlation": round(result.correlation, 4),
            "lag_seconds": lag_seconds,
        }
        if result.confidence is not None:
            sync_metrics["sync_confidence"] = round(result.confidence, 4)
        # Compute recommended playback offset compensation (sync_delay_seconds) and viewer-visible
        # face-fraction desync_risk (inspired by VisualFrisson/ComfyUI-OneShotMV &
        # JavaDerek/music-video-pipeline).
        conf_ok = result.confidence is None or result.confidence >= self._cfg.lipsync_min_confidence
        if conf_ok and result.correlation >= self._cfg.lipsync_bad:
            sync_metrics["sync_delay_seconds"] = lag_seconds
        else:
            sync_metrics["sync_delay_seconds"] = 0.0
        if self._faces is not None and frames:
            sample_step = max(1, len(frames) // 5)
            face_fracs = [
                f[0].height_fraction for fr in frames[::sample_step] if (f := self._faces.faces(fr))
            ]
            if face_fracs:
                med_face = float(np.median(face_fracs))
                sync_metrics["median_face_fraction"] = round(med_face, 4)
                visible_weight = (
                    1.0
                    if med_face >= self._cfg.desync_face_fraction_min
                    else med_face / max(self._cfg.desync_face_fraction_min, 1e-6)
                )
                sync_metrics["desync_risk"] = round(abs(lag_seconds) * visible_weight, 4)
        if in_sync:
            sync_detail, sync_hint = "", None
        elif needed_singers >= 2 and multi_capable:
            sync_detail = (
                "both singers must lip-sync together; co-singer mouth movement does not follow "
                f"the vocal (correlation {result.correlation:.2f})"
            )
            sync_hint = (
                "Both singers must visibly sing together with clear, unobstructed mouth movement "
                "in time with the vocal."
            )
        else:
            sync_detail = (
                f"mouth movement does not follow the vocal (correlation {result.correlation:.2f})"
            )
            sync_hint = (
                "Keep the face frontal with the mouth unobstructed, and simplify body and "
                "camera movement so the mouth can follow the vocal."
            )
        if in_sync:
            # A passing lip-sync check must score at least pass_score so its heavy weight cannot
            # drag an otherwise-passing clip below the overall pass mark and waste retry renders.
            sync_score: float | None = floor_passing_score(
                result.correlation,
                self._cfg.lipsync_min,
                self._cfg.lipsync_good,
                self._qc.pass_score if pass_score is None else pass_score,
            )
        elif blocking:
            sync_score = linear_score(
                result.correlation, self._cfg.lipsync_bad, self._cfg.lipsync_good
            )
        else:
            # A weak advisory result carries no score, so it cannot pull a clip below the pass mark.
            sync_score = None
        sync = CheckResult(
            name="lipsync",
            status=CheckStatus.PASSED if in_sync else CheckStatus.FAILED,
            hard=blocking,
            score=sync_score,
            detail=sync_detail,
            fix_hint=sync_hint,
            metrics=sync_metrics,
        )
        return [presence, sync]

    def _identity_check(
        self,
        frames: list[np.ndarray],
        fps: float,
        reference: IdentityReference | None,
        *,
        margin: float = 0.0,
        pass_score: float | None = None,
    ) -> CheckResult:
        if self._faces is None or reference is None:
            return skipped("identity", "face analysis or reference portrait unavailable")
        effective_pass = self._qc.pass_score if pass_score is None else pass_score
        if self._qc.face_backend != "insightface" and self._qc.vision_identity:
            samples = self._vision_identity_samples(frames)
            has_embeddings = any(f.embedding is not None for _, faces in samples for f in faces)
            if not has_embeddings:
                # No embeddings (MediaPipe): the vision model compares face crops from a few frames.
                eff_vi_min = min(1.0, self._cfg.vision_identity_hard_min + margin)
                return vision_identity_check(
                    self._vision,
                    samples,
                    reference,
                    min_face_px=self._cfg.vision_identity_min_face_px,
                    hard_min=eff_vi_min,
                    good=max(self._cfg.vision_identity_good, eff_vi_min),
                    pass_score=effective_pass,
                )
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
        eff_id_min = min(1.0, self._cfg.identity_hard_min + margin)
        return identity_check(similarity, eff_id_min, max(self._cfg.identity_good, eff_id_min))

    def _vision_identity_samples(
        self, frames: list[np.ndarray]
    ) -> list[tuple[np.ndarray, list[Face]]]:
        """Evenly spaced frames (away from the very first and last) that contain a face."""
        if self._faces is None or not frames:
            return []
        count = self._cfg.vision_identity_frames
        if len(frames) > 2:
            positions = np.linspace(0, len(frames) - 1, num=count + 2)[1:-1]
        elif len(frames) == 2 and count >= 2:
            positions = np.array([0.0, 1.0])
        else:
            positions = np.array([0.0])
        samples: list[tuple[np.ndarray, list[Face]]] = []
        for index in sorted({int(round(float(p))) for p in positions}):
            faces = self._faces.faces(frames[index])
            if faces:
                samples.append((frames[index], faces))
        return samples

    @staticmethod
    def _sample(frames: list[np.ndarray], count: int) -> list[Path | np.ndarray]:
        if count <= 0 or not frames:
            return []
        indices = np.linspace(0, len(frames) - 1, num=min(count, len(frames))).round().astype(int)
        return [frames[i] for i in indices]
