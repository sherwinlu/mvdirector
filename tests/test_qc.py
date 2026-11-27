# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import numpy as np
import pytest

from mvdirector.config import ProjectConfig
from mvdirector.models import (
    AssetState,
    AudioSlice,
    CheckResult,
    CheckStatus,
    Framing,
    Manifest,
    Shot,
    ShotType,
)
from mvdirector.qc import lipsync
from mvdirector.qc.faces import Face, IdentityReference
from mvdirector.qc.image import ImageQC, sharpness
from mvdirector.qc.scoring import aggregate, linear_score, skipped
from mvdirector.qc.video import frame_differences


def _check(name, status=CheckStatus.PASSED, score=None, hard=False):
    return CheckResult(name=name, status=status, score=score, hard=hard)


def test_linear_score_handles_both_directions():
    assert linear_score(0.3, 0.1, 0.5) == pytest.approx(0.5)
    assert linear_score(0.9, 0.1, 0.5) == 1.0 and linear_score(0.0, 0.1, 0.5) == 0.0
    # Lower-is-better metric: bad > good.
    assert linear_score(0.03, 0.12, 0.03) == 1.0 and linear_score(0.12, 0.12, 0.03) == 0.0


def test_aggregate_weights_scores_and_ignores_skipped():
    report = aggregate(
        [_check("lipsync", score=1.0), _check("sharpness", score=0.0), skipped("identity", "n/a")],
        weights={"lipsync": 3.0, "sharpness": 1.0},
        pass_score=0.7,
    )
    assert report.score == 0.75 and report.passed


def test_hard_failure_rejects_even_with_high_score():
    report = aggregate(
        [_check("lipsync", score=1.0), _check("face_present", CheckStatus.FAILED, hard=True)],
        weights={},
        pass_score=0.5,
    )
    assert not report.passed and report.has_hard_failure
    assert report.issues == ["face_present"]


def test_low_score_fails_without_hard_failure():
    report = aggregate([_check("temporal", score=0.4)], weights={}, pass_score=0.7)
    assert not report.passed and not report.has_hard_failure


# ------------------------------------------------------------------ lip-sync maths


def _singing(n_frames=120, fps=24.0, seed=0):
    """A loudness envelope and a mouth series that follows it with some noise."""
    rng = np.random.default_rng(seed)
    time = np.arange(n_frames) / fps
    envelope = np.clip(
        np.sin(2 * np.pi * 1.5 * time) + 0.3 * np.sin(2 * np.pi * 0.4 * time), 0, None
    )
    mouth = 0.08 * envelope + 0.01 * rng.standard_normal(n_frames)
    return envelope, mouth


def test_synced_mouth_scores_high():
    envelope, mouth = _singing()
    result = lipsync.measure_sync(mouth, envelope, max_lag_frames=6)
    assert result.correlation > 0.8 and abs(result.lag_frames) <= 1
    assert result.face_presence == 1.0


def test_unrelated_mouth_scores_low():
    envelope, _ = _singing()
    random_mouth = np.random.default_rng(1).standard_normal(len(envelope))
    assert lipsync.measure_sync(random_mouth, envelope, max_lag_frames=6).correlation < 0.3


def test_static_mouth_scores_zero():
    envelope, _ = _singing()
    result = lipsync.measure_sync(np.full(len(envelope), 0.02), envelope, max_lag_frames=6)
    assert result.correlation == 0.0


def test_lag_is_detected_and_reported():
    envelope, mouth = _singing(n_frames=160)
    delayed = np.roll(mouth, 4)
    result = lipsync.measure_sync(delayed, envelope, max_lag_frames=6)
    assert result.lag_frames == 4 and result.correlation > 0.7


def test_missing_faces_are_interpolated_and_counted():
    envelope, mouth = _singing()
    mouth[::5] = np.nan
    result = lipsync.measure_sync(mouth, envelope, max_lag_frames=6)
    assert result.face_presence == pytest.approx(0.8)
    assert result.correlation > 0.7


def test_envelope_at_fps_tracks_loudness():
    sample_rate, fps = 24000, 24.0
    samples = np.concatenate([np.zeros(sample_rate), 0.5 * np.ones(sample_rate)])
    envelope = lipsync.envelope_at_fps(samples, sample_rate, fps, 48)
    assert np.allclose(envelope[:24], 0.0) and np.allclose(envelope[24:], 0.5)
    assert lipsync.is_silent(envelope[:24]) and not lipsync.is_silent(envelope)


# ------------------------------------------------------------------- image checks


def test_sharpness_separates_detail_from_flat():
    noise = np.random.default_rng(0).integers(0, 255, (256, 256, 3), dtype=np.uint8)
    assert sharpness(noise) > 1000 > 1 > sharpness(np.full((256, 256, 3), 128, dtype=np.uint8))


def test_frame_differences_measure_motion():
    still = [np.zeros((32, 32, 3), dtype=np.uint8)] * 3
    cut = [np.zeros((32, 32, 3), dtype=np.uint8), np.full((32, 32, 3), 255, dtype=np.uint8)]
    assert frame_differences(still).max() == 0.0
    assert frame_differences(cut)[0] == pytest.approx(1.0)


class FakeFaces:
    def __init__(self, faces):
        self._faces = faces

    def faces(self, image):
        return self._faces


def _shot(shot_type=ShotType.LIPSYNC, framing=Framing.CLOSE_UP):
    return Shot(
        id="s001",
        index=0,
        start=0.0,
        end=4.0,
        type=shot_type,
        framing=framing,
        audio_slice=AudioSlice(path="a.wav", start=0.0, end=4.25, head_handle=0.0),
        num_frames=105,
        still=AssetState(attempts_allowed=4),
        video=AssetState(attempts_allowed=4),
    )


@pytest.fixture
def still_path(tmp_path, sample_png):
    path = tmp_path / "still.png"
    path.write_bytes(sample_png)
    return path


def _image_report(still_path, faces, shot, reference_vector=(1.0, 0.0)):
    config = ProjectConfig()
    analyzer = FakeFaces(faces)
    reference_analyzer = FakeFaces([Face(0.5, np.array(reference_vector))])
    reference = IdentityReference(reference_analyzer, still_path)
    gate = ImageQC(config.qc, config.video, analyzer, vision=None)
    manifest = Manifest(title="t", duration=4.0, fps=24, shots=[shot])
    return gate.evaluate(still_path, shot, manifest, reference)


def _status(report, name):
    return next(c for c in report.checks if c.name == name).status


def test_good_still_passes(still_path):
    report = _image_report(still_path, [Face(0.4, np.array([1.0, 0.0]))], _shot())
    assert report.passed
    assert _status(report, "vision_spec") is CheckStatus.SKIPPED


def test_still_without_face_is_rejected(still_path):
    report = _image_report(still_path, [], _shot())
    assert not report.passed and "face_present: no face detected" in report.issues


def test_wrong_person_is_rejected(still_path):
    report = _image_report(still_path, [Face(0.4, np.array([0.0, 1.0]))], _shot())
    assert not report.passed and _status(report, "identity") is CheckStatus.FAILED
    assert report.fix_hints


def test_small_face_is_hard_failure_only_for_lipsync(still_path):
    tiny = [Face(0.1, np.array([1.0, 0.0]))]
    lip = _image_report(still_path, tiny, _shot(ShotType.LIPSYNC))
    performance = _image_report(still_path, tiny, _shot(ShotType.PERFORMANCE))
    assert not lip.passed and lip.has_hard_failure
    assert _status(performance, "face_framing") is CheckStatus.FAILED
    assert not performance.has_hard_failure


def test_broll_skips_face_checks(still_path):
    report = _image_report(still_path, [], _shot(ShotType.BROLL, Framing.WIDE))
    assert report.passed and all(c.name != "face_present" for c in report.checks)


def test_detection_only_face_analyzer_checks_presence_and_framing_and_skips_identity(still_path):
    report = _image_report(still_path, [Face(0.4, None)], _shot())
    assert report.passed
    assert _status(report, "face_present") is CheckStatus.PASSED
    assert _status(report, "face_framing") is CheckStatus.PASSED
    assert _status(report, "identity") is CheckStatus.SKIPPED


def test_two_shot_image_framing_uses_matching_identity_face(still_path):
    # Foreground non-singer face is large (0.6), but active singer face
    # matches identity and is small (0.1).
    faces = [Face(0.6, np.array([0.0, 1.0])), Face(0.1, np.array([1.0, 0.0]))]
    report = _image_report(still_path, faces, _shot(ShotType.LIPSYNC, Framing.CLOSE_UP))
    assert not report.passed
    assert _status(report, "identity") is CheckStatus.PASSED
    assert _status(report, "face_framing") is CheckStatus.FAILED


def test_two_shot_video_lipsync_selects_synced_singer_track(tmp_path, monkeypatch):
    import soundfile as sf

    from mvdirector import media
    from mvdirector.models import Character
    from mvdirector.qc.video import VideoQC

    envelope, synced_mouth = _singing(n_frames=96, fps=24.0, seed=2)
    samples = np.repeat(envelope, 1000).astype(np.float32)
    audio_path = tmp_path / "slice.wav"
    sf.write(str(audio_path), samples, 24000)

    class TwoFaceMouthTracker:
        def track(self, frames):
            return np.full(len(frames), 0.02)  # single-face slot hits the silent partner

        def track_all(self, frames, max_faces=2):
            return [np.full(len(frames), 0.02), synced_mouth]

    monkeypatch.setattr(
        media,
        "probe",
        lambda p: media.MediaInfo(width=1280, height=720, fps=24.0, duration=4.25, has_audio=True),
    )
    rng = np.random.default_rng(3)
    fake_frames = [
        rng.integers(0, 255, (64, 64, 3), dtype=np.uint8) for _ in range(len(synced_mouth))
    ]
    monkeypatch.setattr(media, "extract_frames", lambda p, width: fake_frames)

    config = ProjectConfig()
    gate = VideoQC(config.qc, config.video, None, TwoFaceMouthTracker(), None)
    shot = _shot(ShotType.LIPSYNC, Framing.MEDIUM_CLOSE)
    shot.character = "odalie"
    shot.characters = ["odalie", "elias"]
    manifest = Manifest(
        title="Duet",
        duration=4.0,
        fps=24,
        characters={
            "elias": Character(name="Elias", appearance="Elias."),
            "odalie": Character(name="Odalie", appearance="Odalie."),
        },
        shots=[shot],
    )
    video_path = tmp_path / "clip.mp4"
    video_path.write_bytes(b"fake")
    report = gate.evaluate(video_path, shot, manifest, audio_path, None)
    assert report.passed
    assert _status(report, "face_presence") is CheckStatus.PASSED
    assert _status(report, "lipsync") is CheckStatus.PASSED


def test_quality_gates_fail_closed_by_default_and_report_summary_when_skipped(monkeypatch):
    import sys

    from mvdirector.errors import DependencyMissing
    from mvdirector.qc.factory import build_quality_gates

    monkeypatch.setitem(sys.modules, "mediapipe", None)
    cfg = ProjectConfig()
    assert cfg.qc.on_missing_dependency == "fail"
    with pytest.raises(DependencyMissing, match="mediapipe"):
        build_quality_gates(cfg, llm=None)

    cfg.qc.on_missing_dependency = "skip"
    gates = build_quality_gates(cfg, llm=None)
    active, skipped_checks = gates.check_summary()
    assert active == ["dimensions", "sharpness", "duration", "temporal"]
    skipped_map = dict(skipped_checks)
    assert "mediapipe unavailable" in skipped_map["face_present, face_framing, no_people"]
    assert "mediapipe unavailable" in skipped_map["face_presence, lipsync"]
    assert "llm.enabled is false" in skipped_map["vision_spec, vision_artifacts, vision_character"]


def test_cli_prints_qc_summary_and_status_counts(project, analysis, monkeypatch):
    import sys

    import yaml
    from typer.testing import CliRunner

    from mvdirector.cli import app
    from mvdirector.models import Attempt, QCReport, RetryStrategy
    from mvdirector.planning.planner import build_manifest

    monkeypatch.setitem(sys.modules, "mediapipe", None)
    cfg = project.config
    cfg.llm.enabled = False
    cfg.qc.on_missing_dependency = "skip"
    (project.root / "project.yaml").write_text(yaml.safe_dump(cfg.model_dump(mode="json")))
    manifest = build_manifest(cfg, analysis, "lyrics", "style", llm=None)
    manifest.shots[0].still.attempts.append(
        Attempt(
            number=1,
            strategy=RetryStrategy.INITIAL,
            seed=1,
            prompt="p",
            qc=QCReport(
                passed=True,
                score=0.9,
                checks=[
                    _check("dimensions", CheckStatus.PASSED, 1.0, True),
                    skipped("face_present", "mediapipe missing"),
                ],
            ),
        )
    )
    project.save_manifest(manifest)

    status_res = CliRunner().invoke(app, ["status", "-p", str(project.root)])
    assert status_res.exit_code == 0
    assert "QC checks across attempts:" in status_res.output
    assert "dimensions" in status_res.output and "passed=1" in status_res.output
    assert "face_present" in status_res.output and "skipped=1" in status_res.output

    from mvdirector.errors import ComfyConnectionError

    def stop_comfy(*a, **kw):
        raise ComfyConnectionError("stop-at-comfy")

    monkeypatch.setattr("mvdirector.comfy.provider.ComfyClient", stop_comfy)
    gen_res = CliRunner().invoke(app, ["stills", "--skip-preflight", "-p", str(project.root)])
    assert "QC active: dimensions, sharpness, duration, temporal" in gen_res.output
    assert "QC skipped:" in gen_res.output and "mediapipe unavailable" in gen_res.output


def test_mediapipe_tasks_api_is_available_when_installed():
    """Guards against the dependency drifting away from the Tasks Vision API again."""
    pytest.importorskip("mediapipe")
    from mvdirector.qc.faces import _import_mediapipe_tasks

    _, base_options, vision = _import_mediapipe_tasks("FaceDetector")
    assert base_options is not None
    assert hasattr(vision, "FaceDetector") and hasattr(vision, "FaceLandmarker")


def test_mediapipe_model_resolution_uses_cache_and_explicit_paths(tmp_path, monkeypatch):
    import httpx

    from mvdirector.errors import DependencyMissing
    from mvdirector.qc.faces import _ensure_mediapipe_model

    monkeypatch.setenv("MVD_MEDIAPIPE_CACHE_DIR", str(tmp_path / "cache"))
    cached = tmp_path / "cache" / "model.tflite"
    cached.parent.mkdir(parents=True)
    cached.write_bytes(b"x")
    assert _ensure_mediapipe_model("model.tflite", "https://invalid.example/m") == cached

    explicit = tmp_path / "explicit.task"
    explicit.write_bytes(b"y")
    assert (
        _ensure_mediapipe_model("o", "https://invalid.example/o", model_path=explicit) == explicit
    )
    with pytest.raises(DependencyMissing, match="not found"):
        _ensure_mediapipe_model("o", "https://invalid.example/o", model_path=tmp_path / "missing")

    def offline(*_args, **_kwargs):
        raise httpx.ConnectError("offline")

    # A failed download is a missing dependency with guidance, never a raw httpx error.
    monkeypatch.setattr(httpx.Client, "stream", offline)
    with pytest.raises(DependencyMissing, match="Failed to download"):
        _ensure_mediapipe_model("missing.tflite", "https://invalid.example/missing")
    assert not (tmp_path / "cache" / "missing.tflite").exists()


def test_cli_version_flag_prints_package_version():
    from importlib.metadata import version

    from typer.testing import CliRunner

    from mvdirector.cli import app

    result = CliRunner().invoke(app, ["--version"])
    assert result.exit_code == 0
    assert result.output.strip() == f"mvd {version('mvdirector')}"


def test_calibrated_lipsync_thresholds_reject_noise_on_typical_clips():
    cfg = ProjectConfig().qc.video
    fps = 24.0
    max_lag = max(1, round(cfg.lipsync_max_lag_seconds * fps))
    assert cfg.lipsync_min >= 0.40 and max_lag >= 6

    rng = np.random.default_rng(7)
    n = int(4.0 * fps)
    false_passes = sum(
        1
        for _ in range(100)
        if lipsync.measure_sync(
            np.abs(rng.standard_normal(n)), np.abs(rng.standard_normal(n)), max_lag
        ).correlation
        >= cfg.lipsync_min
    )
    assert false_passes <= 5

    envelope, synced_mouth = _singing(n_frames=n, fps=fps, seed=4)
    assert lipsync.measure_sync(synced_mouth, envelope, max_lag).correlation >= cfg.lipsync_good
    # A genuinely synced clip with a 6-frame (250ms) constant offset still passes without a cliff.
    offset_mouth = np.roll(synced_mouth, 6)
    assert lipsync.measure_sync(offset_mouth, envelope, max_lag).correlation >= cfg.lipsync_min


# --------------------------------------------------------- MediaPipe Tasks API


def _install_fake_mediapipe(monkeypatch, *, detector_cls=None, landmarker_cls=None):
    import sys
    from types import ModuleType, SimpleNamespace

    mp_mod = ModuleType("mediapipe")
    tasks_mod = ModuleType("mediapipe.tasks")
    python_mod = ModuleType("mediapipe.tasks.python")
    vision_mod = ModuleType("mediapipe.tasks.python.vision")

    class FakeImage:
        def __init__(self, image_format, data):
            self.image_format = image_format
            self.data = data

    mp_mod.Image = FakeImage  # type: ignore[attr-defined]
    mp_mod.ImageFormat = SimpleNamespace(SRGB=1)  # type: ignore[attr-defined]

    class FakeBaseOptions:
        def __init__(self, model_asset_path):
            self.model_asset_path = model_asset_path

    class FakeFaceDetectorOptions:
        def __init__(self, base_options, running_mode, min_detection_confidence=0.5):
            self.base_options = base_options
            self.running_mode = running_mode
            self.min_detection_confidence = min_detection_confidence

    class FakeFaceLandmarkerOptions:
        def __init__(self, base_options, running_mode, num_faces=1):
            self.base_options = base_options
            self.running_mode = running_mode
            self.num_faces = num_faces

    vision_mod.RunningMode = SimpleNamespace(IMAGE="IMAGE", VIDEO="VIDEO")  # type: ignore[attr-defined]
    vision_mod.FaceDetectorOptions = FakeFaceDetectorOptions  # type: ignore[attr-defined]
    vision_mod.FaceLandmarkerOptions = FakeFaceLandmarkerOptions  # type: ignore[attr-defined]
    vision_mod.FaceDetector = detector_cls or object  # type: ignore[attr-defined]
    vision_mod.FaceLandmarker = landmarker_cls or object  # type: ignore[attr-defined]

    python_mod.BaseOptions = FakeBaseOptions  # type: ignore[attr-defined]
    python_mod.vision = vision_mod  # type: ignore[attr-defined]
    tasks_mod.python = python_mod  # type: ignore[attr-defined]
    mp_mod.tasks = tasks_mod  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "mediapipe", mp_mod)
    monkeypatch.setitem(sys.modules, "mediapipe.tasks", tasks_mod)
    monkeypatch.setitem(sys.modules, "mediapipe.tasks.python", python_mod)
    monkeypatch.setitem(sys.modules, "mediapipe.tasks.python.vision", vision_mod)


def test_mediapipe_face_analyzer_uses_tasks_api_and_normalizes_height(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from mvdirector.qc.faces import MediaPipeFaceAnalyzer

    captured = {}

    class FakeDetector:
        @classmethod
        def create_from_options(cls, options):
            captured["options"] = options
            return cls()

        def detect(self, mp_image):
            captured["image"] = mp_image
            return SimpleNamespace(
                detections=[
                    SimpleNamespace(bounding_box=SimpleNamespace(height=120)),
                    SimpleNamespace(bounding_box=SimpleNamespace(height=240)),
                ]
            )

        def close(self):
            captured["closed"] = True

    _install_fake_mediapipe(monkeypatch, detector_cls=FakeDetector)
    model_file = tmp_path / "blaze_face_full_range.tflite"
    model_file.write_bytes(b"fake-tflite")

    analyzer = MediaPipeFaceAnalyzer(model_path=model_file)
    faces = analyzer.faces(np.zeros((480, 640, 3), dtype=np.uint8))
    analyzer.close()

    assert captured["options"].running_mode == "IMAGE"
    assert captured["options"].base_options.model_asset_path == str(model_file)
    assert [f.height_fraction for f in faces] == [pytest.approx(0.5), pytest.approx(0.25)]
    assert all(f.embedding is None for f in faces)
    assert captured["closed"] is True


def test_mediapipe_mouth_tracker_uses_video_tasks_api_and_tracks_two_faces(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from mvdirector.qc.faces import MediaPipeMouthTracker

    def _face(x, gap):
        pts = [SimpleNamespace(x=0.0, y=0.0)] * 478
        pts[1] = SimpleNamespace(x=x, y=0.5)
        pts[10] = SimpleNamespace(x=x, y=0.2)
        pts[152] = SimpleNamespace(x=x, y=0.7)  # face_height = 0.5
        pts[13] = SimpleNamespace(x=x, y=0.5)
        pts[14] = SimpleNamespace(x=x, y=0.5 + gap)
        return pts

    captured = {"timestamps": []}

    class FakeLandmarker:
        def __init__(self, options):
            captured["options"] = options

        @classmethod
        def create_from_options(cls, options):
            return cls(options)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            captured["closed"] = True

        def detect_for_video(self, mp_image, timestamp_ms):
            captured["timestamps"].append(timestamp_ms)
            idx = len(captured["timestamps"]) - 1
            if idx == 0:
                return SimpleNamespace(face_landmarks=[_face(0.8, 0.10), _face(0.2, 0.05)])
            return SimpleNamespace(face_landmarks=[_face(0.2, 0.15), _face(0.8, 0.20)])

    _install_fake_mediapipe(monkeypatch, landmarker_cls=FakeLandmarker)
    model_file = tmp_path / "face_landmarker.task"
    model_file.write_bytes(b"fake-task")

    tracker = MediaPipeMouthTracker(model_path=model_file)
    frames = [np.zeros((64, 64, 3), dtype=np.uint8) for _ in range(2)]
    tracks = tracker.track_all(frames, max_faces=2)

    assert captured["options"].running_mode == "VIDEO"
    assert captured["options"].num_faces == 2
    assert captured["timestamps"] == [0, 33]
    assert captured["closed"] is True
    assert tracks[0] == pytest.approx([0.10, 0.30])
    assert tracks[1] == pytest.approx([0.20, 0.40])


def test_corrupt_cached_mediapipe_model_is_downloaded_again_once(tmp_path, monkeypatch):
    import httpx

    from mvdirector.errors import DependencyMissing
    from mvdirector.qc import faces

    cache_dir = tmp_path / "mp_cache"
    monkeypatch.setenv("MVD_MEDIAPIPE_CACHE_DIR", str(cache_dir))
    cache_dir.mkdir()
    (cache_dir / "demo.task").write_bytes(b"corrupt")
    downloads = 0

    def handler(request):
        nonlocal downloads
        downloads += 1
        return httpx.Response(200, content=b"good-bytes", request=request)

    real_client = httpx.Client
    monkeypatch.setattr(
        faces.httpx,
        "Client",
        lambda **kwargs: real_client(transport=httpx.MockTransport(handler), **kwargs),
    )

    def create(path):
        if path.read_bytes() == b"corrupt":
            raise RuntimeError("Unable to open model")
        return "loaded"

    loaded, resolved = faces._load_mediapipe_model(create, "demo.task", "https://x/demo.task", None)
    assert (loaded, downloads) == ("loaded", 1)
    assert resolved.read_bytes() == b"good-bytes"

    # An explicit model path is never deleted or re-downloaded.
    explicit = tmp_path / "mine.task"
    explicit.write_bytes(b"corrupt")
    with pytest.raises(DependencyMissing, match="could not load"):
        faces._load_mediapipe_model(create, "demo.task", "https://x/demo.task", explicit)
    assert explicit.read_bytes() == b"corrupt" and downloads == 1


def test_quality_gates_close_releases_face_backends():
    from mvdirector.qc.factory import QualityGates

    class Closable:
        closed = False

        def close(self):
            self.closed = True

    analyzer, tracker = Closable(), Closable()
    gates = QualityGates(image=None, video=None, face_analyzer=analyzer, mouth_tracker=tracker)  # type: ignore[arg-type]
    gates.close()
    assert analyzer.closed and tracker.closed
    QualityGates(image=None, video=None, face_analyzer=None).close()  # type: ignore[arg-type]


def test_ensure_mediapipe_model_downloads_and_caches_atomically(tmp_path, monkeypatch):
    import httpx

    from mvdirector.errors import DependencyMissing
    from mvdirector.qc import faces

    cache_dir = tmp_path / "mp_cache"
    monkeypatch.setenv("MVD_MEDIAPIPE_CACHE_DIR", str(cache_dir))

    calls = 0

    def handler(request):
        nonlocal calls
        calls += 1
        if "broken" in str(request.url):
            return httpx.Response(503, request=request)
        return httpx.Response(200, content=b"model-bytes", request=request)

    transport = httpx.MockTransport(handler)
    real_client = httpx.Client
    monkeypatch.setattr(
        faces.httpx,
        "Client",
        lambda **kwargs: real_client(transport=transport, **kwargs),
    )

    first = faces._ensure_mediapipe_model("demo.task", "https://example.com/demo.task")
    second = faces._ensure_mediapipe_model("demo.task", "https://example.com/demo.task")
    assert first == cache_dir / "demo.task"
    assert second == first
    assert first.read_bytes() == b"model-bytes"
    assert calls == 1

    with pytest.raises(DependencyMissing, match="Failed to download MediaPipe model"):
        faces._ensure_mediapipe_model("broken.task", "https://example.com/broken.task")
    assert not list(cache_dir.glob("*.part"))


def test_ensure_mediapipe_model_skips_content_length_for_gzip_and_sets_0644(tmp_path, monkeypatch):
    import gzip
    import stat

    import httpx

    from mvdirector.qc import faces

    cache_dir = tmp_path / "mp_cache"
    monkeypatch.setenv("MVD_MEDIAPIPE_CACHE_DIR", str(cache_dir))

    raw = b"uncompressed-model-payload" * 10
    compressed = gzip.compress(raw)

    def handler(request):
        return httpx.Response(
            200,
            content=compressed,
            headers={
                "content-encoding": "gzip",
                "content-length": str(len(compressed)),
            },
            request=request,
        )

    transport = httpx.MockTransport(handler)
    real_client = httpx.Client
    monkeypatch.setattr(
        faces.httpx,
        "Client",
        lambda **kwargs: real_client(transport=transport, **kwargs),
    )

    cached = faces._ensure_mediapipe_model("gzipped.task", "https://example.com/gzipped.task")
    assert cached.read_bytes() == raw
    assert stat.S_IMODE(cached.stat().st_mode) == 0o644


def test_validate_rgb_image_and_analyzer_lifecycle(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from mvdirector.qc.faces import MediaPipeFaceAnalyzer, _validate_rgb_image

    assert _validate_rgb_image(np.zeros((4, 4, 3), dtype=np.uint8)) is True
    assert _validate_rgb_image(np.zeros((0, 4, 3), dtype=np.uint8)) is False
    with pytest.raises(ValueError, match="HxWx3"):
        _validate_rgb_image(np.zeros((4, 4, 4), dtype=np.uint8))

    close_calls = 0

    class FakeDetector:
        @classmethod
        def create_from_options(cls, options):
            return cls()

        def detect(self, mp_image):
            return SimpleNamespace(detections=[])

        def close(self):
            nonlocal close_calls
            close_calls += 1

    _install_fake_mediapipe(monkeypatch, detector_cls=FakeDetector)
    model_file = tmp_path / "blaze_face_full_range.tflite"
    model_file.write_bytes(b"fake-tflite")

    with MediaPipeFaceAnalyzer(model_path=model_file) as analyzer:
        assert analyzer.faces(np.zeros((0, 10, 3), dtype=np.uint8)) == []
    analyzer.close()  # idempotent second close

    assert close_calls == 1
    with pytest.raises(RuntimeError, match="closed"):
        analyzer.faces(np.zeros((10, 10, 3), dtype=np.uint8))


def test_mouth_measurement_handles_short_or_none_landmarks():
    from types import SimpleNamespace

    from mvdirector.qc.faces import MediaPipeMouthTracker

    assert MediaPipeMouthTracker._mouth_measurement([SimpleNamespace(x=0.5, y=0.5)] * 10) is None

    points = [SimpleNamespace(x=0.5, y=0.5) for _ in range(160)]
    points[MediaPipeMouthTracker._UPPER_LIP] = SimpleNamespace(x=0.5, y=None)
    assert MediaPipeMouthTracker._mouth_measurement(points) is None


def test_build_quality_gates_closes_faces_if_mouth_tracker_fails(monkeypatch):
    from mvdirector.config import ProjectConfig
    from mvdirector.qc import factory

    closed = False

    class FakeFaceAnalyzer:
        def close(self):
            nonlocal closed
            closed = True

    class BrokenMouthTracker:
        def __init__(self):
            raise RuntimeError("boom")

    monkeypatch.setattr(factory, "MediaPipeFaceAnalyzer", FakeFaceAnalyzer)
    monkeypatch.setattr(factory, "MediaPipeMouthTracker", BrokenMouthTracker)

    with pytest.raises(RuntimeError, match="boom"):
        factory.build_quality_gates(ProjectConfig(), llm=None)

    assert closed is True


def _unsynced_video_report(tmp_path, monkeypatch, *, blocking):
    import soundfile as sf

    from mvdirector import media
    from mvdirector.qc.video import VideoQC

    envelope, _ = _singing(n_frames=96, fps=24.0, seed=2)
    audio_path = tmp_path / "slice.wav"
    sf.write(str(audio_path), np.repeat(envelope, 1000).astype(np.float32), 24000)

    class FlatMouthTracker:
        def track(self, frames):
            return np.full(len(frames), 0.02)

    monkeypatch.setattr(
        media,
        "probe",
        lambda p: media.MediaInfo(width=1280, height=720, fps=24.0, duration=4.25, has_audio=True),
    )
    rng = np.random.default_rng(3)
    frames = [rng.integers(0, 255, (64, 64, 3), dtype=np.uint8) for _ in range(96)]
    monkeypatch.setattr(media, "extract_frames", lambda p, width: frames)

    config = ProjectConfig()
    config.qc.video.lipsync_blocking = blocking
    gate = VideoQC(config.qc, config.video, None, FlatMouthTracker(), None)
    shot = _shot(ShotType.LIPSYNC, Framing.MEDIUM_CLOSE)
    manifest = Manifest(title="Solo", duration=4.0, fps=24, shots=[shot])
    video_path = tmp_path / "clip.mp4"
    video_path.write_bytes(b"fake")
    return gate.evaluate(video_path, shot, manifest, audio_path, None)


def test_weak_lipsync_is_advisory_by_default(tmp_path, monkeypatch):
    report = _unsynced_video_report(tmp_path, monkeypatch, blocking=False)
    sync = next(c for c in report.checks if c.name == "lipsync")
    assert sync.status is CheckStatus.FAILED and not sync.hard and sync.score is None
    assert not report.has_hard_failure


def test_weak_lipsync_blocks_when_configured(tmp_path, monkeypatch):
    report = _unsynced_video_report(tmp_path, monkeypatch, blocking=True)
    sync = next(c for c in report.checks if c.name == "lipsync")
    assert sync.hard and report.has_hard_failure and not report.passed


def test_borderline_passing_lipsync_is_not_pulled_below_pass_score(tmp_path, monkeypatch):
    import wave

    from mvdirector import media
    from mvdirector.models import CheckStatus
    from mvdirector.qc.lipsync import SyncMeasurement
    from mvdirector.qc.video import VideoQC
    from mvdirector.qc.vision import VisionVerdict

    audio_path = tmp_path / "slice.wav"
    t = np.linspace(0.0, 4.0, 64000, endpoint=False)
    tone = (12000.0 * np.sin(2.0 * np.pi * 440.0 * t)).astype(np.int16)
    with wave.open(str(audio_path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(tone.tobytes())

    monkeypatch.setattr(
        media,
        "probe",
        lambda p: media.MediaInfo(width=896, height=512, fps=24.0, duration=4.0, has_audio=True),
    )
    frames = [np.full((64, 64, 3), 80 + i, dtype=np.uint8) for i in range(96)]
    monkeypatch.setattr(media, "extract_frames", lambda p, width: frames)

    class FixedScorer:
        def __init__(self, corr: float) -> None:
            self.corr = corr

        def score(self, *args, **kwargs):
            return SyncMeasurement(correlation=self.corr, lag_frames=-1, face_presence=1.0)

    class SteadyVision:
        disabled_reason = None

        def judge(self, *args, **kwargs):
            return VisionVerdict(
                spec_match=0.72,
                artifact_free=0.73,
                character_match=0.74,
                issues=[],
            )

    config = ProjectConfig()
    config.qc.video.lipsync_blocking = True
    shot = _shot(ShotType.LIPSYNC, Framing.MEDIUM_CLOSE)
    manifest = Manifest(title="Solo", duration=4.0, fps=24, shots=[shot])
    video_path = tmp_path / "clip.mp4"
    video_path.write_bytes(b"fake")

    borderline_gate = VideoQC(
        config.qc,
        config.video,
        None,
        None,
        SteadyVision(),
        lipsync_scorer=FixedScorer(0.41),
    )
    borderline = borderline_gate.evaluate(
        video_path, shot, manifest, audio_path, None, expected_size=(896, 512)
    )
    sync_low = next(c for c in borderline.checks if c.name == "lipsync")
    assert sync_low.status is CheckStatus.PASSED
    assert sync_low.score is not None and sync_low.score >= config.qc.pass_score
    assert borderline.passed

    strong_gate = VideoQC(
        config.qc,
        config.video,
        None,
        None,
        SteadyVision(),
        lipsync_scorer=FixedScorer(0.65),
    )
    strong = strong_gate.evaluate(
        video_path, shot, manifest, audio_path, None, expected_size=(896, 512)
    )
    sync_hi = next(c for c in strong.checks if c.name == "lipsync")
    assert sync_hi.score is not None and sync_hi.score > sync_low.score
    assert strong.score > borderline.score


def test_oversized_or_squashed_face_in_wide_shot_is_a_hard_failure(still_path):
    # Matches the 0.1172 BlazeFace measurement from a vertically squashed figure in a wide shot.
    squashed = [Face(0.1172, np.array([1.0, 0.0]))]
    report = _image_report(still_path, squashed, _shot(ShotType.PERFORMANCE, Framing.WIDE))
    assert not report.passed and report.has_hard_failure
    framing = next(c for c in report.checks if c.name == "face_framing")
    assert framing.status is CheckStatus.FAILED and framing.hard
    assert "allows at most 10%" in framing.detail


def test_unexpected_face_in_broll_cutaway_is_rejected(still_path):
    report = _image_report(still_path, [Face(0.1172, None)], _shot(ShotType.BROLL, Framing.WIDE))
    assert not report.passed and report.has_hard_failure
    no_people = next(c for c in report.checks if c.name == "no_people")
    assert no_people.status is CheckStatus.FAILED and no_people.hard
    assert "unexpected face detected in cutaway" in no_people.detail


def test_both_singing_still_requires_two_adequately_sized_faces(still_path):
    from mvdirector.models import Character
    from mvdirector.qc.image import ImageQC

    config = ProjectConfig()
    shot = _shot(ShotType.LIPSYNC, Framing.MEDIUM_CLOSE)
    shot.character = "elias"
    shot.characters = ["elias", "odalie"]
    shot.singers = ["elias", "odalie"]
    manifest = Manifest(
        title="Duet",
        duration=4.0,
        fps=24,
        characters={
            "elias": Character(name="Elias", appearance="Elias."),
            "odalie": Character(name="Odalie", appearance="Odalie."),
        },
        shots=[shot],
    )

    # Missing second singer -> hard face_present failure
    one_face_gate = ImageQC(config.qc, config.video, FakeFaces([Face(0.28, None)]), None)
    r_one = one_face_gate.evaluate(still_path, shot, manifest, reference=None)
    assert not r_one.passed and r_one.has_hard_failure
    assert _status(r_one, "face_present") is CheckStatus.FAILED
    assert "requires both singers visible" in r_one.issues[0]

    # Second singer tiny in far background -> hard face_framing failure
    tiny_second_gate = ImageQC(
        config.qc, config.video, FakeFaces([Face(0.26, None), Face(0.06, None)]), None
    )
    r_tiny = tiny_second_gate.evaluate(still_path, shot, manifest, reference=None)
    assert not r_tiny.passed and r_tiny.has_hard_failure
    assert _status(r_tiny, "face_framing") is CheckStatus.FAILED
    assert "co-singer face is 6%" in r_tiny.issues[0]

    # Balanced medium close-up two-shot -> passes
    two_faces_gate = ImageQC(
        config.qc, config.video, FakeFaces([Face(0.24, None), Face(0.18, None)]), None
    )
    r_ok = two_faces_gate.evaluate(still_path, shot, manifest, reference=None)
    assert r_ok.passed
    assert _status(r_ok, "face_present") is CheckStatus.PASSED
    assert _status(r_ok, "face_framing") is CheckStatus.PASSED


def test_both_singing_video_requires_both_mouths_tracked_and_synced(tmp_path, monkeypatch):
    import soundfile as sf

    from mvdirector import media
    from mvdirector.models import Character
    from mvdirector.qc.video import VideoQC

    envelope, synced_mouth = _singing(n_frames=96, fps=24.0, seed=5)
    audio_path = tmp_path / "slice.wav"
    sf.write(str(audio_path), np.repeat(envelope, 1000).astype(np.float32), 24000)

    monkeypatch.setattr(
        media,
        "probe",
        lambda p: media.MediaInfo(width=1280, height=720, fps=24.0, duration=4.25, has_audio=True),
    )
    fake_frames = [
        np.full((64, 64, 3), 100 + (i % 6), dtype=np.uint8) for i in range(len(synced_mouth))
    ]
    monkeypatch.setattr(media, "extract_frames", lambda p, width: fake_frames)

    shot = _shot(ShotType.LIPSYNC, Framing.MEDIUM_CLOSE)
    shot.character = "elias"
    shot.characters = ["elias", "odalie"]
    shot.singers = ["elias", "odalie"]
    manifest = Manifest(
        title="Duet",
        duration=4.0,
        fps=24,
        characters={
            "elias": Character(name="Elias", appearance="Elias."),
            "odalie": Character(name="Odalie", appearance="Odalie."),
        },
        shots=[shot],
    )
    video_path = tmp_path / "clip.mp4"
    video_path.write_bytes(b"fake")
    config = ProjectConfig()
    config.qc.video.lipsync_blocking = True

    # One mouth synced + one mouth frozen: fails on 'both' even though a solo two-shot would pass!
    class OneFrozenTracker:
        def track(self, frames):
            return synced_mouth

        def track_all(self, frames, max_faces=2):
            return [synced_mouth, np.full(len(frames), 0.02)]

    r_frozen = VideoQC(config.qc, config.video, None, OneFrozenTracker(), None).evaluate(
        video_path, shot, manifest, audio_path, None
    )
    assert not r_frozen.passed and r_frozen.has_hard_failure
    assert _status(r_frozen, "lipsync") is CheckStatus.FAILED
    assert "both singers must lip-sync together" in r_frozen.issues[0]

    # Only 1 face tracked: fails face_presence
    class SingleFoundTracker:
        def track(self, frames):
            return synced_mouth

        def track_all(self, frames, max_faces=2):
            return [synced_mouth]

    r_missing = VideoQC(config.qc, config.video, None, SingleFoundTracker(), None).evaluate(
        video_path, shot, manifest, audio_path, None
    )
    assert not r_missing.passed and r_missing.has_hard_failure
    assert _status(r_missing, "face_presence") is CheckStatus.FAILED
    assert "tracked 1 of 2 singing faces" in r_missing.issues[0]

    # Both mouths synced: passes!
    class BothSyncedTracker:
        def track(self, frames):
            return synced_mouth

        def track_all(self, frames, max_faces=2):
            return [synced_mouth, synced_mouth * 0.95]

    r_both = VideoQC(config.qc, config.video, None, BothSyncedTracker(), None).evaluate(
        video_path, shot, manifest, audio_path, None
    )
    assert r_both.passed
    assert _status(r_both, "face_presence") is CheckStatus.PASSED
    assert _status(r_both, "lipsync") is CheckStatus.PASSED


def test_qc_and_retry_config_boundaries_and_partial_framing_thresholds(still_path):
    from pydantic import ValidationError

    from mvdirector.qc.image import ImageQC

    with pytest.raises(ValidationError):
        ProjectConfig.model_validate({"qc": {"image": {"vision_spec_hard_min": 1.5}}})
    with pytest.raises(ValidationError):
        ProjectConfig.model_validate({"qc": {"video": {"vision_spec_hard_min": -0.1}}})
    with pytest.raises(ValidationError):
        ProjectConfig.model_validate({"retry": {"strategies": []}})

    cfg = ProjectConfig()
    cfg.qc.image.min_face_fraction = {}
    cfg.qc.image.max_face_fraction = {}
    shot = _shot(ShotType.PERFORMANCE, Framing.MEDIUM)
    report = ImageQC(cfg.qc, cfg.video, FakeFaces([Face(0.20, None)]), None).evaluate(
        still_path, shot, Manifest(title="T", duration=4.0, fps=24, shots=[shot]), None
    )
    assert report.passed


def test_temporal_checks_allow_smooth_steady_motion_while_catching_flicker():
    from mvdirector.qc.video import VideoQC

    cfg = ProjectConfig()
    gate = VideoQC(cfg.qc, cfg.video, None, None, None)

    # Smooth steady brightness/camera ramp: high median (~0.055) with near-zero spike excess.
    smooth_frames = [np.full((64, 64, 3), int(10 + i * 14), dtype=np.uint8) for i in range(16)]
    smooth = gate._temporal_checks(smooth_frames)[0]
    assert smooth.status is CheckStatus.PASSED
    assert smooth.score is not None and smooth.score >= 0.95
    assert smooth.metrics["median_motion"] > 0.04
    assert smooth.metrics["flicker_excess"] < 0.01

    # Calm clip with intermittent erratic spike (median low, p95 spike high): penalized.
    jitter_frames = [np.full((64, 64, 3), 100 + (i % 2) * 2, dtype=np.uint8) for i in range(16)]
    jitter_frames[8] = np.full((64, 64, 3), 124, dtype=np.uint8)
    jitter = gate._temporal_checks(jitter_frames)[0]
    assert jitter.status is CheckStatus.PASSED
    assert jitter.score is not None and jitter.score < 0.70
    assert jitter.metrics["flicker_excess"] > cfg.qc.video.flicker_good

    # Heavy strobe/flicker exceeding flicker_bad still fails hard with score 0.0.
    strobe_frames = [
        np.full((64, 64, 3), 60 if i % 2 == 0 else 100, dtype=np.uint8) for i in range(16)
    ]
    strobe = gate._temporal_checks(strobe_frames)[0]
    assert strobe.status is CheckStatus.FAILED
    assert strobe.score == 0.0
    assert "flicker" in strobe.detail
