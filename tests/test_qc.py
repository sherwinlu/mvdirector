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
    assert "mediapipe unavailable" in skipped_map["face_present, face_framing"]
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
