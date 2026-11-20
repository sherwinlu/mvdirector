# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Tests for ByteDance LatentSync StableSyncNet + MediaPipe lip-sync scoring."""

from __future__ import annotations

import builtins
import stat
from pathlib import Path
from types import SimpleNamespace

import httpx
import numpy as np
import pytest
import soundfile as sf

from mvdirector import media
from mvdirector.config import ProjectConfig
from mvdirector.errors import ConfigError, DependencyMissing
from mvdirector.models import AssetState, AudioSlice, CheckStatus, Framing, Manifest, Shot, ShotType
from mvdirector.qc import factory, syncnet
from mvdirector.qc.faces import Face, MediaPipeFaceAnalyzer
from mvdirector.qc.lipsync import SyncMeasurement
from mvdirector.qc.syncnet import (
    SYNCNET_FPS,
    SYNCNET_LOWER_HALF_HEIGHT,
    SYNCNET_MEL_WINDOW,
    SYNCNET_NUM_FRAMES,
    SYNCNET_RESOLUTION,
    StableSyncNetScorer,
    crop_lower_face,
    ensure_syncnet_model,
    extract_face_tracks,
    mel_spectrogram,
    mel_window_for_frame,
    resample_frames_to_25fps,
    score_embedding_sequences,
)
from mvdirector.qc.video import VideoQC


def _sine_wave(seconds: float = 2.0, sample_rate: int = 16000) -> np.ndarray:
    t = np.arange(int(seconds * sample_rate), dtype=np.float32) / sample_rate
    return (0.4 * np.sin(2.0 * np.pi * 440.0 * t)).astype(np.float32)


def test_mel_spectrogram_shape_bounds_and_resampling():
    wav_16k = _sine_wave(seconds=2.0, sample_rate=16000)
    wav_48k = _sine_wave(seconds=2.0, sample_rate=48000)
    mel_16 = mel_spectrogram(wav_16k, 16000)
    mel_48 = mel_spectrogram(wav_48k, 48000)

    assert mel_16.shape[0] == 80 and mel_16.dtype == np.float32
    assert mel_16.min() >= -4.0 and mel_16.max() <= 4.0
    # 16 kHz / hop 200 = 80 mel frames/sec -> ~161 frames for 2s.
    assert abs(mel_16.shape[1] - 161) <= 2
    assert abs(mel_16.shape[1] - mel_48.shape[1]) <= 2

    window = mel_window_for_frame(mel_16, 0)
    assert window is not None and window.shape == (80, SYNCNET_MEL_WINDOW)
    assert mel_window_for_frame(mel_16, 10_000) is None
    assert mel_spectrogram(np.zeros(0, dtype=np.float32), 16000).shape == (80, 0)
    assert mel_spectrogram(wav_16k, 0).shape == (80, 0)


def test_resample_frames_to_25fps():
    frames = [np.full((16, 16, 3), i, dtype=np.uint8) for i in range(24)]
    resampled = resample_frames_to_25fps(frames, 24.0)
    assert len(resampled) == 25
    assert resample_frames_to_25fps(frames, 25.0) == frames
    assert resample_frames_to_25fps([], 24.0) == []


def test_crop_lower_face_shape_and_normalization():
    frame = np.zeros((200, 300, 3), dtype=np.uint8)
    frame[100:, :, :] = 255  # bright lower half
    crop = crop_lower_face(frame, (100, 50, 200, 150))
    assert crop.shape == (3, SYNCNET_LOWER_HALF_HEIGHT, SYNCNET_RESOLUTION)
    assert crop.dtype == np.float32
    assert crop.min() >= -1.0 and crop.max() <= 1.0
    # Edge-clipped bbox near corner does not crash.
    corner = crop_lower_face(frame, (0, 0, 20, 20))
    assert corner.shape == (3, SYNCNET_LOWER_HALF_HEIGHT, SYNCNET_RESOLUTION)


def test_extract_face_tracks_interpolates_and_separates_two_faces():
    class TwoFaceDetector:
        def __init__(self):
            self.calls = 0

        def faces(self, frame):
            idx = self.calls
            self.calls += 1
            if idx == 2:
                return [Face(height_fraction=0.3, bbox=(150, 20, 190, 80))]
            return [
                Face(height_fraction=0.3, bbox=(20, 20, 60, 80)),
                Face(height_fraction=0.3, bbox=(150, 20, 190, 80)),
            ]

    frames = [np.zeros((100, 200, 3), dtype=np.uint8) for _ in range(5)]
    tracks = extract_face_tracks(frames, TwoFaceDetector(), max_faces=2)
    assert len(tracks) == 2
    presences = sorted(p for p, _ in tracks)
    assert presences == [pytest.approx(0.8), pytest.approx(1.0)]
    assert all(len(boxes) == 5 for _, boxes in tracks)


def test_extract_face_tracks_solo_keeps_largest_face_over_leftmost_background_face():
    class ForegroundAndBackgroundDetector:
        def faces(self, frame):
            # Largest face first (on the right), smaller background face on the left.
            return [
                Face(height_fraction=0.6, bbox=(120, 20, 180, 80)),
                Face(height_fraction=0.15, bbox=(10, 40, 25, 55)),
            ]

    frames = [np.zeros((100, 200, 3), dtype=np.uint8) for _ in range(3)]
    tracks = extract_face_tracks(frames, ForegroundAndBackgroundDetector(), max_faces=1)
    assert len(tracks) == 1
    presence, boxes = tracks[0]
    assert presence == 1.0
    assert boxes[0] == (120, 20, 180, 80)


def test_score_embedding_sequences_detects_alignment_and_lag():
    rng = np.random.default_rng(7)
    base = rng.standard_normal((40, 32)).astype(np.float32)
    base /= np.linalg.norm(base, axis=1, keepdims=True)

    aligned = score_embedding_sequences(base, base, max_lag_frames=6, face_presence=1.0)
    assert aligned.correlation == pytest.approx(1.0, abs=1e-5)
    assert aligned.lag_frames == 0
    assert aligned.eval_fps == SYNCNET_FPS
    assert aligned.confidence is not None and aligned.confidence > 0.5

    # Visual delayed by +3 frames relative to audio.
    delayed_v = np.vstack([rng.standard_normal((3, 32)).astype(np.float32), base[:-3]])
    delayed_v /= np.linalg.norm(delayed_v, axis=1, keepdims=True)
    lagged = score_embedding_sequences(delayed_v, base, max_lag_frames=6, face_presence=0.95)
    assert lagged.lag_frames == 3
    assert lagged.correlation > 0.99
    assert lagged.face_presence == pytest.approx(0.95)

    empty = score_embedding_sequences(
        np.zeros((0, 32), dtype=np.float32), base, max_lag_frames=6, face_presence=0.0
    )
    assert empty.correlation == 0.0 and empty.lag_frames == 0


def test_stable_syncnet_scorer_end_to_end_shapes_and_lifecycle():
    captured_shapes: list[tuple[tuple[int, ...], tuple[int, ...]]] = []

    def fake_encoder(visual: np.ndarray, audio: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        captured_shapes.append((visual.shape, audio.shape))
        rng = np.random.default_rng(42)
        vecs = rng.standard_normal((max(len(visual), len(audio)), 16)).astype(np.float32)
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs[: len(visual)], vecs[: len(audio)]

    class BorrowedFaceDetector:
        closed = False

        def faces(self, frame):
            return [Face(height_fraction=0.4, bbox=(30, 20, 90, 80))]

        def close(self):
            self.closed = True

    borrowed = BorrowedFaceDetector()
    scorer = StableSyncNetScorer(face_analyzer=borrowed, _encoder=fake_encoder)
    frames = [np.full((120, 120, 3), 128, dtype=np.uint8) for _ in range(30)]
    wav = _sine_wave(seconds=2.0, sample_rate=16000)

    result = scorer.score(frames, 25.0, wav, 16000, max_lag_seconds=0.2, max_faces=1)
    assert result.correlation == pytest.approx(1.0, abs=1e-5)
    assert result.lag_frames == 0
    assert result.face_presence == 1.0
    assert result.eval_fps == SYNCNET_FPS
    assert captured_shapes == [
        (
            (30 - SYNCNET_NUM_FRAMES + 1, 48, SYNCNET_LOWER_HALF_HEIGHT, SYNCNET_RESOLUTION),
            (30 - SYNCNET_NUM_FRAMES + 1, 1, 80, SYNCNET_MEL_WINDOW),
        )
    ]

    # Short clip (< 16 frames) degrades cleanly without invoking encoder.
    short = scorer.score(frames[:10], 25.0, wav, 16000, max_lag_seconds=0.2)
    assert short.correlation == 0.0 and short.face_presence == 1.0

    # Borrowed face detector must NOT be closed by scorer.close().
    scorer.close()
    assert borrowed.closed is False
    with pytest.raises(RuntimeError, match="closed"):
        scorer.score(frames, 25.0, wav, 16000, max_lag_seconds=0.2)


def test_stable_syncnet_scorer_missing_torch_closes_owned_detector(monkeypatch):
    closed = False

    class OwnedDetector:
        def close(self):
            nonlocal closed
            closed = True

    monkeypatch.setattr(syncnet, "MediaPipeFaceAnalyzer", OwnedDetector)
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "torch":
            raise ImportError("no torch")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(DependencyMissing, match="PyTorch is not installed"):
        StableSyncNetScorer()
    assert closed is True


def test_ensure_syncnet_model_downloads_caches_and_validates_paths(tmp_path, monkeypatch):
    cache_dir = tmp_path / "syncnet_cache"
    monkeypatch.setenv("MVD_SYNCNET_CACHE_DIR", str(cache_dir))

    calls = 0

    def handler(request):
        nonlocal calls
        calls += 1
        return httpx.Response(200, content=b"fake-syncnet-weights", request=request)

    real_client = httpx.Client
    monkeypatch.setattr(
        syncnet.httpx,
        "Client",
        lambda **kwargs: real_client(transport=httpx.MockTransport(handler), **kwargs),
    )

    first = ensure_syncnet_model()
    second = ensure_syncnet_model()
    assert first == cache_dir / "stable_syncnet.pt"
    assert second == first and calls == 1
    assert stat.S_IMODE(first.stat().st_mode) == 0o644

    with pytest.raises(DependencyMissing, match="null byte"):
        ensure_syncnet_model(model_path=tmp_path / "bad\x00.pt")
    with pytest.raises(DependencyMissing, match="not found"):
        ensure_syncnet_model(model_path=tmp_path / "missing.pt")

    from mvdirector.qc.common import download_cached_model

    with pytest.raises(DependencyMissing, match="Invalid"):
        download_cached_model("../escape.pt", "https://example.com/a.pt", cache_dir)


def test_corrupt_cached_syncnet_checkpoint_redownloads_once(tmp_path, monkeypatch):
    import zipfile

    cache_dir = tmp_path / "syncnet_cache"
    cache_dir.mkdir()
    monkeypatch.setenv("MVD_SYNCNET_CACHE_DIR", str(cache_dir))
    cached = cache_dir / "stable_syncnet.pt"
    cached.write_bytes(b"corrupt-zip")
    downloads = 0

    def handler(request):
        nonlocal downloads
        downloads += 1
        return httpx.Response(200, content=b"valid-checkpoint", request=request)

    real_client = httpx.Client
    monkeypatch.setattr(
        syncnet.httpx,
        "Client",
        lambda **kwargs: real_client(transport=httpx.MockTransport(handler), **kwargs),
    )

    class FakeModel:
        def load_state_dict(self, state_dict, strict=True):
            pass

        def to(self, device):
            return self

        def eval(self):
            return self

    class FakeTorch:
        @staticmethod
        def load(path, map_location="cpu", weights_only=True):
            if Path(path).read_bytes() == b"corrupt-zip":
                raise zipfile.BadZipFile("truncated")
            return {"state_dict": {}}

    monkeypatch.setattr(syncnet, "build_stable_syncnet", lambda torch_mod: FakeModel())
    loaded = syncnet._load_stable_syncnet_checkpoint(FakeTorch, None, "cpu")
    assert isinstance(loaded, FakeModel)
    assert downloads == 1
    assert cached.read_bytes() == b"valid-checkpoint"

    # Explicit model_path is never deleted or re-downloaded.
    explicit = tmp_path / "explicit.pt"
    explicit.write_bytes(b"corrupt-zip")
    with pytest.raises(DependencyMissing, match="could not load"):
        syncnet._load_stable_syncnet_checkpoint(FakeTorch, explicit, "cpu")
    assert explicit.read_bytes() == b"corrupt-zip" and downloads == 1


def test_mediapipe_face_analyzer_populates_clipped_bbox(tmp_path, monkeypatch):
    from mvdirector.qc import faces as faces_mod

    class FakeDetector:
        @classmethod
        def create_from_options(cls, options):
            return cls()

        def detect(self, mp_image):
            return SimpleNamespace(
                detections=[
                    SimpleNamespace(
                        bounding_box=SimpleNamespace(
                            origin_x=-10, origin_y=20, width=110, height=120
                        )
                    )
                ]
            )

        def close(self):
            pass

    fake_mp = SimpleNamespace(
        Image=lambda image_format, data: SimpleNamespace(data=data),
        ImageFormat=SimpleNamespace(SRGB="SRGB"),
    )
    fake_vision = SimpleNamespace(
        FaceDetector=FakeDetector,
        FaceDetectorOptions=lambda **kw: SimpleNamespace(**kw),
        RunningMode=SimpleNamespace(IMAGE="IMAGE", VIDEO="VIDEO"),
    )
    monkeypatch.setattr(
        faces_mod,
        "_import_mediapipe_tasks",
        lambda feature: (fake_mp, lambda **kw: SimpleNamespace(**kw), fake_vision),
    )
    model_file = tmp_path / "blaze.tflite"
    model_file.write_bytes(b"fake")

    with MediaPipeFaceAnalyzer(model_path=model_file) as analyzer:
        detected = analyzer.faces(np.zeros((240, 320, 3), dtype=np.uint8))
    assert len(detected) == 1
    assert detected[0].bbox == (0, 20, 100, 140)
    assert detected[0].height_fraction == pytest.approx(0.5)


def test_factory_wires_stable_syncnet_and_reuses_mediapipe(monkeypatch):
    constructed: dict[str, object] = {}

    class FakeMPAnalyzer(MediaPipeFaceAnalyzer):
        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

    class FakeScorer:
        def __init__(self, *, model_path=None, device="auto", face_analyzer=None):
            constructed["model_path"] = model_path
            constructed["device"] = device
            constructed["face_analyzer"] = face_analyzer
            self.closed = False

        def close(self):
            self.closed = True

    monkeypatch.setattr(factory, "MediaPipeFaceAnalyzer", FakeMPAnalyzer)
    monkeypatch.setattr(factory, "StableSyncNetScorer", FakeScorer)
    monkeypatch.setattr(
        factory,
        "MediaPipeMouthTracker",
        lambda: pytest.fail("MediaPipeMouthTracker must not be built for stable_syncnet"),
    )

    cfg = ProjectConfig()
    cfg.qc.lipsync_backend = "stable_syncnet"
    cfg.qc.syncnet_device = "cpu"
    gates = factory.build_quality_gates(cfg, llm=None)

    assert isinstance(gates.face_analyzer, FakeMPAnalyzer)
    assert constructed["face_analyzer"] is gates.face_analyzer
    assert constructed["device"] == "cpu"
    assert gates.mouth_tracker is None
    assert gates.lipsync_scorer is not None

    active, skipped = gates.check_summary()
    assert "lipsync" in active and "face_presence" in active
    assert not any("lipsync" in name for name, _ in skipped)

    gates.close()
    assert gates.lipsync_scorer.closed is True  # type: ignore[attr-defined]
    assert gates.face_analyzer.closed is True


def test_video_qc_uses_lipsync_scorer_and_records_confidence(tmp_path, monkeypatch):
    wav = _sine_wave(seconds=2.0, sample_rate=16000)
    audio_path = tmp_path / "slice.wav"
    sf.write(str(audio_path), wav, 16000)

    class StubScorer:
        def score(self, frames, fps, samples, sample_rate, max_lag_seconds, max_faces=1):
            return SyncMeasurement(
                correlation=0.78,
                lag_frames=2,
                face_presence=0.98,
                eval_fps=25.0,
                confidence=0.31,
            )

    monkeypatch.setattr(
        media,
        "probe",
        lambda p: media.MediaInfo(width=1280, height=720, fps=24.0, duration=2.25, has_audio=True),
    )
    rng = np.random.default_rng(1)
    frames = [rng.integers(0, 255, (64, 64, 3), dtype=np.uint8) for _ in range(48)]
    monkeypatch.setattr(media, "extract_frames", lambda p, width: frames)

    cfg = ProjectConfig()
    gate = VideoQC(cfg.qc, cfg.video, None, None, None, lipsync_scorer=StubScorer())
    shot = Shot(
        id="s001",
        index=0,
        start=0.0,
        end=2.0,
        type=ShotType.LIPSYNC,
        framing=Framing.CLOSE_UP,
        audio_slice=AudioSlice(path="a.wav", start=0.0, end=2.25, head_handle=0.0),
        num_frames=57,
        still=AssetState(attempts_allowed=4),
        video=AssetState(attempts_allowed=4),
    )
    manifest = Manifest(title="Solo", duration=2.0, fps=24, shots=[shot])
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"fake")

    report = gate.evaluate(clip, shot, manifest, audio_path, None)
    sync = next(c for c in report.checks if c.name == "lipsync")
    assert sync.status is CheckStatus.PASSED
    assert sync.metrics["correlation"] == pytest.approx(0.78)
    assert sync.metrics["lag_seconds"] == pytest.approx(0.08)  # 2 / 25.0
    assert sync.metrics["sync_confidence"] == pytest.approx(0.31)


def test_syncnet_model_path_env_expansion_respects_allowlist(monkeypatch):
    monkeypatch.setenv("MVD_SYNCNET_PATH", "/models/stable_syncnet.pt")
    cfg = ProjectConfig.model_validate({"qc": {"syncnet_model_path": "${MVD_SYNCNET_PATH}"}})
    assert cfg.qc.syncnet_model_path == "/models/stable_syncnet.pt"

    with pytest.raises(ConfigError, match="not allowed"):
        ProjectConfig.model_validate({"qc": {"syncnet_model_path": "${SECRET_PATH}"}})


def test_build_stable_syncnet_forward_and_checkpoint_roundtrip(tmp_path):
    torch = pytest.importorskip("torch")
    from mvdirector.qc.syncnet import _load_stable_syncnet_checkpoint, build_stable_syncnet

    model = build_stable_syncnet(torch)
    with torch.no_grad():
        visual = torch.randn(1, 48, SYNCNET_LOWER_HALF_HEIGHT, SYNCNET_RESOLUTION)
        audio = torch.randn(1, 1, 80, SYNCNET_MEL_WINDOW)
        v_emb, a_emb = model(visual, audio)

    assert v_emb.shape == (1, 2048)
    assert a_emb.shape == (1, 2048)
    assert float(torch.linalg.norm(v_emb).item()) == pytest.approx(1.0, abs=1e-4)
    assert float(torch.linalg.norm(a_emb).item()) == pytest.approx(1.0, abs=1e-4)

    ckpt_path = tmp_path / "stable_syncnet.pt"
    torch.save({"state_dict": model.state_dict()}, ckpt_path)
    reloaded = _load_stable_syncnet_checkpoint(torch, ckpt_path, "cpu")
    with torch.no_grad():
        v2, a2 = reloaded(visual, audio)
    assert torch.allclose(v_emb, v2, atol=1e-5)
    assert torch.allclose(a_emb, a2, atol=1e-5)
