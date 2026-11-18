# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""The transcription path, exercised with a stand-in for faster-whisper."""

import sys
import types

import numpy as np

from mvdirector.audio import lyrics


def _install_fake_whisper(monkeypatch, behaviour):
    module = types.ModuleType("faster_whisper")
    module.WhisperModel = behaviour
    monkeypatch.setitem(sys.modules, "faster_whisper", module)


def test_samples_are_passed_in_so_whisper_never_decodes_audio(project, monkeypatch):
    seen = {}

    class Model:
        def __init__(self, *args, **kwargs):
            pass

        def transcribe(self, audio, **kwargs):
            seen["audio"] = audio
            word = types.SimpleNamespace(word="first", start=4.0, end=4.4)
            return [types.SimpleNamespace(words=[word])], None

    _install_fake_whisper(monkeypatch, Model)
    words = lyrics.load_or_transcribe_words(project)
    assert isinstance(seen["audio"], np.ndarray) and seen["audio"].dtype == np.float32
    assert [(w.word, w.start) for w in words] == [("first", 4.0)]


def test_transcription_failure_falls_back_to_estimated_timing(project, monkeypatch, caplog):
    class Broken:
        def __init__(self, *args, **kwargs):
            pass

        def transcribe(self, *args, **kwargs):
            raise TypeError("open() got an unexpected keyword argument 'metadata_errors'")

    _install_fake_whisper(monkeypatch, Broken)
    assert lyrics.load_or_transcribe_words(project) == []
    assert "Transcription failed" in caplog.text
