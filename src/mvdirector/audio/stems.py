# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Stem separation for audio-only ingestion (isolating vocals and optional drums)."""

from __future__ import annotations

import importlib.util
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from mvdirector.errors import ConfigError, MediaToolError

log = logging.getLogger(__name__)

SUPPORTED_AUDIO_SUFFIXES = frozenset({".mp3", ".wav"})
_SAFE_MODEL_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_VALID_DEVICES = frozenset({"auto", "cpu", "cuda", "mps"})


@dataclass(frozen=True)
class StemResult:
    """Paths written by a stem separator."""

    vocals: Path
    drums: Path | None = None


class StemSeparator(Protocol):
    """Protocol for separating a vocal stem (and optional drum stem) from a full mix."""

    def separate(
        self,
        mix_path: Path,
        vocals_out: Path,
        *,
        drums_out: Path | None = None,
    ) -> StemResult: ...


def atomic_copy(src: Path, dst: Path) -> Path:
    """Copy ``src`` to ``dst`` atomically via a sibling temp file."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=".mvd_copy_", suffix=dst.suffix or ".wav", dir=dst.parent
    )
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        shutil.copyfile(src, tmp_path)
        os.replace(tmp_path, dst)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise
    return dst


class DemucsSeparator:
    """Local stem separation backed by MIT-licensed Demucs (`python -m demucs`)."""

    def __init__(self, model: str = "htdemucs", device: str = "auto") -> None:
        if not _SAFE_MODEL_NAME.match(model):
            raise ConfigError(
                f"Invalid Demucs model name {model!r}: use letters, digits, '.', '_' or '-'."
            )
        if device not in _VALID_DEVICES:
            raise ConfigError(
                f"Invalid stem_device {device!r}: expected one of {sorted(_VALID_DEVICES)}."
            )
        self.model = model
        self.device = device

    def _build_command(self, mix_path: Path, output_dir: Path, *, include_drums: bool) -> list[str]:
        cmd = [
            sys.executable,
            "-m",
            "demucs",
            "-n",
            self.model,
            "-o",
            str(output_dir.resolve()),
        ]
        if not include_drums:
            cmd.extend(["--two-stems", "vocals"])
        if self.device != "auto":
            cmd.extend(["-d", self.device])
        cmd.append(str(mix_path.resolve()))
        return cmd

    def separate(
        self,
        mix_path: Path,
        vocals_out: Path,
        *,
        drums_out: Path | None = None,
    ) -> StemResult:
        if not mix_path.is_file():
            raise MediaToolError(f"Mix audio file not found: {mix_path}")
        if importlib.util.find_spec("demucs") is None:
            raise ConfigError(
                "Stem separation requires 'demucs'. Install it with: pip install -e '.[stems]'"
            )

        include_drums = drums_out is not None
        with tempfile.TemporaryDirectory(prefix="mvd_stems_") as tmp:
            out_root = Path(tmp)
            cmd = self._build_command(mix_path, out_root, include_drums=include_drums)
            log.info(
                "Separating stems with Demucs (%s, %s)...",
                self.model,
                "4-stem" if include_drums else "vocals",
            )
            log.debug("exec: %s", " ".join(cmd))
            result = subprocess.run(cmd, capture_output=True, text=True, check=False)
            if result.returncode != 0:
                tail = "\n".join(result.stderr.strip().splitlines()[-10:])
                raise MediaToolError(f"demucs failed (exit {result.returncode}):\n{tail}")

            stem_dir = out_root / self.model / mix_path.stem
            vocals_src = stem_dir / "vocals.wav"
            if not vocals_src.is_file():
                raise MediaToolError(f"Demucs did not produce expected vocal stem at {vocals_src}")
            vocals_written = atomic_copy(vocals_src, vocals_out)

            drums_written: Path | None = None
            if drums_out is not None:
                drums_src = stem_dir / "drums.wav"
                if not drums_src.is_file():
                    raise MediaToolError(
                        f"Demucs did not produce expected drum stem at {drums_src}"
                    )
                drums_written = atomic_copy(drums_src, drums_out)

        return StemResult(vocals=vocals_written, drums=drums_written)
