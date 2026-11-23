# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Multi-stem pack ingestion, classification, mixing, and arrangement analysis.

Supports Suno stem separation archives (.zip from Auto Split, Split from Mix, or Premier
Advanced Split), folders of stems, or any number of loose audio files. Extracts safely into
``inputs/stems/``, classifies each stem into a canonical musical role, routes the cleanest
stems to lip-sync (lead vocals) and beat tracking (drums/percussion), synthesizes a full mix
when no master mix was provided, and builds a compact arrangement summary for planning.
"""

from __future__ import annotations

import logging
import math
import os
import re
import stat
import tempfile
import zipfile
import zlib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import numpy as np
import soundfile as sf

from mvdirector import media
from mvdirector.audio.stems import atomic_copy
from mvdirector.errors import ConfigError, MediaToolError
from mvdirector.models import (
    StemArrangement,
    StemRole,
    StemTrackInfo,
    TimeSpan,
    ascii_fold,
)

log = logging.getLogger(__name__)

STEM_AUDIO_SUFFIXES = frozenset({".wav", ".mp3", ".flac"})
ARCHIVE_SUFFIXES = frozenset({".zip"})

# Conservative archive extraction limits to prevent zip bombs while allowing large studio WAVs.
MAX_ZIP_ENTRIES = 512
MAX_STEM_FILES = 64
MAX_STEM_FILE_BYTES = 512 * 1024 * 1024  # 512 MiB per extracted stem
MAX_TOTAL_STEM_BYTES = 2 * 1024 * 1024 * 1024  # 2 GiB total across a pack
_COPY_CHUNK_BYTES = 1024 * 1024

_LEADING_INDEX_RE = re.compile(r"^\s*\d+[\s._-]+")
_TOKEN_SEP_RE = re.compile(r"[^a-z0-9]+")

# Ordered phrase rules: more specific / negation phrases must precede generic instrument tokens.
# Each entry is (role, tuple_of_phrases) matched on space-delimited normalized lowercase words.
_ROLE_RULES: tuple[tuple[StemRole, tuple[str, ...]], ...] = (
    (
        "instrumental",
        (
            "no vocals",
            "no vocal",
            "without vocals",
            "without vocal",
            "minus vocals",
            "minus vocal",
            "instrumental",
            "karaoke",
            "backing track",
            "accompaniment",
            "rest of mix",
            "no lead",
        ),
    ),
    (
        "backing_vocals",
        (
            "backing vocals",
            "backing vocal",
            "background vocals",
            "background vocal",
            "bg vocals",
            "bg vocal",
            "bgv",
            "harmony vocals",
            "harmony vocal",
            "harmonies",
            "harmony",
            "choir",
            "chorus vocals",
            "chorus vocal",
            "gang vocals",
            "group vocals",
            "adlibs",
            "adlib",
            "ad libs",
            "ad lib",
            "vocal chops",
            "vocal chop",
            "bv",
        ),
    ),
    (
        "lead_vocals",
        (
            "lead vocals",
            "lead vocal",
            "main vocals",
            "main vocal",
            "solo vocals",
            "solo vocal",
            "lead voice",
            "lead vox",
            "main vox",
            "primary vocal",
        ),
    ),
    (
        "drums",
        (
            "bass drum",
            "kick drum",
            "snare drum",
            "drum kit",
            "drum machine",
            "drums",
            "drum",
            "kick",
            "snare",
            "hi hat",
            "hihat",
            "hat",
            "hats",
            "cymbal",
            "cymbals",
            "crash",
            "ride",
            "tom",
            "toms",
            "clap",
            "claps",
            "rimshot",
            "rim",
            "breakbeat",
        ),
    ),
    (
        "percussion",
        (
            "percussion",
            "perc",
            "shaker",
            "tambourine",
            "conga",
            "congas",
            "bongo",
            "bongos",
            "timpani",
            "maraca",
            "maracas",
            "cowbell",
            "triangle",
            "cabasa",
            "claves",
            "woodblock",
            "djembe",
            "cajon",
            "tabla",
            "snap",
            "snaps",
            "finger snap",
            "handclap",
        ),
    ),
    (
        "bass",
        (
            "synth bass",
            "sub bass",
            "808 bass",
            "bass guitar",
            "electric bass",
            "acoustic bass",
            "upright bass",
            "double bass",
            "contrabass",
            "slap bass",
            "reese bass",
            "bassline",
            "bass",
            "808",
            "sub",
        ),
    ),
    (
        "guitar",
        (
            "electric guitar",
            "acoustic guitar",
            "nylon guitar",
            "distorted guitar",
            "clean guitar",
            "lead guitar",
            "rhythm guitar",
            "slide guitar",
            "steel guitar",
            "guitar",
            "guitars",
            "ukulele",
            "banjo",
            "mandolin",
            "sitar",
            "lute",
        ),
    ),
    (
        "keys",
        (
            "grand piano",
            "electric piano",
            "upright piano",
            "keyboards",
            "keyboard",
            "keys",
            "piano",
            "rhodes",
            "wurlitzer",
            "organ",
            "hammond",
            "harpsichord",
            "clavinet",
            "accordion",
            "celesta",
            "harp",
            "marimba",
            "vibraphone",
            "xylophone",
            "glockenspiel",
            "kalimba",
        ),
    ),
    (
        "strings",
        (
            "string section",
            "strings",
            "string",
            "violin",
            "violins",
            "viola",
            "cello",
            "cellos",
            "fiddle",
            "orchestra",
            "orchestral",
            "pizzicato",
            "erhu",
        ),
    ),
    (
        "synth",
        (
            "lead synth",
            "synth lead",
            "synth pad",
            "synthesizer",
            "synths",
            "synth",
            "pad",
            "pads",
            "arpeggio",
            "arp",
            "pluck",
            "plucks",
            "bell",
            "bells",
            "chime",
            "chimes",
            "theremin",
            "modular",
            "supersaw",
            "keys synth",
        ),
    ),
    (
        "brass",
        (
            "french horn",
            "brass section",
            "brass",
            "trumpet",
            "trumpets",
            "trombone",
            "trombones",
            "horn",
            "horns",
            "tuba",
            "flugelhorn",
            "cornet",
            "bugle",
        ),
    ),
    (
        "woodwinds",
        (
            "woodwinds",
            "woodwind",
            "saxophone",
            "sax",
            "alto sax",
            "tenor sax",
            "flute",
            "piccolo",
            "clarinet",
            "oboe",
            "bassoon",
            "recorder",
            "harmonica",
            "bagpipe",
            "pan flute",
            "shakuhachi",
        ),
    ),
    (
        "fx",
        (
            "sound effects",
            "sound effect",
            "sfx",
            "fx",
            "riser",
            "risers",
            "sweep",
            "sweeps",
            "impact",
            "impacts",
            "atmosphere",
            "ambient",
            "ambience",
            "noise",
            "vinyl",
            "foley",
        ),
    ),
    (
        "vocals",
        (
            "vocals",
            "vocal",
            "voice",
            "voices",
            "vox",
            "singer",
            "singing",
            "rap",
            "spoken",
            "acapella",
            "a cappella",
        ),
    ),
)

HARMONIC_ROLES: frozenset[StemRole] = frozenset(
    {"bass", "guitar", "keys", "strings", "synth", "brass", "woodwinds", "other", "instrumental"}
)
RHYTHM_ROLES: frozenset[StemRole] = frozenset({"drums", "percussion"})
VOCAL_ROLES: frozenset[StemRole] = frozenset({"lead_vocals", "backing_vocals", "vocals"})


@dataclass(frozen=True)
class IngestedStem:
    """A normalized stem file residing inside ``inputs/stems/``."""

    id: str
    name: str
    role: StemRole
    path: Path


def clean_stem_label(filename: str) -> str:
    """Convert a raw stem filename like ``'0 Lead Vocals.wav'`` into ``'Lead Vocals'``."""
    base = Path(filename).stem
    stripped = _LEADING_INDEX_RE.sub("", base).strip()
    label = re.sub(r"[_]+", " ", stripped or base)
    label = re.sub(r"\s+", " ", label).strip(" -.")
    if not label:
        return "Stem"
    return label.title() if label.islower() else label


def _normalized_words(filename: str) -> str:
    base = ascii_fold(clean_stem_label(filename)).lower()
    words = _TOKEN_SEP_RE.sub(" ", base).strip()
    return f" {words} " if words else " "


def classify_stem_name(filename: str) -> StemRole:
    """Infer a canonical musical role from a Suno/DAW stem filename."""
    padded = _normalized_words(filename)
    for role, phrases in _ROLE_RULES:
        for phrase in phrases:
            if f" {phrase} " in padded:
                return role
    return "other"


def make_stem_id(label: str, existing: set[str]) -> str:
    """Build a unique SAFE_ID-compatible slug for a stem label (at most 64 characters)."""
    folded = ascii_fold(label).lower()
    slug = re.sub(r"[^a-z0-9]+", "_", folded).strip("_")[:64].rstrip("_")
    if not slug or not slug[0].isalnum():
        slug = "stem"
    candidate = slug
    suffix = 2
    while candidate in existing:
        tail = f"_{suffix}"
        candidate = f"{slug[: 64 - len(tail)].rstrip('_') or 'stem'}{tail}"
        suffix += 1
    existing.add(candidate)
    return candidate


def _is_ignored_entry(name: str) -> bool:
    parts = PurePosixPath(name.replace("\\", "/")).parts
    if not parts:
        return True
    return any(part == "__MACOSX" or part.startswith("._") or part == ".DS_Store" for part in parts)


def _validate_zip_member(info: zipfile.ZipInfo) -> str | None:
    """Validate a zip member and return its basename if it is a safe audio file, else None."""
    raw_name = info.filename
    if "\x00" in raw_name:
        raise ConfigError(f"Refusing unsafe path inside zip archive: {raw_name!r}")
    if info.is_dir() or raw_name.endswith("/") or _is_ignored_entry(raw_name):
        return None
    # Reject symlinks embedded inside zip archives.
    mode = info.external_attr >> 16
    if stat.S_ISLNK(mode):
        raise ConfigError(f"Refusing symlink entry inside zip archive: {raw_name!r}")
    normalized = raw_name.replace("\\", "/")
    posix = PurePosixPath(normalized)
    if posix.is_absolute() or ".." in posix.parts or ":" in (posix.parts[0] if posix.parts else ""):
        raise ConfigError(f"Refusing unsafe path inside zip archive: {raw_name!r}")
    basename = posix.name
    if not basename or Path(basename).suffix.lower() not in STEM_AUDIO_SUFFIXES:
        return None
    if info.file_size > MAX_STEM_FILE_BYTES:
        raise ConfigError(
            f"Zip entry {basename!r} exceeds the {MAX_STEM_FILE_BYTES // (1024 * 1024)} MiB limit."
        )
    return basename


def _write_bounded_stream(
    src_stream, dst_path: Path, *, max_file_bytes: int, remaining_total_bytes: int
) -> int:
    """Copy bytes from ``src_stream`` to ``dst_path`` enforcing strict size limits."""
    written = 0
    limit = min(max_file_bytes, remaining_total_bytes)
    with dst_path.open("wb") as out:
        while True:
            chunk = src_stream.read(_COPY_CHUNK_BYTES)
            if not chunk:
                break
            written += len(chunk)
            if written > limit:
                raise ConfigError(
                    "Zip archive exceeds extracted audio size limits; refusing extraction."
                )
            out.write(chunk)
    return written


def _estimate_pcm_wav_bytes(src_audio: Path) -> int | None:
    """Estimate uncompressed 16-bit PCM WAV byte size for a compressed audio file, if readable."""
    try:
        info = sf.info(str(src_audio))
        duration = float(info.duration)
        sr = max(int(info.samplerate), 44100)
        channels = max(int(info.channels), 2)
    except (sf.LibsndfileError, OSError, RuntimeError, ValueError):
        try:
            probed = media.probe(src_audio)
            duration = float(probed.duration)
            sr = 48000
            channels = 2
        except (MediaToolError, OSError, ValueError):
            return None
    if not math.isfinite(duration) or duration <= 0.0:
        return None
    return int(math.ceil(duration * sr * channels * 2))


def _check_transcode_budget(
    src_audio: Path, *, max_file_bytes: int, remaining_total_bytes: int
) -> None:
    if src_audio.suffix.lower() == ".wav":
        return
    estimated = _estimate_pcm_wav_bytes(src_audio)
    if estimated is None:
        return
    limit = min(max_file_bytes, remaining_total_bytes)
    if estimated > limit:
        raise ConfigError(
            f"Compressed stem {src_audio.name!r} would expand to ~{estimated // (1024 * 1024)} MiB "
            f"PCM WAV, exceeding the {limit // (1024 * 1024)} MiB limit."
        )


def _store_stem_audio(src_audio: Path, dest_wav: Path, *, overwrite: bool) -> Path:
    """Place an audio file at ``dest_wav`` as PCM WAV atomically."""
    if dest_wav.is_file() and not overwrite:
        return dest_wav
    if src_audio.suffix.lower() == ".wav":
        return atomic_copy(src_audio, dest_wav)
    return media.transcode_to_wav(src_audio, dest_wav)


def extract_zip_stems(
    zip_path: Path,
    dest_dir: Path,
    *,
    overwrite: bool = False,
    existing_ids: set[str] | None = None,
    budget_bytes: list[int] | None = None,
) -> list[IngestedStem]:
    """Safely extract supported stem audio files from ``zip_path`` into ``dest_dir``."""
    if not zip_path.is_file():
        raise ConfigError(f"Stem archive not found: {zip_path}")
    ids = existing_ids if existing_ids is not None else set()
    remaining = budget_bytes if budget_bytes is not None else [MAX_TOTAL_STEM_BYTES]
    dest_dir.mkdir(parents=True, exist_ok=True)

    extracted: list[IngestedStem] = []
    try:
        zf_ctx = zipfile.ZipFile(zip_path, "r")
    except zipfile.BadZipFile as exc:
        raise ConfigError(f"Invalid zip archive {zip_path}: {exc}") from exc

    with zf_ctx as zf:
        infos = zf.infolist()
        if len(infos) > MAX_ZIP_ENTRIES:
            raise ConfigError(
                f"Zip archive {zip_path.name!r} has too many entries "
                f"({len(infos)} > {MAX_ZIP_ENTRIES})."
            )
        audio_entries: list[tuple[zipfile.ZipInfo, str]] = []
        for info in infos:
            basename = _validate_zip_member(info)
            if basename is not None:
                audio_entries.append((info, basename))

        if not audio_entries:
            raise ConfigError(
                f"No supported audio stems ({', '.join(sorted(STEM_AUDIO_SUFFIXES))}) "
                f"found inside {zip_path.name}."
            )
        if len(ids) + len(audio_entries) > MAX_STEM_FILES:
            raise ConfigError(
                f"Too many stem files ({len(ids) + len(audio_entries)} > {MAX_STEM_FILES})."
            )

        with tempfile.TemporaryDirectory(prefix="mvd_zip_stems_") as tmp:
            tmp_root = Path(tmp)
            for idx, (info, basename) in enumerate(audio_entries):
                label = clean_stem_label(basename)
                role = classify_stem_name(basename)
                stem_id = make_stem_id(label, ids)
                dest_wav = dest_dir / f"{stem_id}.wav"
                if not overwrite and dest_wav.is_file() and not dest_wav.is_symlink():
                    extracted.append(IngestedStem(id=stem_id, name=label, role=role, path=dest_wav))
                    continue

                suffix = Path(basename).suffix.lower()
                staged = tmp_root / f"entry_{idx}{suffix}"
                try:
                    try:
                        with zf.open(info, "r") as member_stream:
                            nbytes = _write_bounded_stream(
                                member_stream,
                                staged,
                                max_file_bytes=MAX_STEM_FILE_BYTES,
                                remaining_total_bytes=remaining[0],
                            )
                    except (zipfile.BadZipFile, zlib.error, RuntimeError, EOFError) as exc:
                        raise ConfigError(
                            f"Corrupt or unsupported zip entry {basename!r} in {zip_path}: {exc}"
                        ) from exc
                    _check_transcode_budget(
                        staged,
                        max_file_bytes=MAX_STEM_FILE_BYTES,
                        remaining_total_bytes=remaining[0],
                    )
                    written_path = _store_stem_audio(staged, dest_wav, overwrite=overwrite)
                    try:
                        actual_bytes = max(nbytes, written_path.stat().st_size)
                    except OSError:
                        actual_bytes = nbytes
                    remaining[0] = max(0, remaining[0] - actual_bytes)
                finally:
                    staged.unlink(missing_ok=True)
                extracted.append(IngestedStem(id=stem_id, name=label, role=role, path=written_path))

    log.info("Extracted %d stems from %s into %s", len(extracted), zip_path.name, dest_dir)
    return extracted


def _collect_directory_sources(directory: Path) -> list[Path]:
    """Collect non-symlink zip archives and audio stem files from a directory tree."""
    root = directory.resolve()
    collected: list[Path] = []
    for entry in sorted(directory.rglob("*")):
        if entry.is_symlink():
            continue
        if _is_ignored_entry(entry.relative_to(directory).as_posix()):
            continue
        if not entry.is_file():
            continue
        try:
            entry.resolve().relative_to(root)
        except ValueError:
            continue
        suffix = entry.suffix.lower()
        if suffix in ARCHIVE_SUFFIXES or suffix in STEM_AUDIO_SUFFIXES:
            collected.append(entry)
    return collected


def ingest_stem_sources(
    sources: Sequence[Path],
    stems_dir: Path,
    *,
    overwrite: bool = False,
) -> list[IngestedStem]:
    """Ingest any combination of .zip archives, directories, and audio files into ``stems_dir``."""
    expanded: list[Path] = []
    for src in sources:
        if not src.exists():
            raise ConfigError(f"Stem source not found: {src}")
        if src.is_dir():
            expanded.extend(_collect_directory_sources(src))
        elif src.is_file():
            suffix = src.suffix.lower()
            if suffix in ARCHIVE_SUFFIXES or suffix in STEM_AUDIO_SUFFIXES:
                expanded.append(src)
            else:
                supported = ", ".join(sorted(STEM_AUDIO_SUFFIXES | ARCHIVE_SUFFIXES))
                raise ConfigError(
                    f"Unsupported stem file {src.name!r}. Supported formats: {supported}."
                )

    if not expanded:
        return []

    stems_dir.mkdir(parents=True, exist_ok=True)
    stems_dir_resolved = stems_dir.resolve()
    existing_ids: set[str] = set()
    budget = [MAX_TOTAL_STEM_BYTES]
    ingested: list[IngestedStem] = []

    # Process .zip archives before loose audio files so previously extracted WAVs sitting in
    # stems_dir alongside a zip are not duplicated as <id>_2.wav on --overwrite re-runs.
    ordered = sorted(expanded, key=lambda p: 0 if p.suffix.lower() in ARCHIVE_SUFFIXES else 1)

    for src in ordered:
        if src.suffix.lower() in ARCHIVE_SUFFIXES:
            ingested.extend(
                extract_zip_stems(
                    src,
                    stems_dir,
                    overwrite=overwrite,
                    existing_ids=existing_ids,
                    budget_bytes=budget,
                )
            )
        else:
            label = clean_stem_label(src.name)
            role = classify_stem_name(src.name)
            base_slug = make_stem_id(label, set())
            if src.parent.resolve() == stems_dir_resolved and base_slug in existing_ids:
                continue
            if len(existing_ids) + 1 > MAX_STEM_FILES:
                raise ConfigError(f"Too many stem files (> {MAX_STEM_FILES}).")
            stem_id = make_stem_id(label, existing_ids)
            dest_wav = stems_dir / f"{stem_id}.wav"
            if src.resolve() == dest_wav.resolve():
                written_path = dest_wav
            else:
                written_path = _store_stem_audio(src, dest_wav, overwrite=overwrite)
            ingested.append(IngestedStem(id=stem_id, name=label, role=role, path=written_path))

    return ingested


def discover_stems(stems_dir: Path) -> list[IngestedStem]:
    """Discover already-extracted stem audio files in ``stems_dir`` (ignores .zip archives)."""
    if not stems_dir.is_dir():
        return []
    root = stems_dir.resolve()
    ids: set[str] = set()
    stems: list[IngestedStem] = []
    for entry in sorted(stems_dir.iterdir()):
        if entry.is_symlink() or not entry.is_file() or _is_ignored_entry(entry.name):
            continue
        if entry.suffix.lower() not in STEM_AUDIO_SUFFIXES:
            continue
        try:
            entry.resolve().relative_to(root)
        except ValueError:
            continue
        label = clean_stem_label(entry.name)
        role = classify_stem_name(entry.name)
        stem_id = make_stem_id(label, ids)
        stems.append(IngestedStem(id=stem_id, name=label, role=role, path=entry))
    return stems


def select_vocal_stems(stems: Sequence[IngestedStem]) -> list[IngestedStem]:
    """Select the cleanest lead-vocal stem(s) for lip-sync, avoiding backing vocals."""
    leads = [s for s in stems if s.role == "lead_vocals"]
    if leads:
        return leads
    return [s for s in stems if s.role == "vocals"]


def select_drum_stems(stems: Sequence[IngestedStem]) -> list[IngestedStem]:
    """Select drum stems (or percussion stems if no drum kit exists) for beat tracking."""
    drums = [s for s in stems if s.role == "drums"]
    if drums:
        return drums
    return [s for s in stems if s.role == "percussion"]


def select_rhythm_stems(stems: Sequence[IngestedStem]) -> list[IngestedStem]:
    """Select all drum and percussion stems for beat tracking and drop detection."""
    return [s for s in stems if s.role in RHYTHM_ROLES]


def select_harmonic_stems(stems: Sequence[IngestedStem]) -> list[IngestedStem]:
    """Select pitched/harmonic instrument stems (excluding drums, percussion, and vocals)."""
    specific = [s for s in stems if s.role in HARMONIC_ROLES and s.role != "instrumental"]
    if specific:
        return specific
    return [s for s in stems if s.role == "instrumental"]


def select_mix_stems(stems: Sequence[IngestedStem]) -> list[IngestedStem]:
    """Select non-redundant stems to synthesize a full mix when no master mix is provided.

    When both individual instrument stems and a full ``instrumental`` stem exist, the
    ``instrumental`` stem is excluded so instruments are not double-counted. For a 2-stem
    Suno Split from Mix (``Vocals`` + ``Instrumental``), both are summed.
    """
    non_vocal_specific = [
        s for s in stems if s.role not in VOCAL_ROLES and s.role != "instrumental"
    ]
    if non_vocal_specific:
        return [s for s in stems if s.role != "instrumental"]
    return list(stems)


def mix_stems_to_wav(stem_paths: Sequence[Path], dst: Path) -> Path:
    """Sum one or more audio stems into a normalized, non-clipping 16-bit PCM WAV atomically."""
    if not stem_paths:
        raise MediaToolError("Cannot mix an empty list of stems.")
    if len(stem_paths) == 1:
        return _store_stem_audio(stem_paths[0], dst, overwrite=True)

    target_sr: int | None = None
    arrays: list[np.ndarray] = []
    max_channels = 1

    for path in stem_paths:
        if not path.is_file():
            raise MediaToolError(f"Stem audio file not found: {path}")
        try:
            if path.suffix.lower() == ".wav":
                data, sr = sf.read(str(path), dtype="float64", always_2d=True)
            else:
                import librosa

                mono, sr = librosa.load(str(path), sr=target_sr, mono=False)
                data = mono.T if mono.ndim == 2 else mono[:, None]
                data = data.astype(np.float64, copy=False)
        except (sf.LibsndfileError, OSError, ValueError) as exc:
            raise MediaToolError(f"Could not read stem audio {path}: {exc}") from exc

        if int(sr) <= 0:
            raise MediaToolError(f"Stem audio has invalid sample rate ({sr}): {path}")
        if target_sr is None:
            target_sr = int(sr)
        elif int(sr) != target_sr and data.shape[0] > 0:
            import librosa

            resampled_cols = [
                librosa.resample(
                    data[:, ch].astype(np.float32), orig_sr=int(sr), target_sr=target_sr
                ).astype(np.float64)
                for ch in range(data.shape[1])
            ]
            data = np.column_stack(resampled_cols)
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
        max_channels = max(max_channels, data.shape[1])
        arrays.append(data)

    assert target_sr is not None
    max_len = max(a.shape[0] for a in arrays)
    mixed = np.zeros((max_len, max_channels), dtype=np.float64)
    for arr in arrays:
        if arr.shape[1] == 1 and max_channels > 1:
            arr = np.repeat(arr, max_channels, axis=1)
        mixed[: arr.shape[0], : arr.shape[1]] += arr

    peak = float(np.max(np.abs(mixed))) if mixed.size else 0.0
    if math.isfinite(peak) and peak > 0.98:
        mixed *= 0.98 / peak

    dst.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=".mvd_mix_", suffix=".wav", dir=dst.parent)
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        sf.write(str(tmp_path), mixed, target_sr, subtype="PCM_16")
        os.replace(tmp_path, dst)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise
    return dst


def load_stems_mono_sum(
    stems: Sequence[IngestedStem], sr: int, *, loader=None
) -> np.ndarray | None:
    """Load and sum a group of stems to a single mono float32 array at ``sr``."""
    if not stems:
        return None
    if loader is None:
        from mvdirector.audio.analysis import load_mono

        loader = load_mono

    arrays = [np.asarray(loader(s.path, sr), dtype=np.float32) for s in stems]
    if not arrays:
        return None
    if len(arrays) == 1:
        return arrays[0]
    max_len = max(a.size for a in arrays)
    out = np.zeros(max_len, dtype=np.float32)
    for arr in arrays:
        out[: arr.size] += arr
    peak = float(np.max(np.abs(out))) if out.size else 0.0
    if math.isfinite(peak) and peak > 1.0:
        out /= np.float32(peak)
    return out


def _detect_rhythm_drops(
    rhythm_bar_energy: list[float], bars: list[float], *, min_jump: float = 0.45
) -> list[float]:
    """Return bar start times where drums+bass energy sharply drops out or slams back in."""
    if len(rhythm_bar_energy) < 3 or len(bars) < 3:
        return []
    count = min(len(rhythm_bar_energy), len(bars))
    drops: list[float] = []
    for idx in range(1, count):
        prev_e = rhythm_bar_energy[idx - 1]
        curr_e = rhythm_bar_energy[idx]
        if abs(curr_e - prev_e) >= min_jump:
            bar_t = round(float(bars[idx]), 3)
            if not drops or bar_t - drops[-1] >= 1.5:
                drops.append(bar_t)
    return drops


def _format_arrangement_summary(
    tracks: Sequence[StemTrackInfo],
    backing_windows: Sequence[TimeSpan],
    rhythm_drops: Sequence[float],
    duration: float,
) -> str:
    """Build a compact, token-efficient instrumentation and arrangement summary for prompts."""
    if not tracks:
        return ""
    audible = [t for t in tracks if t.active_fraction > 0.01]
    if not audible:
        return ""

    instruments = [t.name for t in audible if t.role not in VOCAL_ROLES]
    vocals = [t.name for t in audible if t.role in VOCAL_ROLES]

    parts: list[str] = []
    if vocals:
        parts.append("Vocals: " + ", ".join(vocals))
    if instruments:
        parts.append("Instruments: " + ", ".join(instruments))

    # Summarize arrangement evolution across up to 4 equal song quarters.
    if duration > 0 and len(audible) >= 2:
        buckets = 4 if duration >= 24.0 else 2
        step = duration / buckets
        arc_bits: list[str] = []
        for idx in range(buckets):
            b_start = idx * step
            b_end = (idx + 1) * step
            active_here = [
                t.name
                for t in audible
                if sum(sp.overlap(b_start, b_end) for sp in t.active_spans) >= 0.25 * step
            ]
            if active_here:
                m_s, s_s = divmod(int(round(b_start)), 60)
                m_e, s_e = divmod(int(round(b_end)), 60)
                arc_bits.append(f"{m_s}:{s_s:02d}-{m_e}:{s_e:02d} [{', '.join(active_here[:6])}]")
        if arc_bits:
            parts.append("Arrangement arc: " + "; ".join(arc_bits))

    if backing_windows:
        bv_sec = sum(w.duration for w in backing_windows)
        parts.append(f"Backing vocals active in {len(backing_windows)} sections ({bv_sec:.1f}s)")
    if rhythm_drops:
        drop_labels = ", ".join(
            f"{int(t // 60)}:{int(round(t % 60)):02d}" for t in rhythm_drops[:6]
        )
        parts.append(f"Rhythm/bass transitions at {drop_labels}")

    return ". ".join(parts) + "."


def analyze_stems(
    stems: Sequence[IngestedStem],
    *,
    project_root: Path,
    bars: list[float],
    duration: float,
    sr: int = 22050,
    loader=None,
) -> StemArrangement | None:
    """Measure per-stem activity spans, energy share, backing vocal windows, and rhythm drops."""
    if not stems:
        return None

    from mvdirector.audio import analysis as analysis_mod

    load_fn = loader or analysis_mod.load_mono
    raw_rms: list[tuple[IngestedStem, np.ndarray, np.ndarray, float]] = []
    total_energy = 0.0
    rhythm_samples: list[np.ndarray] = []

    for stem in stems:
        samples = np.asarray(load_fn(stem.path, sr), dtype=np.float32)
        rms, times = analysis_mod.rms_envelope(samples, sr)
        mean_energy = float(np.mean(rms)) if rms.size else 0.0
        total_energy += mean_energy
        raw_rms.append((stem, rms, times, mean_energy))
        if stem.role in RHYTHM_ROLES or stem.role == "bass":
            rhythm_samples.append(samples)

    tracks: list[StemTrackInfo] = []
    backing_windows: list[TimeSpan] = []

    for stem, rms, times, mean_energy in raw_rms:
        spans = analysis_mod.detect_vocal_windows(
            rms, times, threshold_db=-28.0, min_duration=0.4, merge_gap=0.8
        )
        active_sec = sum(sp.duration for sp in spans)
        active_frac = min(1.0, active_sec / duration) if duration > 0 else 0.0
        energy_share = (mean_energy / total_energy) if total_energy > 1e-9 else 0.0
        try:
            rel_path = stem.path.resolve().relative_to(project_root.resolve()).as_posix()
        except ValueError:
            rel_path = stem.path.name
        tracks.append(
            StemTrackInfo(
                id=stem.id,
                name=stem.name,
                role=stem.role,
                path=rel_path,
                active_fraction=round(active_frac, 3),
                energy_share=round(energy_share, 3),
                active_spans=spans,
            )
        )
        if stem.role == "backing_vocals":
            backing_windows.extend(spans)

    # Merge overlapping backing vocal windows if multiple backing vocal stems were provided.
    if len(backing_windows) > 1:
        backing_windows.sort(key=lambda w: (w.start, w.end))
        merged_bv: list[TimeSpan] = [backing_windows[0]]
        for span in backing_windows[1:]:
            if span.start <= merged_bv[-1].end + 0.5:
                merged_bv[-1] = TimeSpan(
                    start=merged_bv[-1].start, end=max(merged_bv[-1].end, span.end)
                )
            else:
                merged_bv.append(span)
        backing_windows = merged_bv

    rhythm_drops: list[float] = []
    if rhythm_samples and bars:
        max_len = max(a.size for a in rhythm_samples)
        rhythm_sum = np.zeros(max_len, dtype=np.float32)
        for arr in rhythm_samples:
            rhythm_sum[: arr.size] += arr
        r_rms, r_times = analysis_mod.rms_envelope(rhythm_sum, sr)
        r_bars = analysis_mod.bar_intensity(r_rms, r_times, bars, duration)
        rhythm_drops = _detect_rhythm_drops(r_bars, bars)

    summary = _format_arrangement_summary(tracks, backing_windows, rhythm_drops, duration)
    return StemArrangement(
        stems=tracks,
        backing_vocal_windows=backing_windows,
        rhythm_drops=rhythm_drops,
        summary=summary,
    )


def active_stems_in_span(
    arrangement: StemArrangement | None,
    start: float,
    end: float,
    *,
    min_fraction: float = 0.25,
    max_items: int = 6,
) -> list[str]:
    """Return names of non-lead stems audibly active during ``[start, end]``."""
    if arrangement is None or end <= start:
        return []
    dur = end - start
    hits: list[tuple[float, str]] = []
    for track in arrangement.stems:
        if track.role in ("lead_vocals", "vocals"):
            continue
        overlap = sum(sp.overlap(start, end) for sp in track.active_spans)
        if overlap / dur >= min_fraction:
            hits.append((track.energy_share, track.name))
    hits.sort(key=lambda item: item[0], reverse=True)
    return [name for _, name in hits[:max_items]]
