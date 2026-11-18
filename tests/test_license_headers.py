# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Every source file carries the MPL-2.0 notice."""

from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NOTICE = (
    "# This Source Code Form is subject to the terms of the Mozilla Public\n"
    "# License, v. 2.0. If a copy of the MPL was not distributed with this\n"
    "# file, You can obtain one at https://mozilla.org/MPL/2.0/.\n"
)


def _source_files() -> list[Path]:
    listed = subprocess.run(
        ["git", "ls-files", "*.py", "docker/*.sh"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    return [ROOT / name for name in listed]


def test_every_source_file_starts_with_the_mpl_notice():
    files = _source_files()
    assert files, "git ls-files found no source files"
    missing = []
    for path in files:
        text = path.read_text(encoding="utf-8")
        body = text.split("\n", 1)[1] if text.startswith("#!") else text  # keep the shebang first
        if not body.startswith(NOTICE):
            missing.append(str(path.relative_to(ROOT)))
    assert not missing, f"missing the MPL-2.0 notice: {missing}"


def test_license_file_is_the_mpl_2_text():
    text = (ROOT / "LICENSE").read_text(encoding="utf-8")
    assert text.startswith("Mozilla Public License Version 2.0")
    assert "Exhibit A - Source Code Form License Notice" in text
