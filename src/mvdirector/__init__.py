# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""mvdirector: plan, generate, quality-check and assemble lip-synced music videos.

Public API (semver applies; while the version is 0.x, minor releases may change it):

- `mvdirector.api`: one function per pipeline stage.
- `Project`, `StateStore`, `AssetStorage`, `ComfyEndpointProvider`: the extension points.
- `Manifest`, `Analysis`, `RunEvent`, and the `MVDError` hierarchy.

Everything else (`mvdirector.generation`, `.qc`, `.planning`, `.comfy.client`, ...) is internal
and may change in any release.
"""

from __future__ import annotations

from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING, Any

try:
    __version__ = version("mvdirector")
except PackageNotFoundError:  # running from a source tree without an install
    __version__ = "0+unknown"

if TYPE_CHECKING:
    from mvdirector.comfy.provider import ComfyEndpointProvider, StaticComfyProvider
    from mvdirector.errors import MVDError
    from mvdirector.generation.runner import RunEvent
    from mvdirector.models import Analysis, Manifest
    from mvdirector.project import Project
    from mvdirector.store import AssetStorage, StateStore

# Imported on first use so `import mvdirector` stays cheap (numpy, librosa and friends are heavy).
_EXPORTS = {
    "Analysis": "mvdirector.models",
    "AssetStorage": "mvdirector.store",
    "ComfyEndpointProvider": "mvdirector.comfy.provider",
    "Manifest": "mvdirector.models",
    "MVDError": "mvdirector.errors",
    "Project": "mvdirector.project",
    "RunEvent": "mvdirector.generation.runner",
    "StateStore": "mvdirector.store",
    "StaticComfyProvider": "mvdirector.comfy.provider",
}

__all__ = [
    "__version__",
    "api",
    "Analysis",
    "AssetStorage",
    "ComfyEndpointProvider",
    "MVDError",
    "Manifest",
    "Project",
    "RunEvent",
    "StateStore",
    "StaticComfyProvider",
]


def __getattr__(name: str) -> Any:
    if name == "api":
        return import_module("mvdirector.api")
    if name in _EXPORTS:
        return getattr(import_module(_EXPORTS[name]), name)
    raise AttributeError(f"module 'mvdirector' has no attribute {name!r}")
