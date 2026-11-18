# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Where a ComfyUI client comes from.

The engine never decides which GPU endpoint serves a job. It asks a `ComfyEndpointProvider` for a
leased client and releases it when the stage is done. The default provider reads the single
`comfy.base_url` from the project config. A hosted service can implement the protocol to route
each lease to a pod pool, check health, enforce concurrency limits or cold-start capacity.
"""

from __future__ import annotations

from contextlib import AbstractContextManager
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from mvdirector.comfy.client import ComfyClient

if TYPE_CHECKING:
    from mvdirector.project import Project


@runtime_checkable
class ComfyEndpointProvider(Protocol):
    def lease(self, project: Project) -> AbstractContextManager[ComfyClient]:
        """Return a context manager yielding a ready client; exiting it releases the endpoint."""
        ...


class StaticComfyProvider:
    """Use the endpoint named in the project's `comfy` config. This is the CLI behaviour."""

    def lease(self, project: Project) -> AbstractContextManager[ComfyClient]:
        return ComfyClient(project.config.comfy, upload_subfolder=project.comfy_subfolder)
