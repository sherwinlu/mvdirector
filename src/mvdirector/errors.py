# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Exception hierarchy. Every error the pipeline raises on purpose derives from MVDError."""

from __future__ import annotations


class MVDError(Exception):
    """Base class for all expected pipeline errors."""


class ConfigError(MVDError):
    """The project configuration or layout is invalid."""


class MediaToolError(MVDError):
    """ffmpeg or ffprobe is missing or returned an error."""


class ComfyError(MVDError):
    """Base class for ComfyUI errors."""


class ComfyConnectionError(ComfyError):
    """ComfyUI could not be reached. The run should stop; retrying a shot will not help."""


class ComfyJobError(ComfyError):
    """ComfyUI accepted the request but the job failed. Counts as a failed attempt."""


class WorkflowBindingError(MVDError):
    """A workflow template and its bindings file do not agree."""


class LLMError(MVDError):
    """The language model call failed or returned unusable output."""


class LLMUnavailableError(LLMError):
    """The model cannot be used at all: unreachable, bad credentials, or no such model.

    Unlike a single bad reply, trying the next request will not help, so callers stop asking.
    """


class ImagesNotSupportedError(LLMError):
    """The configured model or server does not accept image input."""


class DependencyMissing(MVDError):
    """An optional dependency needed for a requested feature is not installed."""


class BudgetExceeded(MVDError):
    """The per-run generation budget is used up."""


class ProjectLocked(MVDError):
    """Another generation run holds the project lock."""
