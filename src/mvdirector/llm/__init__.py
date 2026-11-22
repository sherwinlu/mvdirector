# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Language-model access: one configured model for every text task.

The rest of the application depends only on ``LLMClient`` and the content-block helpers. The
backends (Anthropic, OpenAI and compatible servers, Ollama, Gemini) live in this package.
"""

from mvdirector.llm.base import (
    ChatBackend,
    ContentBlock,
    LLMClient,
    Reply,
    audio_block,
    complete_json_with_optional_audio,
    extract_json,
    image_block,
    maybe_audio_block,
    text_block,
)
from mvdirector.llm.factory import PROVIDERS, build_llm

__all__ = [
    "PROVIDERS",
    "ChatBackend",
    "ContentBlock",
    "LLMClient",
    "Reply",
    "audio_block",
    "build_llm",
    "complete_json_with_optional_audio",
    "extract_json",
    "image_block",
    "maybe_audio_block",
    "text_block",
]
