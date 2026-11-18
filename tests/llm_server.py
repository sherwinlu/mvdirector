# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""A fake language-model server that speaks the OpenAI, Ollama and Gemini protocols.

It listens on a real local socket, so tests exercise the actual HTTP stack end to end. Replies
depend on which prompt it receives (bible, shot direction, vision check, rewrite, or the
``mvd llm-check`` probes), so the whole planning flow can run against it.
"""

from __future__ import annotations

import json
import re
import threading
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

BIBLE = {
    "appearance": "A singer with a small beauty mark under her left eye.",
    "style": "glossy editorial",
    "palette": "obsidian and silver",
    "lighting": "hard side light",
    "negative_prompt": "blurry",
    "concept": "A knight turns a cage into a stage.",
    "rules": ["Keep the face visible."],
    "wardrobe": [
        {"id": "cape", "description": "bodysuit and a black cape"},
        {"id": "bare", "description": "bodysuit, no cape"},
    ],
    "settings": [
        {
            "name": "Corridor",
            "sections": ["Verse"],
            "description": "a narrow black corridor",
            "movement": "walks toward camera",
        }
    ],
}
VERDICT = {
    "spec_match": 0.9,
    "artifact_free": 0.9,
    "character_match": 0.9,
    "issues": [],
    "fix": "",
}


@dataclass
class Seen:
    path: str
    headers: dict[str, str]
    body: dict[str, Any]
    kind: str = ""
    images: int = 0


@dataclass
class FakeLLMServer:
    vision: bool = True  # False: reject image input like a text-only model
    invalid_first: bool = False  # answer the first request with text that is not JSON
    verdicts: list[dict[str, Any]] = field(
        default_factory=lambda: [VERDICT]
    )  # vision replies in order
    requests: list[Seen] = field(default_factory=list)
    _server: ThreadingHTTPServer | None = None

    # ------------------------------------------------------------------ lifecycle

    @property
    def url(self) -> str:
        assert self._server is not None
        return f"http://127.0.0.1:{self._server.server_address[1]}"

    def start(self) -> FakeLLMServer:
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args: Any) -> None:  # keep test output quiet
                pass

            def do_POST(self) -> None:
                length = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(length) or b"{}")
                status, payload = owner.handle(self.path, dict(self.headers), body)
                data = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self._server.serve_forever, daemon=True).start()
        return self

    def stop(self) -> None:
        if self._server:
            self._server.shutdown()
            self._server.server_close()

    def of_kind(self, kind: str) -> list[Seen]:
        return [r for r in self.requests if r.kind == kind]

    # ---------------------------------------------------------------- protocols

    def handle(
        self, path: str, headers: dict[str, Any], body: dict[str, Any]
    ) -> tuple[int, dict[str, Any]]:
        lowered = {k.lower(): v for k, v in headers.items()}
        reject: tuple[int, dict[str, Any]]
        if path.endswith("/chat/completions"):
            system, text, images = _read_openai(body)
            wrap, reject = (
                _wrap_openai,
                (400, {"error": {"message": "This model does not support image input"}}),
            )
        elif path == "/api/chat":
            system, text, images = _read_ollama(body)
            wrap, reject = _wrap_ollama, (400, {"error": "this model does not support images"})
        elif ":generateContent" in path:
            system, text, images = _read_gemini(body)
            wrap, reject = (
                _wrap_gemini,
                (400, {"error": {"message": "Image input is not supported"}}),
            )
        else:
            return 404, {"error": "unknown path"}
        seen = Seen(path, lowered, body, kind=_kind(system), images=images)
        self.requests.append(seen)
        if images and not self.vision:
            return reject
        if self.invalid_first and len(self.requests) == 1:
            return 200, wrap("Sorry, here is some prose and no JSON.")
        return 200, wrap(json.dumps(self.answer(seen.kind, text, images)))

    def answer(self, kind: str, text: str, images: int) -> dict[str, Any]:
        if kind == "bible":
            return BIBLE
        if kind == "shots":
            ids = re.findall(r"- id=(s\d+)", text)
            return {
                "shots": [
                    {
                        "id": shot_id,
                        "type": "lipsync",
                        "framing": "close_up",
                        "setting": "Corridor",
                        "wardrobe": "cape",
                        "still_prompt": f"still prompt for {shot_id}",
                        "motion_prompt": f"motion prompt for {shot_id}",
                    }
                    for shot_id in ids
                ]
            }
        if kind == "vision":
            return self.verdicts.pop(0) if len(self.verdicts) > 1 else self.verdicts[0]
        if kind == "rewrite":
            return {"prompt": "A rewritten prompt."}
        if images:
            return {"color": "blue"}
        return {"ok": True, "word": "ready"}


def _kind(system: str) -> str:
    if "Write the visual bible" in system:
        return "bible"
    if "exactly one entry per shot id" in system:
        return "shots"
    if "quality-control supervisor" in system:
        return "vision"
    if "You fix prompts" in system:
        return "rewrite"
    return "check"


def _read_openai(body: dict[str, Any]) -> tuple[str, str, int]:
    system = body["messages"][0]["content"]
    texts, images = [], 0
    for message in body["messages"][1:]:
        content = message["content"]
        parts: list[dict[str, Any]] = (
            [{"type": "text", "text": content}] if isinstance(content, str) else content
        )
        for part in parts:
            if part["type"] == "image_url":
                assert part["image_url"]["url"].startswith("data:image/jpeg;base64,")
                images += 1
            else:
                texts.append(part["text"])
    return system, "\n".join(texts), images


def _read_ollama(body: dict[str, Any]) -> tuple[str, str, int]:
    system = body["messages"][0]["content"]
    texts, images = [], 0
    for message in body["messages"][1:]:
        texts.append(message["content"])
        images += len(message.get("images", []))
    return system, "\n".join(texts), images


def _read_gemini(body: dict[str, Any]) -> tuple[str, str, int]:
    system = body["systemInstruction"]["parts"][0]["text"]
    texts, images = [], 0
    for content in body["contents"]:
        for part in content["parts"]:
            if "inlineData" in part:
                images += 1
            else:
                texts.append(part["text"])
    return system, "\n".join(texts), images


def _wrap_openai(text: str) -> dict[str, Any]:
    return {
        "choices": [{"message": {"role": "assistant", "content": text}, "finish_reason": "stop"}]
    }


def _wrap_ollama(text: str) -> dict[str, Any]:
    return {
        "message": {"role": "assistant", "content": text},
        "done": True,
        "done_reason": "stop",
        "prompt_eval_count": 500,
    }


def _wrap_gemini(text: str) -> dict[str, Any]:
    return {"candidates": [{"content": {"parts": [{"text": text}]}, "finishReason": "STOP"}]}
