# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Workflow templates: a ComfyUI graph exported in API format plus a bindings file.

The pipeline never hard-codes node ids. A bindings file maps logical parameter names
(``positive_prompt``, ``seed``, ``image`` ...) onto ``node id + input field`` pairs in
your own exported workflow, so workflows can be swapped or upgraded without code changes.
"""

from __future__ import annotations

import copy
import json
import logging
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from mvdirector.config import expand_env
from mvdirector.errors import ConfigError, WorkflowBindingError

log = logging.getLogger(__name__)


class Binding(BaseModel):
    model_config = ConfigDict(extra="forbid")
    node: str
    field: str


class OutputSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    node: str | None = Field(
        default=None, description="Node whose output to download; any node if omitted."
    )
    extensions: list[str] = Field(default_factory=lambda: [".png"])


class WorkflowBindings(BaseModel):
    model_config = ConfigDict(extra="forbid")
    description: str = ""
    workflow: str = Field(description="API-format workflow JSON, relative to this file.")
    inputs: dict[str, list[Binding]]
    required: list[str] = Field(default_factory=list)
    defaults: dict[str, Any] = Field(default_factory=dict)
    output: OutputSpec = Field(default_factory=OutputSpec)


class WorkflowTemplate:
    def __init__(self, name: str, bindings: WorkflowBindings, graph: dict[str, Any]) -> None:
        self.name = name
        self.bindings = bindings
        self._graph = graph

    @classmethod
    def load(cls, name: str, bindings_path: Path, root: Path | None = None) -> WorkflowTemplate:
        if not bindings_path.is_file():
            raise WorkflowBindingError(
                f"Bindings file for workflow '{name}' not found: {bindings_path}"
            )
        try:
            bindings = WorkflowBindings.model_validate_json(bindings_path.read_text("utf-8"))
            bindings.defaults = expand_env(bindings.defaults)
        except (ValidationError, ConfigError) as exc:
            raise WorkflowBindingError(f"Invalid bindings file {bindings_path}:\n{exc}") from exc
        raw_workflow = Path(bindings.workflow)
        if raw_workflow.is_absolute():
            raise WorkflowBindingError(
                f"Workflow '{name}' path '{bindings.workflow}' must be relative."
            )
        allowed_root = (root or bindings_path.parent).resolve()
        graph_path = (bindings_path.parent / raw_workflow).resolve()
        if not graph_path.is_relative_to(allowed_root):
            raise WorkflowBindingError(
                f"Workflow '{name}' path '{bindings.workflow}' escapes the workflows folder "
                f"({allowed_root})."
            )
        if not graph_path.is_file():
            raise WorkflowBindingError(
                f"Workflow '{name}' points at {graph_path}, which does not exist. Export your "
                "workflow from ComfyUI with 'Save (API Format)' and place it there."
            )
        try:
            graph = json.loads(graph_path.read_text("utf-8"))
        except json.JSONDecodeError as exc:
            raise WorkflowBindingError(f"{graph_path} is not valid JSON: {exc}") from exc
        template = cls(name, bindings, graph)
        problems = template.problems()
        if problems:
            raise WorkflowBindingError(
                f"Workflow '{name}' has binding problems:\n- " + "\n- ".join(problems)
            )
        return template

    def accepts(self, param: str) -> bool:
        """True if the workflow has a binding for ``param``."""
        return param in self.bindings.inputs

    def problems(self) -> list[str]:
        """Human-readable mismatches between bindings and graph; empty when consistent."""
        found: list[str] = []
        if "nodes" in self._graph and "links" in self._graph:
            return ["Workflow is in UI format. Re-export it with 'Save (API Format)'."]
        for param, targets in self.bindings.inputs.items():
            for target in targets:
                node = self._graph.get(target.node)
                if node is None:
                    found.append(f"'{param}' is bound to node {target.node}, which does not exist")
                elif target.field not in node.get("inputs", {}):
                    available = ", ".join(sorted(node.get("inputs", {}))) or "none"
                    found.append(
                        f"'{param}' is bound to node {target.node} "
                        f"({node.get('class_type', '?')}), which has no input '{target.field}' "
                        f"(available: {available})"
                    )
        for param in self.bindings.required:
            if param not in self.bindings.inputs:
                found.append(f"Required parameter '{param}' has no binding")
        output_node = self.bindings.output.node
        if output_node is not None and output_node not in self._graph:
            found.append(f"Output node {output_node} does not exist")
        return found

    def render(self, params: dict[str, Any]) -> dict[str, Any]:
        """Return a copy of the graph with ``params`` (over ``defaults``) written in.

        Parameters the workflow has no binding for are ignored, so one caller can serve
        workflows with different capabilities. Missing required parameters are an error.
        """
        values = {**self.bindings.defaults, **{k: v for k, v in params.items() if v is not None}}
        missing = [name for name in self.bindings.required if name not in values]
        if missing:
            raise WorkflowBindingError(
                f"Workflow '{self.name}' is missing required parameters: {', '.join(missing)}"
            )
        graph = copy.deepcopy(self._graph)
        for param, value in values.items():
            targets = self.bindings.inputs.get(param)
            if targets is None:
                log.debug("Workflow '%s' has no binding for '%s'; ignored", self.name, param)
                continue
            for target in targets:
                graph[target.node]["inputs"][target.field] = value
        return graph
