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
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from mvdirector.config import expand_env
from mvdirector.errors import ConfigError, WorkflowBindingError

log = logging.getLogger(__name__)

# Per-shot reference files. A shot often has none (no plate, no prop), so a binding for one of
# these that receives no value is cut out of the graph instead of keeping whatever file the
# export happened to name.
OPTIONAL_FILE_PARAMS = frozenset(
    {
        "reference_profile",
        "reference_three_quarter",
        "reference_face",
        "reference_image_2",
        "set_image",
        "prop_image",
        "prop_image_2",
        "audio_video",
        "reference_video",
        "previz_video",
        "previz_depth_video",
        "end_image",
        "previous_frame",
        "draft_grid_image",
        "sketch_image",
    }
)


_EXTRA_REF_IMAGE_RE = re.compile(r"^reference_image_([1-9][0-9]*)\Z")


def is_optional_file_param(name: str) -> bool:
    """True when ``name`` is an optional per-shot file input whose unused loader should drop."""
    if name in OPTIONAL_FILE_PARAMS:
        return True
    match = _EXTRA_REF_IMAGE_RE.match(name)
    return match is not None and int(match.group(1)) >= 2


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
    def load(
        cls,
        name: str,
        bindings_path: Path,
        root: Path | None = None,
        *,
        environ: Mapping[str, str] | None = None,
    ) -> WorkflowTemplate:
        if not bindings_path.is_file():
            raise WorkflowBindingError(
                f"Bindings file for workflow '{name}' not found: {bindings_path}"
            )
        allowed_root = (root or bindings_path.parent).resolve()
        if root is not None and not bindings_path.resolve().is_relative_to(allowed_root):
            raise WorkflowBindingError(
                f"Bindings file for workflow '{name}' escapes the project folder ({allowed_root})."
            )
        try:
            bindings = WorkflowBindings.model_validate_json(bindings_path.read_text("utf-8"))
            bindings.defaults = expand_env(bindings.defaults, environ=environ)
        except (ValidationError, ConfigError, OSError, UnicodeDecodeError) as exc:
            raise WorkflowBindingError(f"Invalid bindings file {bindings_path}:\n{exc}") from exc
        raw_workflow = Path(bindings.workflow)
        if raw_workflow.is_absolute():
            raise WorkflowBindingError(
                f"Workflow '{name}' path '{bindings.workflow}' must be relative."
            )
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
        except (json.JSONDecodeError, OSError, UnicodeDecodeError) as exc:
            raise WorkflowBindingError(f"{graph_path} is not valid JSON: {exc}") from exc
        except RecursionError:
            raise WorkflowBindingError(f"{graph_path} nests JSON too deeply to parse.") from None
        if not isinstance(graph, dict):
            raise WorkflowBindingError(
                f"{graph_path} must hold a JSON object of ComfyUI nodes, "
                f"got {type(graph).__name__}."
            )
        template = cls(name, bindings, graph)
        problems = template.problems()
        if problems:
            raise WorkflowBindingError(
                f"Workflow '{name}' has binding problems:\n- " + "\n- ".join(problems)
            )
        return template

    @property
    def graph(self) -> dict[str, Any]:
        """Return a shallow read view of the raw workflow graph."""
        return self._graph

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
                if not isinstance(node, dict):
                    found.append(f"'{param}' is bound to node {target.node}, which does not exist")
                else:
                    raw_inputs = node.get("inputs")
                    node_inputs = raw_inputs if isinstance(raw_inputs, dict) else {}
                    if target.field not in node_inputs:
                        available = ", ".join(sorted(node_inputs)) or "none"
                        found.append(
                            f"'{param}' is bound to node {target.node} "
                            f"({node.get('class_type', '?')}), which has no input "
                            f"'{target.field}' (available: {available})"
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
        bound_nodes = {
            target.node for param in values for target in self.bindings.inputs.get(param, ())
        }
        for param in self.bindings.inputs.keys() - values.keys():
            # On video workflows ("image" supplies the starting still), an unbound optional
            # "reference_image" or "audio" loader (e.g. b-roll or non-lipsync shots) should be
            # pruned. Likewise on upscale/V2V workflows ("video" supplies the source clip),
            # unbound optional "audio" or "image" loaders should be pruned.
            optional_loader = (
                is_optional_file_param(param)
                or (param in ("reference_image", "audio") and "image" in values)
                or (param in ("audio", "image") and "video" in values)
            )
            if optional_loader and param not in self.bindings.required:
                for target in self.bindings.inputs[param]:
                    if target.node not in bound_nodes:
                        _drop_loader(graph, target.node)
        return prune_switch_branches(graph, output_node=self.bindings.output.node)


_SWITCH_CLASS_TYPES = frozenset({"ComfySwitchNode", "Switch", "BooleanSwitch"})
_OUTPUT_CLASS_HINTS = ("Save", "Preview", "VideoCombine")


def prune_switch_branches(
    graph: dict[str, Any], *, output_node: str | None = None
) -> dict[str, Any]:
    """Collapse constant ``ComfySwitchNode`` branches and prune unreachable dead loaders.

    Allows a single unified ComfyUI workflow to serve multiple modes (e.g. ``i2v``, ``flf2v``,
    ``lipsync``) without ComfyUI failing on unused dummy ``LoadImage`` / ``LoadAudio`` nodes.

    Credits & Attribution:
        Static ``ComfySwitchNode`` collapsing and backward reachability pruning inspired by
        ``hexeosis-prime/nightshift``
        (https://github.com/hexeosis-prime/nightshift — ``nightshift/workflows.py``).
    """
    has_switch = any(
        str(node.get("class_type", "")) in _SWITCH_CLASS_TYPES
        for node in graph.values()
        if isinstance(node, dict)
    )
    if not has_switch:
        return graph

    resolved = copy.deepcopy(graph)
    rewire: dict[str, Any] = {}
    for node_id, node in list(resolved.items()):
        if not isinstance(node, dict):
            continue
        if str(node.get("class_type", "")) not in _SWITCH_CLASS_TYPES:
            continue
        inputs = node.get("inputs", {})
        cond = inputs.get("switch")
        if cond is None:
            cond = inputs.get("boolean")
        if not isinstance(cond, bool):
            continue
        chosen = inputs.get("on_true") if cond else inputs.get("on_false")
        if chosen is not None:
            rewire[node_id] = chosen

    if not rewire:
        return resolved

    # Follow chained switches to their final non-switch source.
    def _resolve_target(val: Any, seen: set[str]) -> Any:
        if isinstance(val, list) and len(val) >= 1 and isinstance(val[0], str):
            src = val[0]
            if src in rewire and src not in seen:
                return _resolve_target(rewire[src], seen | {src})
        return val

    for node in resolved.values():
        if not isinstance(node, dict):
            continue
        inputs = node.get("inputs", {})
        for field, val in list(inputs.items()):
            inputs[field] = _resolve_target(val, set())
    for sw_id in rewire:
        resolved.pop(sw_id, None)

    # Walk backward from output nodes to keep only reachable nodes.
    roots: list[str] = []
    if output_node is not None and output_node in resolved:
        roots.append(output_node)
    else:
        for node_id, node in resolved.items():
            ctype = str(node.get("class_type", ""))
            if any(hint in ctype for hint in _OUTPUT_CLASS_HINTS):
                roots.append(node_id)
    if not roots:
        return resolved

    reachable: set[str] = set()
    stack = list(roots)
    while stack:
        cur = stack.pop()
        if cur in reachable or cur not in resolved:
            continue
        reachable.add(cur)
        for val in resolved[cur].get("inputs", {}).values():
            if isinstance(val, list) and len(val) >= 1 and isinstance(val[0], str):
                stack.append(val[0])
    return {nid: node for nid, node in resolved.items() if nid in reachable}


def convert_ui_workflow(ui_graph: dict[str, Any], object_info: dict[str, Any]) -> dict[str, Any]:
    """Convert a standard ComfyUI UI-format workflow (``nodes``/``links``) to API prompt format.

    Uses ComfyUI's ``/object_info`` metadata to map positional ``widgets_values`` onto named input
    fields while resolving graph links.

    Credits & Attribution:
        Clean-room UI-to-API workflow conversion concept inspired by ``Edakuban/VocaVid``
        (https://github.com/Edakuban/VocaVid — ``VocaVid/comfy.py``).
    """
    if "nodes" not in ui_graph or "links" not in ui_graph:
        return copy.deepcopy(ui_graph)

    links_by_id: dict[int, tuple[str, int]] = {}
    raw_links = ui_graph.get("links")
    for link in raw_links if isinstance(raw_links, list) else []:
        if isinstance(link, list) and len(link) >= 4:
            try:
                links_by_id[int(link[0])] = (str(link[1]), int(link[2]))
            except (TypeError, ValueError):
                continue
        elif isinstance(link, dict):
            # Newer ComfyUI frontend exports can serialize links as objects.
            lid = link.get("id")
            origin_id = link.get("origin_id")
            origin_slot = link.get("origin_slot")
            if lid is not None and origin_id is not None and origin_slot is not None:
                try:
                    links_by_id[int(lid)] = (str(origin_id), int(origin_slot))
                except (TypeError, ValueError):
                    continue

    api_graph: dict[str, Any] = {}
    raw_nodes = ui_graph.get("nodes")
    for node in raw_nodes if isinstance(raw_nodes, list) else []:
        if not isinstance(node, dict):
            continue
        # Skip muted (mode=2) or bypassed (mode=4) UI nodes.
        if node.get("mode") in (2, 4):
            continue
        raw_id = node.get("id")
        if raw_id is None:
            continue
        node_id = str(raw_id)
        class_type = str(node.get("type", ""))
        if not class_type:
            continue
        inputs: dict[str, Any] = {}
        linked_names: set[str] = set()
        for inp in node.get("inputs", []) or []:
            if not isinstance(inp, dict):
                continue
            name = str(inp.get("name", ""))
            link_id = inp.get("link")
            if name and link_id is not None:
                try:
                    parsed_link = int(link_id)
                except (TypeError, ValueError):
                    continue
                if parsed_link in links_by_id:
                    src_id, src_slot = links_by_id[parsed_link]
                    inputs[name] = [src_id, src_slot]
                    linked_names.add(name)

        raw_widgets = node.get("widgets_values")
        if isinstance(raw_widgets, dict):
            for k, v in raw_widgets.items():
                if str(k) not in linked_names:
                    inputs[str(k)] = v
            api_graph[node_id] = {"class_type": class_type, "inputs": inputs}
            continue
        widgets = list(raw_widgets) if isinstance(raw_widgets, list) else []
        meta = object_info.get(class_type, {})
        meta_inputs = meta.get("input", {}) if isinstance(meta, dict) else {}
        ordered_specs: list[tuple[str, Any]] = []
        for group in ("required", "optional"):
            section = meta_inputs.get(group, {})
            if isinstance(section, dict):
                ordered_specs.extend(section.items())

        w_idx = 0
        for field_name, spec in ordered_specs:
            if field_name in linked_names:
                continue
            if not isinstance(spec, list) or not spec:
                continue
            kind = spec[0]
            is_widget = isinstance(kind, list) or kind in (
                "INT",
                "FLOAT",
                "STRING",
                "BOOLEAN",
                "COMBO",
            )
            if not is_widget:
                continue
            if w_idx >= len(widgets):
                break
            inputs[field_name] = widgets[w_idx]
            w_idx += 1
            # ComfyUI adds an implicit control_after_generate widget after seed ints (or any INT
            # input whose /object_info spec declares control_after_generate: true).
            spec_opts = spec[1] if len(spec) > 1 and isinstance(spec[1], dict) else {}
            has_control = field_name in ("seed", "noise_seed") or bool(
                spec_opts.get("control_after_generate")
            )
            if (
                kind == "INT"
                and has_control
                and w_idx < len(widgets)
                and widgets[w_idx] in ("fixed", "increment", "decrement", "randomize")
            ):
                w_idx += 1

        api_graph[node_id] = {"class_type": class_type, "inputs": inputs}
    return api_graph


def _drop_loader(graph: dict[str, Any], node_id: str) -> None:
    """Remove a ``Load...`` node and the inputs that read from it.

    The consuming input has to be optional in ComfyUI (for example one slot of a multi-image
    input); a fixed input left without a link is reported by ComfyUI when the job is queued.
    """
    node = graph.get(node_id)
    if not isinstance(node, dict) or "Load" not in str(node.get("class_type", "")):
        return
    del graph[node_id]
    for other in graph.values():
        if not isinstance(other, dict):
            continue
        inputs = other.get("inputs")
        if not isinstance(inputs, dict):
            continue
        for name in [n for n, v in inputs.items() if isinstance(v, list) and v[:1] == [node_id]]:
            del inputs[name]
