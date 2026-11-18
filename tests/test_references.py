# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Reference sheets per wardrobe state, set plates, the preflight, and how the runner uses them."""

import logging

import numpy as np
import pytest
from PIL import Image
from typer.testing import CliRunner

from mvdirector.cli import app
from mvdirector.comfy.client import ComfyClient
from mvdirector.errors import ConfigError
from mvdirector.generation import plates, reference
from mvdirector.generation.preflight import check_assets
from mvdirector.generation.runner import RunBudget, ShotRunner, load_templates
from mvdirector.models import (
    AssetKind,
    AssetState,
    Character,
    Framing,
    ReferenceSet,
    SetDesign,
    ShotType,
    WardrobeState,
)
from mvdirector.planning import prompts
from mvdirector.planning.planner import build_manifest
from mvdirector.qc import vision
from mvdirector.qc.faces import Face, IdentityReference
from mvdirector.qc.factory import QualityGates
from mvdirector.retry import RetryPolicy
from tests.conftest import FakeComfy, write_workflows
from tests.test_pipeline import ScriptedGate

CAPE = "Black bodysuit, silver shoulder caps and a floor-length black cape."
BARE = "The same bodysuit and shoulder caps. No cape."
PANEL_COLORS = [(200, 30, 30), (30, 200, 30), (30, 30, 200), (200, 200, 30)]


def make_sheet(path, layout="row"):
    """A sheet whose four panels are solid, distinct colors."""
    width, height = (2048, 896) if layout == "row" else (1024, 1024)
    image = Image.new("RGB", (width, height), "white")
    for index, color in enumerate(PANEL_COLORS):
        if layout == "row":
            box = (index * width // 4, 0, (index + 1) * width // 4, height)
        else:
            x, y = index % 2, index // 2
            box = (x * width // 2, y * height // 2, (x + 1) * width // 2, (y + 1) * height // 2)
        image.paste(Image.new("RGB", (box[2] - box[0], box[3] - box[1]), color), box[:2])
    image.save(path)
    return path


@pytest.fixture
def staged(project, analysis):
    """A manifest with two costumes, two sets, and lip-sync shots s002 (cape) and s004 (bare)."""
    project.config.video.max_clip_seconds = 4.0
    manifest = build_manifest(project.config, analysis, "lyrics", "style", llm=None)
    manifest.character = Character(
        appearance="Singer with a beauty mark under her left eye.",
        wardrobe=[
            WardrobeState(id="cape", description=CAPE),
            WardrobeState(id="bare", description=BARE),
        ],
    )
    manifest.visual_bible.negative_prompt = "blurry"
    manifest.visual_bible.style = "glossy"
    manifest.visual_bible.settings = [
        SetDesign(name="Corridor", description="a narrow black corridor"),
        SetDesign(name="The Stage", description="a large black stage"),
    ]
    manifest.shot("s002").setting, manifest.shot("s002").wardrobe = "Corridor", "cape"
    manifest.shot("s004").setting, manifest.shot("s004").wardrobe = "The Stage", "bare"
    return manifest


def lock(project, manifest, tmp_path, key):
    path = make_sheet(tmp_path / f"{key}_in.png")
    return reference.lock_reference(project, manifest, path, state_id=key)


def comfy_client(project, fake):
    return ComfyClient(project.config.comfy, transport=fake.transport(), sleep=lambda _: None)


# ------------------------------------------------------------------ splitting a sheet


def _center(image):
    return image.getpixel((image.width // 2, image.height // 2))


def test_split_row_sheet_gives_panels_in_order(tmp_path):
    sheet = Image.open(make_sheet(tmp_path / "s.png", "row"))
    panels = reference.split_sheet(sheet, "row", inset=0.0)
    assert list(panels) == ["front", "profile", "three_quarter", "face"]
    assert [_center(panels[n]) for n in panels] == PANEL_COLORS
    assert panels["front"].size == (512, 896)


def test_split_grid_sheet_reads_left_to_right_then_down(tmp_path):
    sheet = Image.open(make_sheet(tmp_path / "s.png", "grid"))
    panels = reference.split_sheet(sheet, "grid", inset=0.0)
    assert [_center(panels[n]) for n in panels] == PANEL_COLORS
    assert panels["face"].size == (512, 512)


def test_inset_trims_every_edge(tmp_path):
    sheet = Image.open(make_sheet(tmp_path / "s.png", "row"))
    panel = reference.split_sheet(sheet, "row", inset=0.1)["front"]
    assert panel.size == (512 - 2 * 51, 896 - 2 * 89)


# --------------------------------------------------------------------- locking sheets


def test_locking_cuts_panels_and_records_them(project, staged, tmp_path):
    locked = lock(project, staged, tmp_path, "cape")
    assert locked.sheet == "assets/reference/cape_sheet.png"
    assert set(locked.panels) == {"front", "profile", "three_quarter", "face"}
    assert locked.panels["front"] == "assets/reference/cape/cape_front.png"
    front = Image.open(project.path(locked.panels["front"]))
    assert _center(front) == PANEL_COLORS[0]
    assert staged.character.references["cape"] == locked


def test_each_wardrobe_state_keeps_its_own_sheet(project, staged, tmp_path):
    lock(project, staged, tmp_path, "cape")
    lock(project, staged, tmp_path, "bare")
    refs = staged.character.references
    assert set(refs) == {"cape", "bare"}
    assert refs["cape"].panels["front"] != refs["bare"].panels["front"]
    assert project.path(refs["cape"].panels["front"]).is_file()


def test_locking_without_a_state_uses_the_first_costume(project, staged, tmp_path):
    reference.lock_reference(project, staged, make_sheet(tmp_path / "x.png"))
    assert list(staged.character.references) == ["cape"]


def test_locking_a_character_without_costumes_uses_the_base_key(project, analysis, tmp_path):
    manifest = build_manifest(project.config, analysis, "lyrics", "style", llm=None)
    reference.lock_reference(project, manifest, make_sheet(tmp_path / "x.png"))
    assert list(manifest.character.references) == ["base"]


def test_locking_rejects_an_unknown_state(project, staged, tmp_path):
    with pytest.raises(ConfigError, match="Unknown wardrobe state 'gown'.*cape, bare"):
        reference.lock_reference(project, staged, make_sheet(tmp_path / "x.png"), state_id="gown")


def test_locking_rejects_an_image_that_is_not_a_row_sheet(project, staged, tmp_path):
    square = tmp_path / "portrait.png"
    Image.new("RGB", (800, 800), "grey").save(square)
    with pytest.raises(ConfigError, match="does not look like a 'row' sheet.*--single"):
        reference.lock_reference(project, staged, square, state_id="cape")


def test_single_image_is_kept_whole_as_the_front_reference(project, staged, tmp_path):
    square = tmp_path / "portrait.png"
    Image.new("RGB", (800, 800), "grey").save(square)
    locked = reference.lock_reference(project, staged, square, state_id="cape", split=False)
    assert locked.panels == {} and locked.panel("front") == locked.sheet


# --------------------------------------------------------------- reference set helpers


def test_reference_lookup_falls_back_to_the_base_sheet():
    base, bare = ReferenceSet(sheet="cape.png"), ReferenceSet(sheet="bare.png")
    character = Character(
        wardrobe=[
            WardrobeState(id="cape", description="a"),
            WardrobeState(id="bare", description="b"),
        ],
        references={"cape": base},
    )
    assert character.reference_for("cape") is base
    assert character.reference_for("bare") is base  # no sheet of its own yet
    character.references["bare"] = bare
    assert character.reference_for("bare") is bare
    assert character.reference_for(None) is base


def test_states_without_a_sheet_are_listed_but_the_first_costume_is_covered_by_a_base_sheet():
    wardrobe = [
        WardrobeState(id="cape", description="a"),
        WardrobeState(id="bare", description="b"),
    ]
    with_base = Character(wardrobe=wardrobe, references={"base": ReferenceSet(sheet="old.png")})
    assert with_base.states_without_reference({"cape", "bare"}) == ["bare"]
    empty = Character(wardrobe=wardrobe)
    assert empty.states_without_reference({"cape", "bare"}) == ["cape", "bare"]
    assert empty.states_without_reference({"bare"}) == ["bare"]


def test_identity_images_skip_the_profile_panel():
    full = ReferenceSet(
        sheet="s.png", panels={n: f"{n}.png" for n in ("front", "profile", "three_quarter", "face")}
    )
    assert full.identity_images() == ["face.png", "front.png", "three_quarter.png"]
    assert ReferenceSet(sheet="s.png").identity_images() == ["s.png"]
    assert ReferenceSet().identity_images() == []


# ------------------------------------------------------------- identity over several angles


class SequenceFaces:
    """Returns one face per call, from a fixed list; ``None`` entries mean no face was found."""

    def __init__(self, embeddings):
        self._queue = list(embeddings)

    def faces(self, image):
        embedding = self._queue.pop(0)
        return [] if embedding is None else [Face(0.4, np.array(embedding))]


@pytest.fixture
def three_images(tmp_path):
    paths = []
    for i in range(3):
        path = tmp_path / f"p{i}.png"
        Image.new("RGB", (8, 8), "grey").save(path)
        paths.append(path)
    return paths


def test_identity_reference_averages_the_angles(three_images):
    frontal, angled = (1.0, 0.0), (0.6, 0.8)
    single = IdentityReference(SequenceFaces([frontal]), three_images[0])
    averaged = IdentityReference(SequenceFaces([frontal, (0.8, 0.6), angled]), three_images)
    seen_at_an_angle = Face(0.4, np.array(angled))
    assert averaged.similarity(seen_at_an_angle) > single.similarity(seen_at_an_angle)
    assert float(np.linalg.norm(averaged.embedding)) == pytest.approx(1.0)


def test_identity_reference_ignores_panels_with_no_face(three_images):
    reference_ = IdentityReference(SequenceFaces([None, (1.0, 0.0), None]), three_images)
    assert reference_.similarity(Face(0.4, np.array([1.0, 0.0]))) == pytest.approx(1.0)


def test_identity_reference_needs_at_least_one_face(three_images):
    with pytest.raises(ValueError, match="No face found"):
        _ = IdentityReference(SequenceFaces([None, None, None]), three_images).embedding


# ------------------------------------------------------------ generating later costumes


def test_later_sheets_are_edits_of_the_base_sheet(project, staged, tmp_path, sample_png):
    lock(project, staged, tmp_path, "cape")
    fake = FakeComfy(png=sample_png)
    templates = load_templates(project, ["t2i", "still"])
    with comfy_client(project, fake) as comfy:
        paths = reference.generate_candidates(project, staged, comfy, templates, "bare", count=2)
    assert [p.name for p in paths] == ["bare_candidate_01.png", "bare_candidate_02.png"]
    assert fake.upload_files == ["cape_sheet.png"]
    graph = fake.queued[0]
    assert graph["3"]["inputs"]["image"] == "mvdirector/upload_0"
    prompt = graph["1"]["inputs"]["text"]
    assert "Only the wardrobe changes" in prompt and "No cape." in prompt
    # The face is not described again next to a reference image.
    assert "beauty mark" not in prompt
    assert fake.queued[0]["2"]["inputs"]["seed"] != fake.queued[1]["2"]["inputs"]["seed"]


def test_a_later_sheet_needs_the_base_sheet_first(project, staged, sample_png):
    templates = load_templates(project, ["t2i", "still"])
    with (
        comfy_client(project, FakeComfy(png=sample_png)) as comfy,
        pytest.raises(ConfigError, match="Lock the base sheet before 'bare'"),
    ):
        reference.generate_candidates(project, staged, comfy, templates, "bare", count=1)


def test_candidates_for_an_unknown_state_are_refused(project, staged, sample_png):
    templates = load_templates(project, ["t2i", "still"])
    with (
        comfy_client(project, FakeComfy(png=sample_png)) as comfy,
        pytest.raises(ConfigError, match="Unknown wardrobe state"),
    ):
        reference.generate_candidates(project, staged, comfy, templates, "gown", count=1)


# --------------------------------------------------------------------------- set plates


def test_plates_are_generated_for_every_set(project, staged, sample_png):
    fake = FakeComfy(png=sample_png)
    template = load_templates(project, ["t2i"])["t2i"]
    with comfy_client(project, fake) as comfy:
        results = plates.generate_candidates(project, staged, comfy, template, None, count=2)
    assert list(results) == ["Corridor", "The Stage"]
    assert [p.name for p in results["The Stage"]] == [
        "the_stage_candidate_01.png",
        "the_stage_candidate_02.png",
    ]
    prompt = fake.queued[0]["1"]["inputs"]["text"]
    assert "a narrow black corridor" in prompt and "completely empty of people" in prompt
    assert "glossy" in prompt
    assert "people" in fake.queued[0]["1"]["inputs"]["negative"]
    video = project.config.video
    assert (fake.queued[0]["2"]["inputs"]["width"], fake.queued[0]["2"]["inputs"]["height"]) == (
        video.width,
        video.height,
    )
    assert len({g["2"]["inputs"]["seed"] for g in fake.queued}) == 4


def test_plates_can_be_generated_for_one_set(project, staged, sample_png):
    fake = FakeComfy(png=sample_png)
    template = load_templates(project, ["t2i"])["t2i"]
    with comfy_client(project, fake) as comfy:
        results = plates.generate_candidates(project, staged, comfy, template, "the stage", count=1)
    assert list(results) == ["The Stage"]
    with comfy_client(project, fake) as comfy, pytest.raises(ConfigError, match="No set named"):
        plates.generate_candidates(project, staged, comfy, template, "Nowhere", count=1)


def test_locking_a_plate_records_it_on_the_set(project, staged, tmp_path):
    image = tmp_path / "room.jpg"
    Image.new("RGB", (64, 36), "navy").save(image)
    locked = plates.lock_plate(project, staged, image, "The Stage")
    assert locked == project.path("assets/plates/the_stage.png") and locked.is_file()
    assert staged.visual_bible.find_set("The Stage").plate == "assets/plates/the_stage.png"
    assert staged.visual_bible.find_set("Corridor").plate is None


# --------------------------------------------------------------------------- preflight

EXTRA = {"set_image": "set", "reference_face": "face", "reference_profile": "profile",
         "reference_three_quarter": "three_quarter"}  # fmt: skip


def with_extra_bindings(project):
    names = {"t2i": ".png", "still": ".png", "lipsync": ".mp4", "i2v": ".mp4"}
    write_workflows(project.path("workflows"), names, still_extra=EXTRA)


def test_preflight_lists_every_missing_asset(project, staged):
    with_extra_bindings(project)
    templates = load_templates(project, ["t2i", "still"])
    result = check_assets(project, staged, templates)
    text = "\n".join(result.problems)
    assert "No reference sheet is locked" in text
    assert "wardrobe state 'bare'" in text and "mvd reference generate --state bare" in text
    assert "No plate for set 'Corridor'" in text and "No plate for set 'The Stage'" in text
    with pytest.raises(ConfigError, match="--skip-preflight"):
        result.raise_if_blocked()


def test_preflight_passes_once_everything_is_locked(project, staged, tmp_path):
    with_extra_bindings(project)
    lock(project, staged, tmp_path, "cape")
    lock(project, staged, tmp_path, "bare")
    for name in ("Corridor", "The Stage"):
        image = tmp_path / f"{name}.png"
        Image.new("RGB", (64, 36)).save(image)
        plates.lock_plate(project, staged, image, name)
    templates = load_templates(project, ["t2i", "still"])
    result = check_assets(project, staged, templates)
    assert result.problems == [] and result.warnings == []
    result.raise_if_blocked()


def test_plates_are_not_required_when_the_workflow_cannot_take_them(project, staged, tmp_path):
    lock(project, staged, tmp_path, "cape")
    lock(project, staged, tmp_path, "bare")
    templates = load_templates(project, ["t2i", "still"])  # no set_image binding
    assert check_assets(project, staged, templates).problems == []


def test_locked_plates_that_cannot_be_used_produce_a_warning(project, staged, tmp_path):
    image = tmp_path / "room.png"
    Image.new("RGB", (64, 36)).save(image)
    plates.lock_plate(project, staged, image, "Corridor")
    templates = load_templates(project, ["t2i", "still"])
    warnings = check_assets(project, staged, templates).warnings
    assert warnings and "no 'set_image' binding" in warnings[0]


def test_preflight_only_checks_the_selected_shots(project, staged, tmp_path):
    with_extra_bindings(project)
    lock(project, staged, tmp_path, "cape")
    templates = load_templates(project, ["t2i", "still"])
    result = check_assets(project, staged, templates, shot_ids=["s002"])
    text = "\n".join(result.problems)
    assert "No plate for set 'Corridor'" in text
    assert "The Stage" not in text and "'bare'" not in text


def test_preflight_notices_a_missing_file(project, staged, tmp_path):
    lock(project, staged, tmp_path, "cape")
    lock(project, staged, tmp_path, "bare")
    project.path(staged.character.references["bare"].panels["front"]).unlink()
    templates = load_templates(project, ["t2i", "still"])
    assert "missing" in "\n".join(check_assets(project, staged, templates).problems)


# -------------------------------------------------------------------- runner behaviour


def run_still(project, manifest, fake, shot_id):
    comfy = ComfyClient(project.config.comfy, transport=fake.transport(), sleep=lambda _: None)
    gate = ScriptedGate([True])
    gates = QualityGates(image=gate, video=ScriptedGate([True]), face_analyzer=None)
    policy = RetryPolicy(project.config.retry, project.config.seed, llm=None)
    templates = load_templates(project, ["t2i", "still", "lipsync", "i2v"])
    runner = ShotRunner(project, manifest, comfy, gates, policy, templates, RunBudget(50))
    runner.run([AssetKind.STILL], shot_ids=[shot_id])
    return gate


def test_each_shot_is_built_from_the_front_panel_of_its_own_costume(
    project, staged, tmp_path, sample_png
):
    lock(project, staged, tmp_path, "cape")
    lock(project, staged, tmp_path, "bare")
    fake = FakeComfy(png=sample_png)
    run_still(project, staged, fake, "s002")
    run_still(project, staged, fake, "s004")
    assert fake.upload_files == ["cape_front.png", "bare_front.png"]


def test_the_judge_gets_the_full_sheet_of_the_shots_costume(project, staged, tmp_path, sample_png):
    lock(project, staged, tmp_path, "cape")
    lock(project, staged, tmp_path, "bare")
    gate = run_still(project, staged, FakeComfy(png=sample_png), "s004")
    assert gate.kwargs["reference_image"] == project.path("assets/reference/bare_sheet.png")


def test_extra_angles_and_plate_are_sent_only_when_the_workflow_binds_them(
    project, staged, tmp_path, sample_png
):
    lock(project, staged, tmp_path, "cape")
    image = tmp_path / "corridor.png"
    Image.new("RGB", (64, 36)).save(image)
    plates.lock_plate(project, staged, image, "Corridor")
    plain = FakeComfy(png=sample_png)
    run_still(project, staged, plain, "s002")
    assert plain.upload_files == ["cape_front.png"]  # nothing bound beyond the front panel

    with_extra_bindings(project)
    staged.shot("s002").still = AssetState(attempts_allowed=4)  # generate it again
    fake = FakeComfy(png=sample_png)
    run_still(project, staged, fake, "s002")
    assert fake.upload_files == [
        "cape_front.png",
        "cape_profile.png",
        "cape_three_quarter.png",
        "cape_face.png",
        "corridor.png",
    ]
    inputs = fake.queued[0]["3"]["inputs"]
    assert inputs["image"] == "mvdirector/upload_0" and inputs["set"] == "mvdirector/upload_4"
    assert inputs["face"] == "mvdirector/upload_3"


def test_a_plate_replaces_the_text_description_in_the_prompt(project, staged, tmp_path, sample_png):
    lock(project, staged, tmp_path, "cape")
    with_extra_bindings(project)
    image = tmp_path / "corridor.png"
    Image.new("RGB", (64, 36)).save(image)
    plates.lock_plate(project, staged, image, "Corridor")
    run_still(project, staged, FakeComfy(png=sample_png), "s002")
    prompt = staged.shot("s002").still.attempts[0].prompt
    assert "second reference image shows the location" in prompt
    assert "a narrow black corridor" not in prompt
    # A set without a plate keeps its text description.
    lock(project, staged, tmp_path, "bare")
    run_still(project, staged, FakeComfy(png=sample_png), "s004")
    assert "Setting: a large black stage" in staged.shot("s004").still.attempts[0].prompt


def test_a_missing_costume_sheet_falls_back_to_the_base_sheet_with_a_warning(
    project, staged, tmp_path, sample_png, caplog
):
    lock(project, staged, tmp_path, "cape")
    fake = FakeComfy(png=sample_png)
    with caplog.at_level(logging.WARNING, logger="mvdirector.generation.runner"):
        run_still(project, staged, fake, "s004")
    assert fake.upload_files == ["cape_front.png"]
    assert "No reference sheet for wardrobe state 'bare'" in caplog.text


def test_no_reference_at_all_is_an_error_not_a_guess(project, staged, sample_png):
    fake = FakeComfy(png=sample_png)
    with pytest.raises(ConfigError, match="No reference sheet is locked"):
        run_still(project, staged, fake, "s002")


# ------------------------------------------------------------------------- CLI


def test_stills_are_blocked_until_assets_are_locked(project, staged):
    project.save_manifest(staged)
    result = CliRunner().invoke(app, ["stills", "-p", str(project.root)])
    assert result.exit_code == 1
    assert "Assets are not ready for stills" in result.output
    assert "mvd reference lock" in result.output
    assert not project.is_locked  # nothing started


def test_assets_command_shows_what_is_locked(project, staged, tmp_path):
    lock(project, staged, tmp_path, "cape")
    project.save_manifest(staged)
    output = CliRunner().invoke(app, ["assets", "-p", str(project.root)]).output
    assert "cape" in output and "locked, 4 panel(s)" in output
    assert "bare" in output and "MISSING" in output
    assert "Corridor" in output and "not set" in output


def test_lock_commands_update_the_manifest(project, staged, tmp_path):
    project.save_manifest(staged)
    sheet = make_sheet(tmp_path / "sheet.png")
    runner = CliRunner()
    result = runner.invoke(
        app, ["reference", "lock", str(sheet), "--state", "bare", "-p", str(project.root)]
    )
    assert result.exit_code == 0 and "4 panel(s)" in result.output
    room = tmp_path / "room.png"
    Image.new("RGB", (64, 36)).save(room)
    result = runner.invoke(
        app, ["plates", "lock", str(room), "--set", "Corridor", "-p", str(project.root)]
    )
    assert result.exit_code == 0
    saved = project.load_manifest()
    assert "bare" in saved.character.references
    assert saved.visual_bible.find_set("Corridor").plate == "assets/plates/corridor.png"


# ---------------------------------------------------- matching detail to shot size


def spec_for(staged, framing):
    shot = staged.shot("s002")
    shot.framing = framing
    return vision.VisionJudge.specification(shot, staged)


def test_the_judge_is_told_what_can_be_resolved_at_each_framing(staged):
    close = spec_for(staged, Framing.CLOSE_UP)
    wide = spec_for(staged, Framing.WIDE)
    assert "Detail level for a close_up shot" in close and "must be correct" in close
    assert "Detail level for a wide shot" in wide and "do not penalize" in wide
    assert vision.DETAIL_BY_FRAMING.keys() == set(Framing)


def test_no_detail_line_for_shots_without_a_singer(staged):
    shot = staged.shot("s002")
    shot.type = ShotType.BROLL
    assert "Detail level" not in vision.VisionJudge.specification(shot, staged)


def test_the_planner_is_told_to_keep_small_details_for_close_ups():
    assert "Match detail to shot size" in prompts.SHOTS_SYSTEM
    assert "close_up or medium_close" in prompts.SHOTS_SYSTEM


# ---------------------------------------------------- multi-character references and runner


def test_duet_references_preflight_runner_and_vision_spec(project, staged, tmp_path, sample_png):
    staged.characters = {
        "elias": Character(
            name="Elias",
            appearance="Elias, a 22-year-old male.",
            wardrobe=[WardrobeState(id="coat", description="Velvet coat.")],
        ),
        "odalie": Character(
            name="Odalie",
            appearance="Odalie, a 20-year-old female.",
            wardrobe=[WardrobeState(id="gown", description="Silk gown.")],
        ),
    }
    staged.character = staged.characters["elias"]
    s2, s4 = staged.shot("s002"), staged.shot("s004")
    s2.character, s2.characters, s2.wardrobe = "elias", ["elias"], "coat"
    s4.character, s4.characters, s4.wardrobe = "odalie", ["odalie", "elias"], "gown"
    s4.character_wardrobes = {"odalie": "gown", "elias": "coat"}

    templates = load_templates(project, ["t2i", "still"])
    problems = "\n".join(
        check_assets(project, staged, templates, shot_ids=["s002", "s004"]).problems
    )
    assert "--character elias" in problems and "--character odalie" in problems

    reference.lock_reference(
        project, staged, make_sheet(tmp_path / "elias.png"), state_id="coat", character_id="elias"
    )
    reference.lock_reference(
        project, staged, make_sheet(tmp_path / "odalie.png"), state_id="gown", character_id="odalie"
    )
    assert check_assets(project, staged, templates, shot_ids=["s002", "s004"]).problems == []

    fake = FakeComfy(png=sample_png)
    run_still(project, staged, fake, "s002")
    gate = run_still(project, staged, fake, "s004")
    assert fake.upload_files == ["elias_coat_front.png", "odalie_gown_front.png"]
    assert gate.kwargs["reference_image"] == project.path("assets/reference/odalie_gown_sheet.png")
    assert gate.kwargs["extra_references"] == [
        ("Elias", project.path("assets/reference/elias_coat_sheet.png"))
    ]

    with pytest.raises(ConfigError, match="Unknown character"):
        reference.lock_reference(
            project, staged, make_sheet(tmp_path / "bad.png"), state_id="coat", character_id="a"
        )

    spec = vision.VisionJudge.specification(s4, staged)
    assert "Singer (must match exactly): Odalie" in spec
    assert "Additional character in shot (Elias): Elias" in spec
