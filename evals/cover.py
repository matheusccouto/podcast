"""Multimodal cover evaluation dataset, rubrics, and DeepEval case builder."""

from pathlib import Path

from deepeval.dataset import EvaluationDataset, Golden
from deepeval.metrics import BaseMetric, GEval
from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase, MLLMImage, SingleTurnParams

from podcast.cover import headline, render_prompt, to_square_jpeg
from podcast.models import Script

_COVER_GOLDEN_DIR = Path(__file__).parent / "golden" / "cover"
_THUMBNAIL_SIZE = 50

ORTHOGRAPHY_THRESHOLD = 0.8
THUMBNAIL_THRESHOLD = 0.75
BRAND_THRESHOLD = 0.8

_ORTHOGRAPHY_CRITERIA = (
    "Every visible word on the cover is spelled correctly in Brazilian "
    "Portuguese. Diacritics appear wherever standard pt-BR orthography requires "
    "them, including ã, õ, ç, é and ê. Every letter is complete: no letter is "
    "mirrored, duplicated, merged with a neighbour, or replaced by a lookalike "
    "glyph."
)
_ORTHOGRAPHY_STEPS = [
    (
        "Read every visible text element on the cover: the masthead, the Episode "
        "headline, and any secondary label."
    ),
    (
        "List each word and compare its spelling with standard Brazilian Portuguese "
        "orthography, including diacritics."
    ),
    (
        "Inspect each word for malformed or lookalike letters, doubled characters, "
        "and letters merging into their neighbours."
    ),
    (
        "Score 1 when every visible word is correctly spelled and every letter is "
        "complete; score 0 when any word is misspelled, carries a wrong diacritic, "
        "or contains a broken letter."
    ),
]

_THUMBNAIL_CRITERIA = (
    "At 50x50 pixels the cover still reads as a legible thumbnail: the masthead "
    "and the Episode headline stay recognisable as two separate, high-contrast "
    "text blocks, and the central illustration stays one distinguishable shape. "
    "No critical element disappears into the background or gets clipped by the "
    "center safe-zone margin."
)
_THUMBNAIL_STEPS = [
    (
        "Examine the second image, which is the same cover rendered at exactly "
        "50x50 pixels."
    ),
    (
        "Confirm the masthead and the Episode headline remain recognisable as two "
        "separate text blocks with clear contrast against the background."
    ),
    (
        "Confirm the central illustration reads as one distinguishable focal shape "
        "rather than visual noise."
    ),
    (
        "Confirm the masthead, headline and illustration all sit inside the safe "
        "zone without clipping at the edges."
    ),
    (
        "Score 1 only when all three checks hold; score 0 when any critical element "
        "becomes unreadable, merges into another element, or is clipped."
    ),
]

_BRAND_CRITERIA = (
    "The Episode cover belongs to the same publication as the master cover. It "
    "reproduces the master's masthead wording and placement, its palette, and "
    "its heavy editorial typeface, while adding exactly one new Episode-specific "
    "illustration and one new Episode headline."
)
_BRAND_STEPS = [
    (
        "Compare the full-size Episode cover with the master cover reference that "
        "accompanies the test case."
    ),
    "Confirm both carry the same masthead wording in the same position.",
    "Confirm both use the same palette and the same heavy editorial typeface.",
    (
        "Confirm the Episode cover adds one new illustration and one new headline "
        "tied to its lead Story instead of reusing the master artwork."
    ),
    (
        "Score 1 when masthead, palette and typeface match and the new illustration "
        "and headline are present; score 0 when the cover looks like a different "
        "publication or merely repeats the master."
    ),
]


def load_cover_dataset() -> EvaluationDataset:
    """Load evals/golden/cover/*.json Scripts into a DeepEval EvaluationDataset."""
    goldens: list[Golden] = []
    for path in sorted(_COVER_GOLDEN_DIR.glob("*.json")):
        script = Script.model_validate_json(path.read_text(encoding="utf-8"))
        goldens.append(
            Golden(
                name=path.stem,
                input=render_prompt(script),
                additional_metadata={
                    "script": script.model_dump(mode="json"),
                    "headline": headline(script),
                },
            ),
        )
    return EvaluationDataset(goldens=goldens)


def build_cover_metrics(judge: DeepEvalBaseLLM) -> list[BaseMetric]:
    """Build the three multimodal GEval rubrics that gate Episode cover art."""
    return [
        GEval(
            name="Cover Orthography (pt-BR)",
            criteria=_ORTHOGRAPHY_CRITERIA,
            evaluation_steps=_ORTHOGRAPHY_STEPS,
            evaluation_params=[SingleTurnParams.ACTUAL_OUTPUT],
            model=judge,
            threshold=ORTHOGRAPHY_THRESHOLD,
        ),
        GEval(
            name="Cover Thumbnail Legibility (50x50)",
            criteria=_THUMBNAIL_CRITERIA,
            evaluation_steps=_THUMBNAIL_STEPS,
            evaluation_params=[SingleTurnParams.ACTUAL_OUTPUT],
            model=judge,
            threshold=THUMBNAIL_THRESHOLD,
        ),
        GEval(
            name="Cover Brand Consistency",
            criteria=_BRAND_CRITERIA,
            evaluation_steps=_BRAND_STEPS,
            evaluation_params=[
                SingleTurnParams.ACTUAL_OUTPUT,
                SingleTurnParams.EXPECTED_OUTPUT,
            ],
            model=judge,
            threshold=BRAND_THRESHOLD,
        ),
    ]


async def cover_test_case(
    script: Script,
    cover_jpeg: bytes,
    *,
    master_cover_path: Path,
    workdir: Path,
) -> LLMTestCase:
    """Write the full cover plus a 50x50 thumbnail and build the multimodal case."""
    slug = script.published_at.isoformat()
    full_path = workdir / f"{slug}.jpg"
    thumbnail_path = workdir / f"{slug}-50x50.jpg"
    full_path.write_bytes(cover_jpeg)
    thumbnail_path.write_bytes(
        await to_square_jpeg(cover_jpeg, size=_THUMBNAIL_SIZE),
    )
    return LLMTestCase(
        name=slug,
        input=render_prompt(script),
        actual_output=(
            f"Full-size Episode cover: {MLLMImage(url=str(full_path))}\n"
            f"Same cover at 50x50 pixels: {MLLMImage(url=str(thumbnail_path))}"
        ),
        expected_output=(
            f"Master cover reference: {MLLMImage(url=str(master_cover_path))}"
        ),
        metadata={
            "headline": headline(script),
            "lead_story": script.segments[0].title,
        },
    )
