"""Phase 7 (Write-up): README.md content contract.

Spec 07_write_up.md and the ADR-009 amendment.

The README answers the brief's three questions, takes every model number from
reports/results/, cites Phase 1-3 results by gate test and config constant (no
hard-coded counts), and describes the API (Phase 5) and Docker (Phase 6) as planned.
"""

import json
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
README_PATH = REPO_ROOT / "README.md"
RESULTS_DIR = REPO_ROOT / "reports" / "results"

BRIEF_QUESTIONS = [
    "What steps did you take to complete the project?",
    "What challenges did you face?",
    "What would you do differently if you had more time?",
]

_HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")


def _readme_text() -> str:
    return README_PATH.read_text(encoding="utf-8")


def _sections(text: str) -> list[tuple[int, str, str]]:
    """Split markdown into (level, heading, body); body runs to the next heading of
    the same or higher level, so it includes subsections. Fenced code is not parsed
    for headings."""
    lines = text.splitlines()
    headings = []  # (line index, level, title)
    in_fence = False
    for i, line in enumerate(lines):
        if line.lstrip().startswith(("```", "~~~")):
            in_fence = not in_fence
            continue
        match = None if in_fence else _HEADING.match(line)
        if match:
            headings.append((i, len(match.group(1)), match.group(2).strip()))
    sections = []
    for k, (start, level, title) in enumerate(headings):
        end = len(lines)
        for later_start, later_level, _ in headings[k + 1 :]:
            if later_level <= level:
                end = later_start
                break
        sections.append((level, title, "\n".join(lines[start + 1 : end])))
    return sections


def _matching_sections(pattern: str) -> list[tuple[str, str]]:
    regex = re.compile(pattern, re.IGNORECASE)
    return [
        (title, body)
        for _, title, body in _sections(_readme_text())
        if regex.search(title)
    ]


def test_readme_answers_brief_questions():
    """The brief's three questions are non-empty headings; template text is gone."""
    text = _readme_text()
    assert "Use this repository as a **template**" not in text, (
        "README.md still contains the course template introduction"
    )
    by_title = {title.casefold(): body for _, title, body in _sections(text)}
    for question in BRIEF_QUESTIONS:
        assert question.casefold() in by_title, f"missing README heading: {question!r}"
        assert by_title[question.casefold()].strip(), (
            f"empty README section: {question!r}"
        )


def test_readme_model_numbers_match_results():
    """Results shows lock and variant A test R^2 from reports/results/ (4 dp)
    and names both source files."""
    lock = json.loads((RESULTS_DIR / "baseline_lock.json").read_text(encoding="utf-8"))
    metrics = json.loads((RESULTS_DIR / "metrics.json").read_text(encoding="utf-8"))
    results = _matching_sections(r"\bresults\b")
    assert results, "README has no Results section heading"
    body = "\n".join(b for _, b in results)
    for label, value in [
        ("baseline_lock.json test_r2", lock["test_r2"]),
        ("metrics.json variant_a.test_r2", metrics["variant_a"]["test_r2"]),
    ]:
        assert f"{value:.4f}" in body, f"Results section lacks {label} = {value:.4f}"
    for source in ("baseline_lock.json", "metrics.json"):
        assert source in body, f"Results section does not name its source file {source}"


@pytest.mark.parametrize(
    "citation",
    [
        "tests/data_gates/test_parity.py",
        "config.PARITY_RTOL",
        "ADR-003",
        "tests/unit/test_cleaning.py",
    ],
    ids=[
        "parity-gate-test",
        "parity-tolerance-constant",
        "drop-rule-adr",
        "drop-rule-test",
    ],
)
def test_readme_cites_parity_gates(citation):
    """Phase 1-3 results cite gate test path and config constant, not counts."""
    assert citation in _readme_text(), f"README does not cite {citation}"


@pytest.mark.parametrize(
    ("heading_pattern", "run_command"),
    [(r"api\b", r"make\s+api\b"), (r"docker", r"make\s+docker-up\b")],
    ids=["phase5-api", "phase6-docker"],
)
def test_readme_stretch_phases_planned(heading_pattern, run_command):
    """API and Docker sections exist, say 'planned', and give no run command yet."""
    sections = _matching_sections(heading_pattern)
    assert sections, f"README has no heading matching {heading_pattern!r}"
    for title, body in sections:
        assert "planned" in body.casefold(), f"section {title!r} does not say 'planned'"
        assert not re.search(run_command, body), (
            f"section {title!r} documents a run command for unbuilt stretch work"
        )
