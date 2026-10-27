"""R9: the refactored notebook stays thin (ADR-001, Phase 3 spec step 4, convention 2).

Static checks on the notebook JSON (stdlib json, no kernel): fast, no data needed.
The notebook must import the pipeline from kc_housing, contain none of the reference
notebook's cleaning/feature code, and keep its modeling cells unchanged.
"""

import ast
import hashlib
import json
import re
from pathlib import Path

import pytest

from kc_housing import config

# Proposed config constant (config.py is human-owned): REFACTORED_NOTEBOOK.
REFACTORED_NOTEBOOK = getattr(
    config,
    "REFACTORED_NOTEBOOK",
    config.ROOT / "notebooks" / "king_county_refactored.ipynb",
)

# King-County.ipynb cell indices (all cells, 0-based, as nbformat counts them).
CLEANING_CELLS = range(7, 81)  # load (7), cleaning (26-43), features (56-70), X (78-80)
MODELING_START_CELL = (
    81  # `y = kc_data.price`, then split, models, error analysis, save
)

# SHA-256 of the reference notebook's code-cell sources, joined by CELL_SEPARATOR,
# taken at commit dcb4be8 (the only commit touching King-County.ipynb).
CELL_SEPARATOR = "\n# ---- cell ----\n"
REFERENCE_CODE_SHA256 = (
    "224d17f7ecc0588adacfe746ea436bba6f8773adbc7d082fa6895b8265f90471"
)

# Markers of the notebook's cleaning/feature logic. Each one occurs in the reference
# cells 7-80 and in none of its modeling cells (checked by the test itself).
CLEANING_MARKERS = {
    # cell 7: the raw CSV is loaded by kc_housing.cleaning.load_raw with config paths.
    "read_raw_csv": re.compile(r"\bread_csv\("),
    # cell 28: hard-coded outlier label; the rule lives in clean() (ADR-003).
    "drop_outlier_label": re.compile(r"\.drop\(\s*15856"),
    # cells 35, 38: imputation of view/waterfront.
    "fillna": re.compile(r"\.fillna\("),
    # cells 41, 43: only the cleaning step reads yr_renovated.
    "yr_renovated": re.compile(r"yr_renovated"),
    # cell 41: the last_known_change loop.
    "last_known_change_loop": re.compile(r"last_known_change\.append"),
    # cells 30, 42, 60, 70: writing a cleaned/derived column into a frame.
    "derived_column_assignment": re.compile(
        r"\[\s*[\"'](sqft_basement|last_known_change|center_distance|"
        r"water_distance|delta_lat|delta_long)[\"']\s*\]\s*=(?!=)"
    ),
    # cells 60, 67: distance maths (np.radians is not used by any modeling cell).
    "radians": re.compile(r"\bradians\("),
    # cell 60: city-center constants, which belong in config.
    "center_constants": re.compile(r"47\.62774|122\.24194|47\.6219"),
    # cell 67: the distance helper.
    "def_dist": re.compile(r"\bdef\s+dist\("),
    # cells 68-69: waterfront reference list and nearest-water loop.
    "water_list": re.compile(r"\bwater_list\b"),
}

PIPELINE_ENTRY_POINTS = {"build_feature_pipeline", "run"}


def _code_cells(path: Path) -> list[tuple[int, str]]:
    """(cell index, source) for every code cell of the notebook at path."""
    if not path.exists():
        pytest.fail(f"notebook not found: {path}")
    notebook = json.loads(path.read_text(encoding="utf-8"))
    cells = []
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            source = cell["source"]
            cells.append(
                (index, source if isinstance(source, str) else "".join(source))
            )
    return cells


def _without_magics(source: str) -> str:
    """Drop IPython magic and shell lines so the rest parses as Python."""
    return "\n".join(
        line for line in source.splitlines() if not line.lstrip().startswith(("%", "!"))
    )


def _imports_pipeline(source: str) -> bool:
    """True if source imports kc_housing.pipeline or one of its entry points."""
    try:
        tree = ast.parse(_without_magics(source))
    except SyntaxError:
        return False
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(alias.name == "kc_housing.pipeline" for alias in node.names):
                return True
        elif isinstance(node, ast.ImportFrom):
            names = {alias.name for alias in node.names}
            if node.module == "kc_housing.pipeline" and names & PIPELINE_ENTRY_POINTS:
                return True
            if node.module == "kc_housing" and "pipeline" in names:
                return True
    return False


def test_refactored_notebook_imports_pipeline():
    """Convention 2: some code cell imports the Phase 3 entry point from
    kc_housing.pipeline (build_feature_pipeline or run), not cleaning/features alone."""
    cells = _code_cells(REFACTORED_NOTEBOOK)

    importing = [index for index, source in cells if _imports_pipeline(source)]

    assert importing, (
        "no code cell imports kc_housing.pipeline "
        f"({' / '.join(sorted(PIPELINE_ENTRY_POINTS))})"
    )


@pytest.mark.parametrize("marker", CLEANING_MARKERS, ids=list(CLEANING_MARKERS))
def test_refactored_notebook_has_no_cleaning_code(marker):
    """Convention 2: no code cell contains the reference notebook's cleaning or
    feature logic (cells 7-80); that logic lives only in src/kc_housing/."""
    pattern = CLEANING_MARKERS[marker]
    reference = _code_cells(config.REFERENCE_NOTEBOOK)
    assert any(pattern.search(s) for i, s in reference if i in CLEANING_CELLS), (
        f"marker {marker!r} does not occur in the reference cleaning cells"
    )
    assert not any(
        pattern.search(s) for i, s in reference if i >= MODELING_START_CELL
    ), f"marker {marker!r} would flag the reference modeling cells"

    offending = [i for i, s in _code_cells(REFACTORED_NOTEBOOK) if pattern.search(s)]

    assert not offending, f"cleaning marker {marker!r} found in code cells {offending}"


def test_refactored_notebook_keeps_modeling_cells_unchanged():
    """ADR-001: every reference code cell from cell 81 on appears verbatim (whitespace
    at the ends ignored) and in the same order in the refactored notebook."""
    modeling = [
        s.strip()
        for i, s in _code_cells(config.REFERENCE_NOTEBOOK)
        if i >= MODELING_START_CELL
    ]
    refactored = [s.strip() for _, s in _code_cells(REFACTORED_NOTEBOOK)]

    remaining = iter(refactored)
    missing = [cell for cell in modeling if cell not in remaining]

    assert not missing, (
        f"{len(missing)} modeling cell(s) missing or out of order; first:\n{missing[0]}"
    )


def test_reference_notebook_code_is_unchanged():
    """ADR-001: King-County.ipynb is frozen; its code cells hash to the known value."""
    sources = [s for _, s in _code_cells(config.REFERENCE_NOTEBOOK)]

    digest = hashlib.sha256(CELL_SEPARATOR.join(sources).encode("utf-8")).hexdigest()

    assert digest == REFERENCE_CODE_SHA256
