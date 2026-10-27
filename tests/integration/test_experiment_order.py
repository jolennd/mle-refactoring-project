"""Phase 4 spec step 1: the baseline lock (ADR-006, R10/R11).

The lock is ``config.BASELINE_LOCK_PATH`` (reports/results/baseline_lock.json), written
once from the frozen notebook procedure and committed alone, before any model or
evaluation code.

Lock schema (this test defines it):
    cv_r2       float, GridSearchCV best_score_ (mean 5-fold R^2), finite, <= 1
    best_params dict, GridSearchCV best_params_; keys = config.PARAM_GRID keys, each
                value one of that key's grid values
    test_r2     float, R^2 of the best estimator on the 30 % test split, finite, <= 1
Extra keys (e.g. provenance notes) are allowed.
"""

import json
import math
import subprocess

import pytest

from kc_housing import config

LOCK_REL = config.BASELINE_LOCK_PATH.relative_to(config.ROOT).as_posix()
GUARDED_FILES = ["src/kc_housing/models.py", "src/kc_housing/evaluate.py"]


def _git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args], cwd=config.ROOT, capture_output=True, text=True, check=False
    )


@pytest.fixture
def git_repo():
    """Skip with a reason outside a git checkout (e.g. a source tarball)."""
    try:
        result = _git("rev-parse", "--is-inside-work-tree")
    except FileNotFoundError:
        pytest.skip("git not installed: commit order cannot be checked")
    if result.returncode != 0 or result.stdout.strip() != "true":
        pytest.skip(f"{config.ROOT} is not a git checkout: cannot check commit order")
    if _git("rev-parse", "--is-shallow-repository").stdout.strip() == "true":
        pytest.skip("shallow clone: full history needed to check commit order")


def _adding_commit(path: str) -> str | None:
    """Oldest commit on HEAD's history that adds `path`, or None."""
    shas = _git("log", "--diff-filter=A", "--format=%H", "--", path).stdout.split()
    return shas[-1] if shas else None


@pytest.mark.integration
def test_lock_precedes_test_eval(git_repo):
    """ADR-006: baseline_lock.json is added in a commit touching only that file, and
    that commit is a strict ancestor of any commit adding models.py or evaluate.py."""
    lock_commit = _adding_commit(LOCK_REL)
    assert lock_commit, f"{LOCK_REL} is not committed (spec step 1 comes first)"

    touched = _git("show", "--name-only", "--format=", lock_commit).stdout.split()
    assert touched == [LOCK_REL], f"lock commit {lock_commit[:7]} touches {touched}"

    for path in GUARDED_FILES:
        code_commit = _adding_commit(path)
        if code_commit is None:  # not committed yet; the lock already is
            continue
        assert code_commit != lock_commit, f"{path} added in the lock commit"
        is_ancestor = _git("merge-base", "--is-ancestor", lock_commit, code_commit)
        assert is_ancestor.returncode == 0, f"{path} committed before {LOCK_REL}"


@pytest.mark.integration
def test_baseline_lock_schema():
    """The lock records cv_r2, best_params (a point of config.PARAM_GRID) and test_r2,
    as finite R^2 values <= 1."""
    assert config.BASELINE_LOCK_PATH.exists(), (
        f"{config.BASELINE_LOCK_PATH} missing: run the frozen notebook (spec step 1)"
    )
    lock = json.loads(config.BASELINE_LOCK_PATH.read_text())

    for key in ("cv_r2", "test_r2"):
        value = lock[key]
        assert isinstance(value, float) and math.isfinite(value) and value <= 1, key
    best = lock["best_params"]
    assert set(best) == set(config.PARAM_GRID)
    for name, grid in config.PARAM_GRID.items():
        assert best[name] in grid, f"{name}={best[name]} not in grid {grid}"
