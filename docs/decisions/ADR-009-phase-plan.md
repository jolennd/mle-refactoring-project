# ADR-009: Phase plan for a refactoring project

Status: Accepted (2026-10-27, set by the human)

## Context
The playbook's default phases (Ingest, Split, EDA, Features, Baseline, Models, Final evaluation, Communication) fit a new modeling project. This project refactors existing code; EDA is already done in the notebook and is out of scope.

## Decision
Phases: 1 Load and clean, 2 Features, 3 Pipeline (parity), 4 Modeling (stretch), 5 FastAPI CRUD (stretch), 6 Docker (stretch), 7 Write-up. There is no EDA phase. The split moves into Phase 4. The evaluator joins Phase 4 for metrics only.

## Consequences
PROJECT_WORKFLOW.md, the agent descriptions and the specs use this numbering. Phases 4 to 6 may be skipped without breaking the required deliverables (Phases 1 to 3 and 7).

Amendment (2026-10-28, before Phase 7): execution order changed by the human. Phase 7 (Write-up) runs after Phase 4 and before the stretch Phases 5 (FastAPI CRUD) and 6 (Docker). Phase numbers stay as they are. The README describes Phases 5 and 6 as planned stretch work, with placeholder API and Docker sections to fill in when those phases close. The write-up uses only Phases 1 to 4: model numbers come from `reports/results/` (`baseline_lock.json`, `metrics.json`), and Phase 1 to 3 results that have no file there are cited by their gate test and `config` tolerance, not hard-coded. If Phases 5 and 6 are done later, their README sections are added under those phases' own gates. Human-approved.
