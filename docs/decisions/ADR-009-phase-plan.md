# ADR-009: Phase plan for a refactoring project

Status: Accepted (2026-10-27, set by the human)

## Context
The playbook's default phases (Ingest, Split, EDA, Features, Baseline, Models, Final evaluation, Communication) fit a new modeling project. This project refactors existing code; EDA is already done in the notebook and is out of scope.

## Decision
Phases: 1 Load and clean, 2 Features, 3 Pipeline (parity), 4 Modeling (stretch), 5 FastAPI CRUD (stretch), 6 Docker (stretch), 7 Write-up. There is no EDA phase. The split moves into Phase 4. The evaluator joins Phase 4 for metrics only.

## Consequences
PROJECT_WORKFLOW.md, the agent descriptions and the specs use this numbering. Phases 4 to 6 may be skipped without breaking the required deliverables (Phases 1 to 3 and 7).
