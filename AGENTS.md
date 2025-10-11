# AGENTS Playbook

## 1) Scope & Principles
- Keep tasks ≤ 30 minutes each.
- Prefer small, independent changes with tests.
- Never run two tasks in parallel that modify the same file(s).
- Favor CLI-driven smoke tests (`dnd-transcribe --help`) when touching user flows.
- Preserve existing configuration defaults in `src/dnd_session_transcribe/util/config`; override via parameters or env vars instead of editing defaults.
- Treat the playbook below as the authoritative guidance. Legacy repo notes (see Appendix) remain for context but defer to these rules when conflicts arise.

## 2) Roles
- **Project Manager (PM):** Breaks epics into child tasks, defines dependencies (DAG), starts all zero-dependency tasks in parallel, monitors progress, merges when checks pass.
- **Implementer:** Executes a child task in isolation on its own branch; writes/updates tests; opens a focused PR.
- **Reviewer (Codex):** Runs lint/tests, enforces touch-list & ownership checks, leaves actionable comments.
- **Domain Specialist (optional):** Provides subject-matter context for transcription quality or audio-processing constraints when requested by PM.

## 3) Task Lifecycle
1. PM defines **child tasks** with:
   - Title, Description, Acceptance checks
   - Estimated effort (≤ 30 min)
   - Explicit **Dependencies** (IDs or names)
2. PM **starts all child tasks with no dependencies in parallel**.
3. Implementers:
   - Create branch: `codex/<task-id>-<kebab-name>`
   - Keep changes scoped; avoid shared files unless declared
   - Add/update tests and docs
   - Respect repo norms: use module-level `logger = logging.getLogger(__name__)`, follow import ordering (stdlib → third-party → local), and prefer `pathlib.Path`.
4. Reviewer:
   - Runs CI (lint, unit, integration as applicable)
   - Blocks merges if conflicts or ownership violations exist
   - Verifies CLI usage (`dnd-transcribe --help`) when CLI surface changes.
5. PM merges when all checks are green.

## 4) Parallelization Rules
- Shard work by directory/component whenever possible (e.g., `src/dnd_session_transcribe/features/*` vs. `src/dnd_session_transcribe/util/*`).
- If two tasks would touch any of the same files, **serialize** them or split scope.
- Use a **touch list** before starting: compute intended file changes; if intersection ≠ ∅ with running tasks, queue it.
- Safe to parallelize:
  - Test creation/expansion across modules
  - Read-only audits (docs, inventory, issue triage)
  - Component-isolated refactors (one subdir per task)
- Risky (serialize or shard carefully):
  - Build tooling, top-level configs, cross-cutting renames
  - Changes to `pyproject.toml`, `pytest.ini`, `src/dnd_session_transcribe/util/config.py`, or CLI entry points.

## 5) Branching & PR Policy
- Branch name: `codex/<task-id>-<kebab-name>`
- One logical change per PR; small diffs preferred
- PR must include:
  - Summary, Rationale, Risk
  - **Touch list** (glob or explicit file list)
  - Test plan and results (see pytest commands below)
  - Notes for CLI behavior or configuration adjustments when affected
- Auto-merge only when:
  - CI green (lint+tests)
  - No overlap with concurrently open PRs (verify touch lists)
  - Required reviewers (Codex/owners) approved
  - CLI smoke checks confirm expected flags/help output when relevant

## 6) File-Ownership (optional)
- <!-- TODO: establish CODEOWNERS mapping for critical subpackages (features, adapters, util). -->
- Tasks modifying owned paths must include the owner as reviewer once the mapping exists.

## 7) Safety & Secrets
- Never add secrets/tokens to code or docs.
- Redact or reference via environment variables and secret stores.
- Do not commit proprietary audio samples beyond the provided `sample_audio/` fixtures.

## 8) Observability & Reporting
- PM posts a brief status after fan-out:
  - Active tasks, blocked tasks (with blocker), recently merged
  - Include links to tasks and PRs.
- Capture noteworthy CLI behavior changes or regression risks in status notes.

## 9) Reusable Templates

### 9.1 Child Task Template
**Title:** <prefix>: <concise outcome>
**Description:** <what/why, scope constraints>
**Dependencies:** <none | list IDs>
**Acceptance:**
- [ ] Implementation complete and scoped to <paths>
- [ ] Tests added/updated and passing locally
- [ ] PR opened with touch list + test plan
**Estimate:** ≤30m
**Start Conditions:** Start immediately if no dependencies and no file overlap.

### 9.2 PR Description Template
**Summary**
<one paragraph>

**Touch list**
- <path/glob 1>
- <path/glob 2>

**Test plan (pytest)**
Run locally:
- `pytest -q`
- `pytest --cov --cov-branch --cov-report=term-missing`
Include in PR:
- Short summary of failing/xfail/skipped
- Coverage % headline and notable gaps

**Risk & Rollback**
- Risk: <low/med/high>
- Rollback: revert PR; confirm tests green

**Ownership/Reviewers**
- <@team/owner if applicable>

## Appendix: Repository Reference (Legacy Guidance)
These notes summarize the prior AGENTS guidelines. They are retained for institutional memory and supplemental context; if any instruction conflicts with Sections 1–9 above, follow the playbook rules first.

### Installation & CLI
- Install the project in editable mode for development: `pip install -e .` (add `--extra-index-url` if CUDA wheels are needed).
- Use the CLI entry point `dnd-transcribe` defined in `src/dnd_session_transcribe/cli.py` for smoke tests (e.g., `dnd-transcribe --help`). Avoid legacy scripts like `python run_whisperx.py`.
- Import configuration via `dnd_session_transcribe.util.config`; do not assume a working directory or mutate defaults in place.
- Runtime defaults emphasize GPU-friendly Faster-Whisper settings. When CPU- or memory-constrained, surface overrides through CLI flags instead of editing defaults.

### Code Organization
- `src/dnd_session_transcribe/cli.py` handles argument parsing and top-level pipeline wiring—push heavy logic into feature modules.
- `features/` stores the core stages: `asr.py`, `alignment.py`, `diarization.py`, and `precise_rerun.py`.
- `adapters/` contains external-integration glue (WhisperX, pyannote, UVR). Keep API calls isolated here.
- `util/` centralizes helpers: configuration defaults (`util/config`), path helpers (`next_outdir.py`), preprocessing (`processing.py`), and writers (`write_files.py`).
- Place new or updated tests under `tests/`, mirroring the module they validate.

### Coding Standards
- Target Python 3.10+ with complete type hints and `pathlib.Path` for filesystem interactions.
- Order imports as stdlib → third-party → local; never wrap imports in try/except.
- Define module-level loggers with `logger = logging.getLogger(__name__)` and prefer structured prefixes (e.g., `[ASR]`, `[ALIGN]`).
- Keep CLI help text synchronized with README examples and user workflows.

### Dependencies
- Runtime dependencies belong in `pyproject.toml`; optional extras live under `[project.optional-dependencies]`.
- Document optional import requirements (e.g., UVR) in module docstrings and README entries when gating features.

### Testing & Verification
- Run unit tests with `pytest`, using targeted paths for faster feedback when necessary.
- For CLI changes, execute `dnd-transcribe --help` to confirm arguments and defaults.
- Maintain resume/checkpoint behavior via JSON artifacts when adjusting long-running stages.
- Ensure local `ffmpeg` availability (install via apt) before running audio pipelines.

### Performance Constraints & Model Selection
- Sandbox environment is CPU-only with limited RAM and wall-clock budgets.
- Limit audio clips to ≤ 60 seconds for local verification.
- Whisper defaults to the `tiny` model; upgrade to `base` only when clips are short (<60s) and runtime allows (<90s CPU).
- Avoid large model variants in this environment. Hosted STT models are allowed only if policy permits network use.

### Configuration Knobs
- `WHISPER_MODEL` environment variable selects the local Whisper model (`tiny` or `base`).
- Assume `tiny` if unset. Start with `tiny` and escalate to `base` only when resources permit.

### Documentation & PR Expectations
- Update `README.md` when workflows or CLI flags change to keep user guidance current.
- Preserve docstrings for shared helpers and adapters, especially those wrapping external APIs.
- Summarize pipeline impacts (latency, accuracy, outputs) in PR descriptions alongside required touch lists and test plans.

## Getting Started (Tests)
We use **pytest** with full coverage reporting.
- Quick run: `pytest -q`
- Coverage run: `pytest --cov --cov-branch --cov-report=term-missing`
<!-- TODO: if a top-level package name is required for --cov, replace with: pytest --cov=<PACKAGE_NAME> --cov-branch --cov-report=term-missing -->
