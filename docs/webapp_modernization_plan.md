# Modern Web Application Refactor Plan

## Goals
- Provide a responsive, modern web experience for uploading audio and monitoring transcription jobs.
- Preserve the existing Python-based transcription pipeline, including CUDA-enabled dependencies and CLI parity.
- Support incremental rollout so the current CLI and Web UI keep working throughout the refactor.

## Current Architecture Snapshot
- The repository already exposes an end-to-end CLI (`dnd-transcribe`) that orchestrates preprocessing, Faster-Whisper ASR, diarization, and output generation through a single command.【F:README.md†L5-L58】
- A FastAPI app (`dnd-transcribe-web`) wraps the CLI and serves uploads, job status, and result downloads, wiring CLI arguments via `build_cli_args` so the web flow stays aligned with CLI behavior.【F:README.md†L83-L123】【F:src/dnd_session_transcribe/web.py†L1-L91】
- CUDA-enabled wheels are bundled via `requirements.txt` and the README explicitly documents using the PyTorch extra index to install GPU builds of Torch, Torchaudio, and Torchvision.【F:README.md†L28-L45】【F:requirements.txt†L1-L89】

## Refactor Strategy
1. **Backend service layer**
   - Keep FastAPI as the orchestration layer but split the current `web.py` monolith into submodules (e.g., `api.routes`, `services.jobs`, `services.storage`) to improve maintainability and testability while preserving the same CLI invocation path.【F:src/dnd_session_transcribe/web.py†L1-L91】
   - Introduce an asynchronous task runner (Celery, Dramatiq, or a minimal asyncio queue) so long-running transcription jobs execute outside the request lifecycle. This ensures the API remains responsive even when CUDA workloads run for several minutes.
   - Wrap CLI entry points in reusable service functions that accept the same arguments as `build_cli_args`, so both the CLI and web service call a single execution function and stay feature-equivalent.
   - Expose REST endpoints for job submission, status polling, configuration presets, and artifact downloads. Optionally add WebSocket endpoints for live progress updates.

2. **Frontend modernization**
   - Build a single-page application (SPA) using React, Vue, or Svelte with TypeScript for stronger typing and component reuse.
   - Implement upload workflows with drag-and-drop support, client-side validation, and a progress bar backed by the new REST/WebSocket APIs.
   - Provide job dashboards, filtering, and artifact download components that query the backend for metadata (including CUDA/compute details) so power users can confirm GPU-specific settings.
   - Use a design system (e.g., Tailwind, Chakra, or Material UI) to accelerate styling while keeping accessibility (ARIA roles, keyboard navigation) front of mind.

3. **Deployment and CUDA compatibility**
   - Package the backend in Docker images that include the CUDA-enabled requirements and expose environment variables for device selection, mirroring the existing CLI flags (e.g., `--asr-device cuda`).【F:README.md†L59-L82】
   - Document deployment tiers: CPU-only (falling back to alternative compute types) and GPU-enabled (leveraging the existing `+cu121` dependencies). Use container runtime arguments or Kubernetes node selectors to ensure GPU nodes run CUDA workloads.
   - Provide infrastructure scripts (Docker Compose or Helm charts) that start the API, background worker, and frontend assets together, sharing a volume for `webui_runs` so outputs remain accessible.【F:README.md†L105-L123】

4. **Testing and validation**
   - Maintain CLI regression tests to ensure backend refactors do not change transcript outputs for representative fixtures.
   - Add API integration tests that submit jobs, simulate worker completion, and verify artifact availability.
   - Implement frontend end-to-end tests (Playwright/Cypress) covering upload, status polling, and download flows.

## Migration Phases
1. **Foundational cleanup** – Extract service layers from `web.py`, add background worker abstraction, and ensure existing templates continue to render.
2. **API-first release** – Finalize REST/WebSocket endpoints, ship OpenAPI documentation, and update CLI/Web UI docs.
3. **New frontend** – Develop the SPA, integrate with APIs, and release behind a feature flag while keeping the legacy templates available.
4. **Cutover** – Promote the SPA to the default experience once telemetry and user feedback confirm parity.
5. **Enhancement backlog** – Iterate on multi-tenant storage, authentication, analytics dashboards, or GPU utilization monitoring as needed.

## Task Breakdown

### Task 1 – Foundational cleanup
**Title:** backend: extract services and job runner

**Description:**
- Refactor `src/dnd_session_transcribe/web.py` into modular packages for routing, services, and storage integration.
- Introduce an asynchronous job runner abstraction that executes the existing CLI pipeline outside the request lifecycle while preserving CUDA configuration parity.
- Backfill unit tests for the new service layer.

**Dependencies:** none

**Acceptance:**
- [ ] Service modules created under `src/dnd_session_transcribe/web/` with clear separation of routing and business logic.
- [ ] Background worker interface and implementation added with tests covering success and failure paths.
- [ ] Legacy FastAPI endpoints delegate to the new services without breaking current CLI integration.
- [ ] Tests covering the refactored modules pass locally.

**Estimate:** ≤30m

**Start Conditions:** Start immediately.

### Task 2 – API-first release
**Title:** api: ship async endpoints and documentation

**Description:**
- Expand the FastAPI surface to expose REST endpoints for job submission, status polling, configuration presets, and artifact downloads.
- Add WebSocket support (or Server-Sent Events) for streaming job progress updates.
- Publish OpenAPI docs and update README/CLI help to describe the new endpoints.

**Dependencies:** backend: extract services and job runner

**Acceptance:**
- [ ] REST endpoints and WebSocket stream documented and tested via integration tests.
- [ ] OpenAPI schema updated and committed.
- [ ] README and CLI help text reflect new API capabilities.
- [ ] Tests covering API routes and background worker interactions pass locally.

**Estimate:** ≤30m

**Start Conditions:** Start after **backend: extract services and job runner** merges.

### Task 3 – New frontend
**Title:** frontend: implement SPA and integrate APIs

**Description:**
- Scaffold a TypeScript-based SPA (React, Vue, or Svelte) with routing, upload UI, status dashboard, and artifact download views.
- Integrate the new REST/WebSocket APIs for job creation and monitoring.
- Apply a design system to ensure accessible, responsive layouts.

**Dependencies:**
- backend: extract services and job runner
- api: ship async endpoints and documentation

**Acceptance:**
- [ ] SPA codebase lives under a dedicated `frontend/` directory with build tooling.
- [ ] Upload flow with drag-and-drop, validation, and progress reporting wired to backend APIs.
- [ ] Dashboard displays job history and links to transcription artifacts.
- [ ] Automated frontend tests (unit + e2e) cover critical flows and pass locally.

**Estimate:** ≤30m

**Start Conditions:** Start after dependent backend/API tasks are merged.

### Task 4 – Cutover
**Title:** release: enable SPA by default

**Description:**
- Promote the SPA to the default web experience while preserving a toggle for the legacy templates during a grace period.
- Add telemetry hooks and logging to monitor adoption, error rates, and GPU utilization trends post-cutover.
- Document rollback steps in case issues arise.

**Dependencies:**
- frontend: implement SPA and integrate APIs

**Acceptance:**
- [ ] Deployment configuration serves the SPA assets by default with an opt-out flag for the legacy UI.
- [ ] Telemetry dashboards or logs capture key metrics (traffic, job success, GPU usage).
- [ ] Rollback plan documented alongside deployment instructions.
- [ ] Release notes updated to guide users through the transition.

**Estimate:** ≤30m

**Start Conditions:** Start after the SPA is feature-complete and stabilized.

### Task 5 – Enhancement backlog
**Title:** backlog: prioritize advanced capabilities

**Description:**
- Evaluate and prioritize enhancements such as multi-tenant storage, authentication, analytics dashboards, and GPU utilization monitoring.
- Break down high-priority enhancements into new child tasks with clear scope and acceptance criteria.
- Capture user feedback gathered during the cutover phase to inform prioritization.

**Dependencies:** release: enable SPA by default

**Acceptance:**
- [ ] Backlog document created with ranked enhancements and rationale.
- [ ] At least the top two enhancements decomposed into actionable child tasks following the project template.
- [ ] Feedback summary from users/operators attached to the backlog for context.

**Estimate:** ≤30m

**Start Conditions:** Start after the SPA cutover stabilizes and telemetry confirms parity.

By layering these changes, you can deliver a contemporary web experience without rewriting the proven CUDA-aware transcription core. The backend retains the Python ecosystem that already manages GPU execution, while the frontend gains the ergonomics of a modern SPA.
