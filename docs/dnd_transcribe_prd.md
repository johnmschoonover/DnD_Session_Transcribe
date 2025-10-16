# DnD Session Transcribe Product Requirements Document

## 1. Product Overview
DnD Session Transcribe delivers an end-to-end workflow that converts long-form tabletop roleplaying recordings into structured, speaker-attributed transcripts by chaining preprocessing, Faster-Whisper ASR, alignment, and diarization behind a single CLI and optional web interface.【F:README.md†L5-L123】 The product targets hobby groups, streamers, and actual-play producers who need reliable transcripts for accessibility, editing, and content repurposing.

## 2. Problem Statement
Tabletop sessions routinely span multiple hours, involve overlapping speakers, and require diarization to preserve narrative clarity. Manual transcription is time-consuming and error-prone, while general-purpose ASR tools often lack diarization or require fragmented workflows. Users need a turnkey pipeline tuned for long-form, multi-speaker RPG content with minimal setup overhead.

## 3. Goals and Success Metrics
- **Accessible transcripts:** Generate accurate, speaker-tagged transcripts and subtitle artifacts from multi-hour sessions without manual stitching. Success metric: ≥95% session coverage exported as `.srt`, `.vtt`, `.txt`, and `.json` artifacts per run.【F:README.md†L124-L140】
- **Operational efficiency:** Provide a single-command CLI and an optional web UI so hobbyists with limited technical expertise can launch jobs quickly. Success metric: Median setup time ≤10 minutes for new users following documented steps.【F:README.md†L13-L123】
- **Configurability without complexity:** Expose tuning knobs (models, devices, prompts, resume behavior) that advanced users can adopt without overwhelming defaults. Success metric: ≥80% of surveyed beta users rate configuration options as “clear” or “very clear”.【F:README.md†L59-L123】
- **GPU-aware performance:** Maintain CUDA-friendly defaults while allowing CPU fallbacks for resource-constrained hosts. Success metric: Documentation includes tested CPU override workflow and GPU installation guidance.【F:README.md†L28-L82】

## 4. Target Personas
1. **Game Master Recorder:** Runs monthly sessions, exports audio, and needs searchable transcripts for recap prep and lore consistency. Prefers the CLI with resume checkpoints to recover from interruptions.
2. **Actual-Play Producer:** Publishes edited video/audio episodes and requires subtitle files and diarization to align video cuts. Relies on GPU-enabled workflows and web UI job monitoring to coordinate post-production.
3. **Community Accessibility Advocate:** Volunteers to produce captions for community streams, often using CPU-only laptops. Requires clear installation instructions and preview tooling to sample segments before full runs.【F:README.md†L90-L123】

## 5. User Stories & Use Cases
- **US1 – Rapid CLI transcription:** As a Game Master, I run `dnd-transcribe session.wav` after exporting my weekly recording and receive timestamped transcripts plus subtitle formats to share with my party.【F:README.md†L59-L82】【F:README.md†L124-L140】
- **US2 – GPU batch processing:** As an Actual-Play Producer, I queue multiple jobs with CUDA defaults, leveraging vocal extraction and precise reruns to improve clarity before editing.【F:README.md†L13-L82】
- **US3 – Accessibility preview:** As a Community Advocate, I generate preview snippets to validate diarization and speaker tags before committing hours of compute time.【F:README.md†L98-L123】
- **US4 – Web UI monitoring:** As a Producer, I upload audio via the web dashboard, monitor progress, and download transcripts and logs from the same interface for collaboration.【F:README.md†L83-L140】

## 6. Scope
### In Scope
- CLI execution pipeline covering preprocessing, ASR, optional precise rerun, alignment, diarization, and artifact writing.【F:README.md†L13-L140】
- FastAPI-based web interface for job submission, monitoring, and artifact retrieval, including environment configuration for deployment.【F:README.md†L83-L140】
- Documentation covering installation, GPU/CPU configuration, environment variables, preview workflows, and troubleshooting.【F:README.md†L24-L140】

### Out of Scope
- Real-time streaming transcription (focus on batch processing of recorded sessions).
- Native mobile applications (web UI remains the remote-access option).
- Automated content editing or summarization beyond the core transcription outputs.

## 7. Functional Requirements
1. **End-to-end transcription command:** Provide a CLI command (`dnd-transcribe <audio>`) that orchestrates all pipeline stages, handles optional preprocessing, and writes outputs in multiple formats.【F:README.md†L13-L140】
2. **Configurable preprocessing:** Support RAM-disk copying, bandpass vocal extraction, and resume checkpoints via CLI flags.
3. **Model selection:** Allow users to override ASR model, device, and compute precision at runtime, with defaults optimized for CUDA and documented CPU fallbacks.【F:README.md†L59-L123】
4. **Precise rerun capability:** Enable a secondary ASR pass on difficult segments when requested, with separate configuration controls.【F:README.md†L13-L82】
5. **Speaker diarization:** Integrate pyannote diarization requiring a Hugging Face token, surfacing clear errors when credentials are missing.【F:README.md†L41-L123】
6. **Preview generation:** Offer snippet rendering with independent output directories to verify quality before full transcription.【F:README.md†L98-L123】
7. **Web UI parity:** Expose equivalent functionality through a FastAPI-powered interface with configurable storage location and network settings.【F:README.md†L83-L140】
8. **Artifact management:** Persist transcripts, subtitles, previews, and logs in predictable output directories for automation workflows.【F:README.md†L124-L140】

## 8. Non-Functional Requirements
- **Performance:** Support multi-hour session processing within reasonable time on GPU hosts; provide guidance for CPU throughput expectations.【F:README.md†L59-L123】
- **Reliability:** Resume runs from checkpoints, log errors, and fail gracefully when dependencies (e.g., Hugging Face token) are missing.【F:README.md†L41-L140】
- **Usability:** Maintain coherent CLI help text, consistent documentation, and intuitive web UI navigation.【F:README.md†L13-L123】
- **Security:** Do not store secrets in repositories; require users to supply environment variables for tokens and credentials.【F:README.md†L41-L123】
- **Portability:** Ensure instructions cover virtual environments and dependency installation for both CPU and CUDA deployments.【F:README.md†L24-L82】

## 9. Dependencies & Constraints
- Requires FFmpeg availability on host systems for audio preprocessing.【F:README.md†L46-L58】
- Pyannote diarization depends on a valid Hugging Face token exported in the environment.【F:README.md†L41-L58】
- GPU acceleration assumes access to CUDA-enabled hardware with compatible Torch wheels; CPU fallback reduces throughput but must remain viable.【F:README.md†L24-L82】
- Preview export and resume checkpoints rely on filesystem write access; deployments must provision adequate storage.

## 10. Metrics & Telemetry
- Track job duration, success/failure counts, and error categories in CLI logs and web UI dashboards.
- Collect optional anonymous telemetry on configuration choices (model, compute type) to guide documentation priorities (opt-in only).
- Monitor GPU utilization and storage consumption in production deployments where applicable.

## 11. Release Phases
1. **Alpha (internal):** Validate CLI pipeline stability on representative session audio; confirm CUDA and CPU paths operate as documented.
2. **Beta (community pilots):** Invite select game masters and producers to test CLI and web UI, gathering feedback on configuration clarity and diarization accuracy.
3. **General Availability:** Publish installation guide, troubleshooting tips, and finalize documentation; ensure PRD requirements meet success metrics and telemetry thresholds.

## 12. Risks & Mitigations
- **Risk:** Diarization quality varies with audio conditions. *Mitigation:* Provide guidance on speaker count overrides and preprocessing adjustments.【F:README.md†L41-L123】
- **Risk:** GPU dependency complexity discourages adoption. *Mitigation:* Document explicit installation commands for CUDA wheels and CPU fallbacks.【F:README.md†L24-L82】
- **Risk:** Long job runtimes can fail mid-process. *Mitigation:* Encourage use of resume checkpoints and log review via CLI/web UI outputs.【F:README.md†L59-L140】

## 13. Open Questions
- Should the product integrate automated summarization or tagging post-transcription?
- What authentication or multi-tenant requirements emerge for hosted web deployments?
- How can we best support localization of transcripts beyond English (e.g., translation workflows)?

