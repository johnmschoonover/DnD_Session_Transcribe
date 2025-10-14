# ASR Preflight & Auto-Tune

The preflight pipeline inspects each audio file prior to ASR, derives quality metrics,
and recommends (or applies) decoding and VAD settings that reduce looping while
preserving accuracy. It also reports when a lightweight FFmpeg normalisation pass would
help stabilise decoding.

## Enabling from the CLI

```bash
# Apply suggestions automatically (default mode)
dnd-transcribe input.wav --auto-tune

# Inspect suggestions without mutating runtime configuration
dnd-transcribe input.wav --auto-tune --auto-tune-mode suggest

# Force pre-normalisation off or on
# (defaults to `suggest`, which only reports when normalisation is recommended)
dnd-transcribe input.wav --auto-tune --pre-norm apply
```

### Additional Flags

| Flag | Description |
| ---- | ----------- |
| `--autotune-cache-ttl <seconds>` | Control the cache TTL for analysis results (default 86400s). |
| `--autotune-dump <dir>` | Directory for diagnostics, suggested config, and final config JSON artifacts (defaults to the output directory). |
| `--no-cache` | Disable the analysis cache for the current run. |
| `--log-preflight` | Emit a JSON summary of preflight metrics and the applied config. |
| `--redact-paths` | Redact file paths from diagnostics artifacts and log summaries. |

## Python API

```python
from dnd_session_transcribe.helpers import preflight_analyze_and_suggest

cfg, diag = preflight_analyze_and_suggest(
    "input.wav",
    mode="apply",            # or "suggest"
    pre_norm_mode="suggest", # "off" | "suggest" | "apply"
)
```

* `cfg` contains the runtime overrides that should be merged into `ASRConfig` when
  `mode="apply"`.
* `diag` contains:
  * `metrics`: computed audio statistics (SNR, speech gaps, etc.).
  * `suggestion`: the raw auto-tune recommendation plus rationale metadata.
  * `final_config`: the effective configuration after precedence rules.
  * `pre_norm`: the resolved pre-normalisation action and any generated path.
  * `cache`: indicates whether the diagnostics were loaded from cache.

## Generated Artifacts

When `--auto-tune` is enabled, three JSON files are written to the dump directory:

1. `<audio>.preflight.json` — metrics, rationale, and final configuration snapshot.
2. `<audio>.autotune.suggested.json` — the raw suggestions produced by the heuristics.
3. `<audio>.autotune.final.json` — the final config after applying user overrides and
   precedence rules.

Artifacts respect `--redact-paths` and store paths as `null` when redaction is active.

## Caching

Diagnostics are cached using a SHA1 hash of the audio bytes, the auto-tune version,
and the current TTL. Use `--no-cache` to bypass the cache for one-off re-analysis, or
lower the TTL when working on rapidly changing fixtures.

## Troubleshooting

* **Suggestions clipped** — warnings in the log indicate which fields were skipped
  because a user override (CLI, config, or environment) already set the value.
* **FFmpeg unavailable** — the preflight report downgrades pre-normalisation to
  `missing_ffmpeg`; install FFmpeg and rerun with `--pre-norm apply` to generate the
  normalised file automatically.
* **Still seeing decoder loops** — raise `vad_min_silence_ms` or force sampling via
  CLI overrides (`--beam-size none --temperature 0.2,0.4,0.6`) and rerun. The
  diagnostics JSON includes speech/gap quantiles to guide the adjustments.
