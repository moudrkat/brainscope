# Changelog

## 0.3.0 (2026-08-03)

- **Instruction-hierarchy steering** (`POST /hierarchy`, `GET /hierarchy`, and
  a per-request `"hierarchy"` object in `/v1/chat/completions`): V-Steer,
  arXiv:2607.26228. You ship a new system prompt and the conversation keeps
  obeying the old one; mark the messages that lost authority with
  `{"stale": [2, 3]}` and the attention heads still taking orders from them
  get that span's cached V rescaled. Nothing is added to the residual stream
  and nothing leaves the context. Prefill-only, so decoding runs at normal
  speed.
- **hierarchy tab**: the per-layer split of the last prompt position's
  attention between the privileged and demoted spans, the per-head DLA scores
  with the rescaled KV groups outlined, and a ranked list of the heads that
  actually decided. The tab appears once a generation uses the feature.
- `/demo` gains ready-made conflicts (prefix, casing, bullets, an inline
  `[1] [2] [3]` option list), an "app updated here" divider that marks
  everything above it as pre-update, and live γ+ / γ− controls — so the whole
  thing can be reproduced by clicking rather than by writing code.
- Self-disables with a reason on architectures it cannot support (KV shared
  across layers, sliding-window attention), instead of producing quiet
  nonsense.

## 0.2.2 (2026-07-30)

- **Probe factory** (`POST /probes/train`): train your own probe from two
  contrast personas — the server generates the answers, extracts the
  diff-of-means direction from their activations, reports holdout AUC and
  a calibrated threshold, and optionally arms the probe. No sklearn, no
  external scripts.
- `/demo`: a light-themed stand-in chat app (system prompt + steering
  controls) served same-origin — poke at behaviors without wiring your
  real application. Optional deploy-local `avatar.png` (gitignored).
- `/gen` now returns the probe series; `_persist_directions` demotes a
  read-only dirs.json to a warning instead of a 500.

- **Cheap activation probes** (`POST /probes`, `GET /probes`, `--probes`):
  per-token scalar readouts (`h·v̂` or cosine) of one decoder layer's output
  residual against a loaded direction, computed by a forward hook — so they
  run even in dark mode (viz off) at ~zero cost. Optional signed
  `threshold` marks tokens as `fired`; `"trip": "viz"` turns the full
  capture on the moment a probe first fires (the probe → deep-dive cascade
  production monitors converged on). Score series is streamed on the
  websocket (`"probe"` per token) and persisted in traces (`"probe"` key).

## 0.2.1 (2026-07-28)

- **Forced diff now disables a live global steer** before running: a global
  `/steer` state would otherwise contaminate the clean pass, the baseline,
  and the per-prompt clean-side cache (which is keyed without steering
  state). Re-enable the slider after a replay if you were using it.
- Rerouting monitor: two new per-(layer, head) fields on the forced diff —
  `clean_entropy_mean` (normalized clean-row attention entropy, for
  separating "sharp heads flip easily" from genuine rerouting) and
  `sink_mass_delta` (attention-mass change on position 0, for catching
  sink-attraction artifacts).

## 0.2.0 (2026-07-28)

- **Rerouting monitor** on the forced diff: `/replay` with
  `"attn_divergence": true` returns per-head Jensen–Shannon divergence
  between clean and steered attention rows at matched positions, plus
  the focus-mass delta (how much attention mass leaves the tokens the
  clean pass concentrated on). Opt-in, off by default — the forced diff
  is unchanged without the flag.
- Motivation and field validation: measuring whether a steering vector
  makes heads re-route attention (cf. SKOP, arXiv 2605.06342). First
  live sighting on a production-extracted vector: a single head one
  layer above the injection site rerouting at JSD 0.51 while the
  injection layer's own rows stay at 0.0.
- 53 tests (fixed points, defaults, strong-steering structure).

## 0.1.1

- Initial public release: OpenAI-compatible serving with live residual
  stream view, teacher-forced causal replay (`/replay {forced: true}`),
  direct logit attribution (`/directions/{name}/unembed`), per-token
  cos & J-lens capture.
