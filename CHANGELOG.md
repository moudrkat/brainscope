# Changelog

## Unreleased

- `stream: true` on `/v1/chat/completions`: the same generation as
  server-sent events, one chunk per token, OpenAI-shaped. With
  `logprobs: true` each chunk carries the sampled token's log-probability
  and its top five rivals at temperature 1, so a client can draw the
  model's certainty and its roads not taken from the API alone. The viz
  keeps receiving every payload as before.
- `--cors`: answer cross-origin requests, for a web app served from
  elsewhere that points its OpenAI client at this server (the first one is
  brave-new-world, which lets a page dream its worlds through brainscope
  and watch the residual stream over here). Off by default; the API has no
  auth.

## 0.4.3 (2026-09-14)

- On CUDA with torch >= 2.14, the first request no longer dies with
  `fatal error: Python.h: No such file or directory` on a box without
  python3-dev. torch's new Triton-backed native ops compile a C shim on
  first use, and torch reads its `TORCH_DISABLE_NATIVE_JIT` switch at
  import; brainscope now checks for `Python.h` and a C compiler before
  importing torch and, if either is missing, sets the switch (plain aten
  kernels, same numbers) and says so at model load. Docker images were
  never affected.

## 0.4.2 (2026-09-04)

- Emoji and other multi-byte characters no longer show up as `���` in the
  answer text, the traces and the per-token instruments: byte-level tokens
  are held back until the character is complete and the finished character
  rides on the token that completed it. The API response was never affected
  (it decodes the whole id sequence).

## 0.4.1 (2026-09-04)

- The viz websocket now also carries the twelve largest components of the
  residual stream at each layer (with their indices) and each head's attention
  mass on token 0 (the sink). Both were already computed for the lens and the
  per-head entropy; a client can now draw the residual stream and the sink
  without polling `/gen/heads` per layer.
- Model loading passes `dtype` instead of the deprecated `torch_dtype`
  (transformers ≥ 4.56 is now the floor).
- Docs: the README commands are copy-pasteable again (`--model mid` was never
  a preset; `--bake` takes a bake artifact folder; the CPU quickstart needs
  `--lens on`), `stream: true` is documented as ignored rather than
  unsupported, and the agent guide gains the CLI flags, env knobs and the
  read-only `/gen*` / traces API. CI checks the packaged guide matches the
  repo copy.

## 0.4.0 (2026-08-07)

- **`--pace SECONDS`** (cinema mode): slow decoding to an external clock, so a
  generation can be watched token by token or synced to a slower device.
- **ESP32 matchbox LLM example** (`examples/esp32/`): the whole instrument
  driving a tiny model as if it lived on a microcontroller — no hardware
  needed.
- **Parity probe for `syntax_mute`**: `brainscope.parity` now first checks
  whether a backend implements `syntax_mute` at all before comparing outputs
  (hotwire-vllm drops the flag silently, which used to look like a mismatch).
- **V-Steer substring spans**: a hierarchy conflict that lives *inside* one
  message can be marked as a substring span instead of a whole message, and
  the per-conflict figures are generated per conflict.
- **Release pipeline**: pushing a `v*` tag builds, verifies (wheel must carry
  the static UI) and publishes to PyPI via Trusted Publishing; the build fails
  if the tag and `pyproject.toml` disagree about the version.
- README: hierarchy figure rebuilt from Qwen3-4B with a script that
  regenerates it, every instrument section says how to run it, lab map gains
  old-news.

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
