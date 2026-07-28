# Changelog

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
