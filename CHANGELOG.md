# Changelog

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
