# Using brainscope (guide for coding agents)

You are helping someone use `brainscope`, an OpenAI-compatible chat server
over any Hugging Face model with a live view into its internals. Run
`brainscope --guide` for this text from an installed copy.

## Start a server

    brainscope --model tiny                     # 0.5B, CPU is fine, for a first try
    brainscope --model Qwen/Qwen2.5-7B-Instruct --quantize 8bit   # fit big models on ~16 GB
    brainscope --model M --directions dirs.json # also load steering directions

Serves on `http://localhost:8010` by default (`--port`, `--host`). A
directions file is JSON `{"name": [n_layers, hidden] matrix, ...}` or a
hidden-directions direction dict.

## Talk to it (it's OpenAI-compatible)

Point any OpenAI client at `http://localhost:8010/v1`. Standard
`POST /v1/chat/completions`. To steer a request, add a `steering` object:

    {"messages": [...], "steering": {"id": "NAME", "layer": 20, "scale": 3,
                                     "decode_only": true}}

`decode_only: true` steers only generated tokens (not the prompt) — use it
unless you have measured the long-context regime, because steering a long
prefill is a much larger dose.

## Look inside

- `POST /replay {"messages": [...], "steering": {...}, "forced": true, "kl": true}`
  — teacher-forced clean-vs-steered diff: per-layer cosine with the direction,
  KL, and (with a fitted lens) which "forming words" the vector suppressed.
- Add `"attn_divergence": true` to the forced diff for the rerouting monitor:
  per-(layer, head) Jensen–Shannon divergence between the clean and steered
  attention patterns, plus the mass change on each head's focus set (the
  tokens carrying 0.8 of clean attention). `"attn_layers": [ints]` picks the
  watched layers (default: injection layer onward, capped at 8). Off by
  default — attention capture costs memory.
- `GET /directions` — list loaded directions. `POST /steer` — set a global
  steering state instead of per-request.
- `POST /hierarchy` — instruction hierarchy (V-Steer, arXiv:2607.26228). Not a
  direction: after prefill it asks each attention head whether it took its cue
  from the system prompt or from a demoted span, and rescales the *cached V*
  of the losers. Use it when a system prompt keeps losing to something a user
  said earlier in the same conversation — the "we changed the rule, the
  transcript didn't" case. The edit lives in the KV cache, so decoding costs
  nothing extra.

      {"stale": [2, 3], "gamma_plus": 2.5, "gamma_minus": 0.75}

  `stale` = message indices that lost authority; `privileged` defaults to the
  system messages. Same object per-request as `"hierarchy"` in
  `/v1/chat/completions`. `GET /hierarchy` returns the spec plus the last
  run's report (heads edited, tokens touched). Defaults are the paper's;
  rules that only show up at the *end* of an answer often need `gamma_plus`
  5-8, and past ~10 the model starts to degrade.
- `POST /probes` — arm cheap activation probes: per-token scalar readouts
  (`h·v̂` or cosine) of one layer's residual against a loaded direction. They
  cost ~nothing and stay alive even with viz off, so they can screen every
  token of real traffic; give a probe a `threshold` and `"trip": "viz"` and
  the full capture turns itself on the moment the probe fires — the
  probe → deep-dive cascade. `GET /probes` shows config + last scores;
  `--probes probes.json` arms them at startup.

      {"probes": [{"direction": "v_pref", "layer": 20, "threshold": 3.0,
                   "trip": "viz"}], "enabled": true}

- `POST /probes/train` — the probe factory: train your OWN probe from two
  contrast personas, no external tooling. The server answers your prompts
  under both system prompts, extracts the diff-of-means direction from the
  answers' activations, reports holdout AUC + a calibrated threshold, and
  (with `arm`) switches the probe on:

      {"name": "slop", "pos_system": "answer in generic AI-slop style…",
       "neg_system": "answer densely and concretely…",
       "prompts": ["…", "…", "…", "…"], "arm": {"ema": 0.15, "trip": "viz"}}

  Takes minutes (it generates live — watch the viz). Check `auc_holdout`
  before trusting the meter: ~0.5 means it learned noise.

- `GET /info` — model + config.

## Gotchas

- Lens-based readouts (`suppressed_positional`) need a fitted J-lens
  (`brainscope-jlens fit ...`); without one they are empty, not zero.
- Thinking models emit a `<think>` block; use `/no_think` or strip it before
  reading disposition-style readouts of the answer.

Make the steering vectors with
[hidden-directions](https://github.com/moudrkat/hidden-directions); take them
to production with [hotwire-vllm](https://github.com/moudrkat/hotwire-vllm).
