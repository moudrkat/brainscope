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
- `GET /directions` — list loaded directions. `POST /steer` — set a global
  steering state instead of per-request.
- `GET /info` — model + config.

## Gotchas

- Lens-based readouts (`suppressed_positional`) need a fitted J-lens
  (`brainscope-jlens fit ...`); without one they are empty, not zero.
- Thinking models emit a `<think>` block; use `/no_think` or strip it before
  reading disposition-style readouts of the answer.

Make the steering vectors with
[hidden-directions](https://github.com/moudrkat/hidden-directions); take them
to production with [hotwire-vllm](https://github.com/moudrkat/hotwire-vllm).
