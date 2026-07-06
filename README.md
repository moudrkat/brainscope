# brainscope

**Watch your model think while your app talks to it.**

An OpenAI-compatible chat server over any Hugging Face causal LM that streams
per-token, per-layer residual-stream activity to a live browser visualization -
No changes to your app: point its OpenAI `base_url` at brainscope and open
the viz in a window next to it. Live activation steering is wired in as an
early work-in-progress - see below.

> Built at [Lifeheck](https://www.lifeheck.com/) while evaluating local models for a
> Czech agentic assistant - thanks to the whole team for the playground. 💛

![brainscope demo](docs/demo.gif)

## Quickstart

```bash
git clone https://github.com/moudrkat/brainscope && cd brainscope
pip install -e .                     # needs Python 3.11+
brainscope --model tiny              # 0.5B, runs on CPU - good first try
# → your app:  http://<host>:8010/v1   (chat completions, incl. tool calls)
# → your eyes: http://<host>:8010      (opens automatically)
```

No app handy? The viz page has a built-in chat box - type and watch.

`--model` takes any Hugging Face model id, plus a few presets: `tiny`
(Qwen2.5-0.5B, CPU-friendly), `qwen3-4b`, `qwen3-8b`, `qwen3.5-9b`,
`gemma-e4b`. Bigger models on a 16 GB card:

```bash
brainscope --model qwen3.5-9b --quantize 8bit
```

Pointing your app at it is one line - wherever it builds its OpenAI client:

```python
client = OpenAI(base_url="http://localhost:8010/v1", api_key="unused")
```

## What am I looking at?

Left: a live cross-section of the model - the prompt enters at the bottom
(embedding), the spine of decoder layers lights up as each token passes
through, the next word exits at the top (lm_head). Right: the same spine
unrolled over time - one column per generated token, one row per layer, color
= how loudly that layer works on that token relative to its own average.
Hover for details, ● record exports a WebM, PNG saves a snapshot.

## Steering (WIP)

Extract a direction from contrast pairs (ActAdd/repeng style), load it, and
drive it live with the slider in the header - activation addition on real
traffic:

```bash
python -m brainscope.extract --model qwen3-4b \
    --pairs examples/czech_pairs.jsonl --layer 18 --name czech
brainscope --model qwen3-4b --directions dirs.json
```

The slider adds `strength × direction` to the residual stream of the chosen
layers on every forward pass - positive pushes toward the "positive" side of
your pairs, negative away. Watch the spine change color as you drag.

This part is a work in progress. The bundled extractor is the naive
mean-difference - it nails strong directions (language switching works
beautifully) but subtle styles need more pairs and better methods (PCA over
diffs, per-layer vectors); proper extraction tooling is coming. Meanwhile,
bring your own vectors: anything shaped `{"name": [hidden_size floats]}` loads.

## Will it work with my app?

Works when your app:

- talks the **OpenAI chat-completions API** (any language or framework),
- uses **non-streaming** responses (`stream: true` is not supported yet),
- calls tools in **hermes/qwen**, **gemma fenced**, or plain-JSON format
  (covers most open models' chat templates; PRs welcome for more).

Honest limitations: generation runs on plain `transformers` - expect tens of
tokens per second, not vLLM speeds; requests are served one at a time (a lock
guards the GPU); no auth (put it behind a tunnel/VPN if exposed); context is
bounded by your VRAM. brainscope is a lab instrument, not a production
server - run it next to production, not instead of it.

## Why not vLLM?

vLLM is a black box by design - fused kernels, paged memory, CUDA graphs;
per-layer states are consumed the moment they're produced. `transformers`
exposes `output_hidden_states` for every architecture with one flag. That's
the trade: brainscope is slower, but it sees everything, for any model.

## License

MIT © Kateřina Fajmanová
