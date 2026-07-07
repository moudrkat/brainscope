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

Left: the model itself - the prompt enters at the bottom (embedding), one
**clickable row per decoder layer** (attn | mlp | residual cells glowing as
the token passes through, plus what that layer would emit right now), and the
next word exits at the top (lm_head). Click a layer to drill in. On the
right, four instruments:

- **activity over time** - the spine unrolled: one column per generated
  token, one row per layer, color = how loudly that layer works on that
  token relative to its own average.
- **attention** - for the clicked layer: rows = answer tokens, columns =
  every position it looks back at, dashed line = the prompt/answer boundary.
  **heads** splits the newest token per attention head.
- **logit lens** (click lm_head) - every layer's next-token readout: watch
  the answer crystallize with depth. Hover a cell for the top-5 candidates
  with probabilities; click to pin the tooltip for screenshots.
- **the answer text is an instrument too** - each word is tinted by the
  layer where it was decided (clean = early, amber = late, red = never
  settled before lm_head), hovering shows what the model almost said
  instead, and clicking a word lists the prompt positions that fed it.

The ◉ scope button pauses all capture when you just want fast generation;
● record exports a WebM of the activity view, PNG saves a snapshot.

An example of what the lens view can catch:

![Logit lens: the meaning decodes mid-stack in English and Chinese; the Czech surface form assembles only in the last few layers](docs/img/lens-concept-before-language.png)

*Qwen3-4B writing the Czech word "zážitkům" (experiences), fragment by
fragment. Mid-stack readouts decode as the meaning — English and Chinese —
while the Czech surface form assembles only in the last few layers. No
anthropomorphism required: it's the geometry of multilingual representations,
studied properly in Wendler et al. 2024 (arXiv:2402.10588). Readouts are a
raw logit lens, so mid-stack tokens are approximate.*

## Steering

Extract a direction from contrast pairs, load it, and drive it live from
the header - activation addition (Turner et al., arXiv:2308.10248) on real
traffic, with representation-engineering-style extraction (Zou et al.,
arXiv:2310.01405; Rimsky et al., arXiv:2312.06681 - see
[Standing on shoulders](#standing-on-shoulders)). Two extractors ship with
brainscope:

- `brainscope.extract` - quick mean-difference at one layer you pick. Fine
  for strong directions (language switching works beautifully).
- `brainscope.hidden_directions` - the serious one: it reads the hidden
  states of the *completion* tokens for a positive and a negative
  continuation of the same prompt, takes the first principal component of
  the per-pair differences at every layer, and scores each layer by how
  cleanly the direction separates the two sides - so you learn *where* in
  the model the behaviour lives instead of guessing:

```bash
python -m brainscope.hidden_directions --model qwen3-4b \
    --pairs pairs.jsonl --name no-smalltalk --out dirs.json
# prints a per-layer score table and the suggested steering layer range
brainscope --model qwen3-4b --directions dirs.json
```

`pairs.jsonl` lines look like `{"prompt": ..., "positive": ..., "negative":
..., "system": ...}` - the same prompt with two continuations that differ in
exactly the behaviour you want.

In the viz header pick a direction, drag the strength slider and set the
steered layer range (use the extractor's table); the spine changes color as
you drag. Or script it:

```bash
curl -X POST localhost:8010/steer \
    -d '{"name": "no-smalltalk", "strength": 8, "layer_from": 16, "layer_to": 18}'
```

The vector library is just `dirs.json` next to the server - and it is
manageable over HTTP: `GET /directions`, `POST /directions` with `{"name",
"vector"}` (any `[hidden_size floats]` loads, bring your own), `DELETE
/directions/{name}`.

The slider and `/steer` are **global** - they steer every request the server
serves, which is right for hand-exploration and wrong for apps (we learned
this by breaking a websearch agent with a vector tuned for a chat agent).
Apps should instead scope steering to a single request by adding a
`steering` object to the chat completions body:

```python
client.chat.completions.create(model=..., messages=...,
    extra_body={"steering": {"name": "no-smalltalk", "strength": 8,
                             "layer_from": 16, "layer_to": 18}})
```

It applies only to that generation and the global setting is restored after;
`{"strength": 0}` explicitly opts one request *out* of global steering.

Even better, keep the app steering-agnostic: tag requests with standard
OpenAI `metadata` (e.g. `{"agent": "support-bot", "phase": "triage"}`) and
give brainscope a **steering policy** - rules that map tags to steering,
managed over HTTP and persisted with `--policy policy.json`:

```bash
curl -X POST localhost:8010/policy -d '{"policy": [
  {"match": {"agent": "support-bot", "phase": "triage"},
   "steer": {"name": "no-smalltalk", "strength": 8, "layer_from": 16, "layer_to": 18}}]}'
```

First matching rule wins, missing match keys are wildcards, and the viz
labels every generation with its tags - so you also see *who* is talking.

Still early: extraction quality decides everything, over-steering degrades
the model into repetition, and per-layer vectors are on the roadmap. Before
steering anything real, read [docs/steering.md](docs/steering.md) - a case
study and the lessons we learned the hard way (what data works, which layers
to steer, how over-steering fails, and how the global/per-request split
saves your other agents).

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

## Standing on shoulders

The instruments here implement or adapt published techniques — the credit
belongs with the originals:

- **Logit lens** — nostalgebraist, *interpreting GPT: the logit lens*
  (LessWrong, 2020). Brainscope's per-layer readout is the raw lens; for the
  cleaned-up successor see Belrose et al., *Eliciting Latent Predictions from
  Transformers with the Tuned Lens* (arXiv:2303.08112).
- **Concept-before-language** — the multilingual soup you'll see mid-stack is
  the phenomenon studied in Wendler et al., *Do Llamas Work in English? On the
  Latent Language of Multilingual Transformers* (arXiv:2402.10588).
- **Activation steering** — adding a direction to the residual stream follows
  Turner et al., *Activation Addition* (arXiv:2308.10248); the contrast-pair
  extraction in `hidden_directions.py` is in the spirit of Zou et al.,
  *Representation Engineering* (arXiv:2310.01405) and Rimsky et al.,
  *Steering Llama 2 via Contrastive Activation Addition* (arXiv:2312.06681).
- **Attention aggregation** — the "sources" view averages attention across
  layers; for the principled version of cross-layer attention flow see
  Abnar & Zuidema, *Quantifying Attention Flow in Transformers*
  (arXiv:2005.00928).

## License

MIT © Kateřina Fajmanová
