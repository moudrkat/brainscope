# brainscope

**Watch your model think while your app talks to it.**

An OpenAI-compatible chat server over any Hugging Face causal LM that streams
per-token, per-layer residual-stream activity to a live browser visualization.
No changes to your app: point its OpenAI `base_url` at brainscope and open
the viz in a window next to it. Live activation steering is wired in as an
early work-in-progress.

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

Or skip Python entirely and run the Docker image:

```bash
docker run -p 127.0.0.1:8010:8010 -v ~/.cache/huggingface:/root/.cache/huggingface \
  ghcr.io/moudrkat/brainscope:cpu
```

Anything after the image name goes to the brainscope CLI. With an NVIDIA GPU
(and nvidia-container-toolkit) use the `:cuda` tag and bigger models:

```bash
docker run --gpus all -p 127.0.0.1:8010:8010 -v ~/.cache/huggingface:/root/.cache/huggingface \
  ghcr.io/moudrkat/brainscope:cuda --model qwen3-4b
```

The cache mount keeps downloaded model weights on your disk, so they survive
container restarts. The `127.0.0.1:` binding keeps the port private to your
machine — brainscope has no auth, and Docker port mappings bypass ufw-style
firewalls, so only drop it (`-p 8010:8010`) on a network you trust.

`--model` takes any Hugging Face model id, plus presets: `tiny`
(Qwen2.5-0.5B, CPU-friendly), `qwen3-4b`, `qwen3-8b`, `qwen3.5-9b`,
`gemma-e4b`. Bigger models fit a 16 GB card with `--quantize 8bit`.
Pointing your app at it is one line - wherever it builds its OpenAI client:

```python
client = OpenAI(base_url="http://localhost:8010/v1", api_key="unused")
```

## What am I looking at?

Left: the model itself - the prompt enters at the bottom, one **clickable row
per decoder layer**, the next word exits at the top (lm_head). On the right,
four instruments:

- **activity over time** - one column per generated token, one row per
  layer, color = how loudly that layer works relative to its own average.
- **attention** - for the clicked layer: what each answer token looks back
  at; **heads** splits the newest token per attention head.
- **logit lens** (click lm_head) - every layer's next-token readout: watch
  the answer crystallize with depth. Hover a cell for the top-5 candidates,
  click to pin the tooltip.
- **the answer text is an instrument too** - each word is tinted by the
  layer where its prediction settled (clean = early, amber = late, red =
  never before lm_head); hovering shows what the model almost said instead.

The ◉ scope button pauses capture when you just want fast generation;
● record exports a WebM, PNG saves a snapshot.

An example of what the lens view can catch:

![Logit lens: the meaning decodes mid-stack in English and Chinese; the Czech surface form assembles only in the last few layers](docs/img/lens-concept-before-language.png)

*Qwen3-4B writing the Czech word "zážitkům" (experiences). Mid-stack readouts
decode the meaning - in English and Chinese - while the Czech surface form
assembles only in the last few layers: the geometry of multilingual
representations, studied properly in Wendler et al. 2024 (arXiv:2402.10588).
Readouts are a raw logit lens, so mid-stack tokens are approximate.*

## Steering

Extract a direction from contrast pairs, load it, and drive it live -
activation addition (Turner et al., arXiv:2308.10248) on real traffic. Two
extractors ship with brainscope:

- `brainscope.extract` - quick mean-difference at one layer you pick. Takes
  `{"positive": ..., "negative": ...}` lines (see `examples/*.jsonl`); fine
  for strong directions like language switching.
- `brainscope.hidden_directions` - the serious one: the first principal
  component of completion-hidden differences at *every* layer, scored by how
  cleanly it separates the two sides - you learn *where* the behaviour lives
  instead of guessing. Takes `{"prompt": ..., "positive": ..., "negative":
  ..., "system": ...}` lines - the same prompt with two continuations that
  differ in exactly the behaviour you want.

```bash
python -m brainscope.hidden_directions --model qwen3-4b \
    --pairs pairs.jsonl --name no-smalltalk --out dirs.json
# prints a per-layer score table and the suggested steering layer range
brainscope --model qwen3-4b --directions dirs.json
```

In the viz header pick a direction, drag the strength slider and set the
layer range - or script it: `curl -X POST localhost:8010/steer -d '{"name":
"no-smalltalk", "strength": 8, "layer_from": 16, "layer_to": 18}'`. The
vector library is `dirs.json` next to the server, manageable over HTTP
(`GET`/`POST /directions`, `DELETE /directions/{name}`).

The slider and `/steer` are **global** - right for hand-exploration, wrong
for apps (a vector tuned for one agent breaks another; we know). Apps scope
steering to a single request instead:

```python
client.chat.completions.create(model=..., messages=...,
    extra_body={"steering": {"name": "no-smalltalk", "strength": 8,
                             "layer_from": 16, "layer_to": 18}})
```

`{"strength": 0}` opts a request *out* of global steering. Even better, keep
the app steering-agnostic: tag requests with standard OpenAI `metadata`
(e.g. `{"agent": "support-bot"}`) and give brainscope a **steering policy**
mapping tags to steering (`--policy policy.json`, managed via `POST
/policy`). First matching rule wins, and the viz labels every generation
with its tags - so you also see *who* is talking.

Still early: extraction quality decides everything, and over-steering
degrades the model into repetition. Before steering anything real, read
[docs/steering.md](docs/steering.md) - a case study and the lessons we
learned the hard way.

## Will it work with my app?

Works when your app talks the **OpenAI chat-completions API** with
**non-streaming** responses (`stream: true` not supported yet); tool calls
are parsed in hermes/qwen, gemma-fenced and plain-JSON formats.

Honest limitations: generation runs on plain `transformers` - tens of tokens
per second, one request at a time, no auth, context bounded by VRAM. Why not
vLLM? vLLM is a black box by design - per-layer states are consumed the
moment they're produced; `transformers` exposes them for every architecture
with one flag. That's the trade: brainscope is slower, but it sees
everything. It's a lab instrument for development - run it next to
production, not instead of it.

## Standing on shoulders

The instruments implement or adapt published techniques - the credit belongs
with the originals:

- **Logit lens** - nostalgebraist, *interpreting GPT: the logit lens*
  (LessWrong, 2020); the cleaned-up successor is the tuned lens, Belrose
  et al. (arXiv:2303.08112).
- **Concept-before-language** - Wendler et al., *Do Llamas Work in English?*
  (arXiv:2402.10588).
- **Activation steering** - Turner et al., *Activation Addition*
  (arXiv:2308.10248); extraction in the spirit of Zou et al.
  (arXiv:2310.01405) and Rimsky et al. (arXiv:2312.06681).
- **Attention aggregation** - the "sources" view averages attention across
  layers; the principled cross-layer flow is Abnar & Zuidema
  (arXiv:2005.00928).

## License

MIT © Kateřina Fajmanová
