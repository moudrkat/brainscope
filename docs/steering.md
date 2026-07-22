# Steering

Extract a behaviour direction from contrast pairs, load it, and drive it live —
activation addition (Turner et al., arXiv:2308.10248) on real traffic. This is
the full guide: how to extract a direction, how to drive it, and — most
importantly — [the lessons](#field-notes--lessons-learned) from shipping one for
real, because extraction quality decides everything and over-steering quietly
breaks the model.

← back to the [README](../README.md).

## Extracting a direction

Two extractors ship with brainscope:

- `brainscope.extract` — quick mean-difference at one layer you pick. Takes
  `{"positive": ..., "negative": ...}` lines (see `examples/*.jsonl`); fine
  for strong directions like language switching.
- `brainscope.pca_directions` — the serious one: the first principal
  component of completion-hidden differences at *every* layer, scored by how
  cleanly it separates the two sides — you learn *where* the behaviour lives
  instead of guessing. Takes `{"prompt": ..., "positive": ..., "negative":
  ..., "system": ...}` lines — the same prompt with two continuations that
  differ in exactly the behaviour you want (see
  `examples/no_smalltalk_prompt_pairs.jsonl`).

```bash
python -m brainscope.pca_directions --model qwen3-4b \
    --pairs pairs.jsonl --name no-smalltalk --out dirs.json
# prints a per-layer score table and the suggested steering layer range
brainscope --model qwen3-4b --directions dirs.json
```

**The better starting point on a supported model:
[hidden-directions](https://github.com/moudrkat/hidden-directions).** The sister
repo is a whole steering toolkit, not just a file of vectors:

- **Use its vectors as-is.** The dictionary ships 40 *pre-verified* directions —
  sycophant, refusal, a dozen contested-factual personas — each with a
  recommended strength and layer that prefills the controls the moment you pick
  it. Load a folder with `--directions
  hidden-directions/direction_dict/qwen2.5-7b` and you skip the extraction
  lottery entirely; extraction quality is the hard part, so this is often the
  better place to start.
- **Build more with its pipeline.** It carries the extraction recipes and the
  contrast-pair datasets behind those 40 directions — the template to add your
  own vector to the catalogue instead of starting from a blank `pairs.jsonl`.
- **Follow its references.** The methodology and the papers behind each
  direction are documented there, so it doubles as the reading list for the
  method summarised in [References](#references) below.

See [docs/auditing.md](auditing.md) for the catalogue and how the same vectors
also power the persona audit.

## Driving it live

In the viz header pick a direction, drag the strength slider and set the
layer range — or script it:

```bash
curl -X POST localhost:8010/steer -d '{"name":
  "no-smalltalk", "strength": 8, "layer_from": 16, "layer_to": 18}'
```

The vector library is `dirs.json` next to the server, manageable over HTTP
(`GET`/`POST /directions`, `DELETE /directions/{name}`).

The slider and `/steer` are **global** — right for hand-exploration, wrong
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
with its tags — so you also see *who* is talking.

## Field notes & lessons learned

What we learned shipping a real steering vector with brainscope — extracting a
behaviour direction from a working agent application and using it to switch
off one concrete misbehaviour of the model.

### The case study

An assistant had one conversational behaviour it was never supposed to show in
a particular mode — violated often enough that a dedicated validator LLM ran
after every response just to catch it. Goal: replace that check with an
activation-steering vector.

- **Data**: contrast pairs `{prompt, positive, negative}` — same prompt, a
  compliant completion vs. one showing the unwanted behaviour. Two datasets:
  one built from real traffic, one fully synthetic (templated topics, the
  exact forbidden phrasings injected).
- **Extraction**: `brainscope.pca_directions` — per-layer PCA over
  completion-token hidden-state diffs + a per-layer separation score.
- **Eval**: held-out real prompts, generated baseline vs. steered at several
  strengths; a deterministic violation checker mirroring the validator's rules
  plus coherence guards (repetition, language intact). The eval's baseline
  violation rate matched what the validator saw in practice — which is how you
  know the eval measures the real thing.

Result: with the synthetic vector applied to a narrow band of mid-depth
layers, violations on held-out prompts dropped by roughly an order of
magnitude — and the rare survivor was also the rare incoherent output. Among
coherent outputs: zero violations.

### The lessons

1. **Clean templated contrast beats realistic-but-noisy pairs — by a lot.**
   The synthetic direction scored ~10× higher layer separation and steered
   cleanly. The real-traffic direction was toxic at every dose: below the
   effective strength it did nothing, above it the model collapsed. Real
   diffs carry format, length and topic noise, so the principal component
   isn't purely the behaviour. If you must use real data, pair same-length
   completions on the same prompt and topic.

2. **Steer a narrow window near the score peak.** The per-layer table showed a
   broad plateau; steering all plateau layers at once meant a huge total
   injection — the model degraded before the behaviour fully disappeared.
   A few layers around the peak tolerated ~3× the strength with fewer side
   effects. Rule of thumb: total injection ≈ strength × number of layers.

3. **Low layers amplify.** The real-data vector peaked in the first quarter of
   the stack and destroyed the model at any effective strength — an early
   perturbation cascades through everything downstream. Mid-depth (40–70 %)
   is the safe playground, exactly where the literature puts behaviour
   directions.

4. **Over-steering has a signature.** Repetition loops first (a short phrase
   echoing over and over), then language switching, then token salad. Sweep
   strengths and score *coherence* alongside your target metric — a checker
   that only counts violations will happily report a broken model as perfect.

5. **Evaluating the agent ≠ evaluating the deployment.** `/steer` is global
   server state: every request through the server generates under the vector.
   Our eval covered the target agent only; in the full pipeline a *different*
   agent (different prompt, same server) collapsed into repetition on first
   contact. Scope steering to the calls that want it — per-request steering in
   the chat payload, a tag-matched steering policy, or route only the target
   agent through brainscope.

6. **Keep a boring baseline.** Our violation checker reproduced the deployed
   validator's fail rate before any steering. If your eval's baseline doesn't
   match reality, fix the eval before believing anything else it says.

## Recipe

```bash
# 1. build pairs.jsonl - templated, one behaviour, same prompt both sides
# 2. extract + read the layer table
python -m brainscope.pca_directions --model qwen3-4b \
    --pairs pairs.jsonl --name my-direction --out dirs.json
# 3. serve with the vector loaded
brainscope --model qwen3-4b --directions dirs.json
# 4. sweep: narrow layer window around the best layer, several strengths,
#    check BOTH your target metric and coherence on held-out prompts
curl -X POST localhost:8010/steer \
    -d '{"name": "my-direction", "strength": 6, "layer_from": 20, "layer_to": 22}'
```

## References

The method here is activation addition (Turner et al., *Activation
Addition: Steering Language Models Without Optimization*, arXiv:2308.10248)
with contrast-pair, PCA-per-layer extraction in the spirit of representation
engineering (Zou et al., arXiv:2310.01405) and contrastive activation
addition (Rimsky et al., arXiv:2312.06681). The layer-depth intuitions in
the lessons match what those papers report; the over-steering failure modes
are our own scars.
## Hotwire interop: one spec dialect, explicit regimes

The canonical steering spec is [hotwire](https://github.com/moudrkat/hotwire-vllm)'s
— that is the format people deploy with, so brainscope accepts it natively:

```json
{"id": "calm", "layer": 20, "scale": 3, "decode_only": true}
```

either as a `steering` object or as the full hotwire wire format
(`"vllm_xargs": {"hotwire": "<json string>"}`) — a request built for a
hotwire-vLLM server steers brainscope unchanged. The legacy
`{"name", "strength", "layer_from", "layer_to"}` dialect still works
(UI slider, old policy files).

**Regimes are explicit now.** Two per-spec flags say *what* gets steered:

- `decode_only: true` (or `prefill: false`) — steer generated tokens only,
  never the prompt. This is the regime most vectors are calibrated in;
  applying such a vector to a long prompt as well multiplies the effective
  dose and can wreck coherence.
- `syntax_mute: false` — opt out of the tool-call scaffolding mute
  (default on: JSON syntax stays well-formed, the persona speaks only
  inside argument string values).

**Export with a passport:** `python -m brainscope.export_hotwire --dirs
dirs.json --out ./vectors` writes hotwire-ready `.pt` files plus a
`manifest.json` recording shape, norms, model, and the calibration regime —
a vector should never travel without its regime again.

**Parity check:** `python -m brainscope.parity` sends the same prompts and
spec to a brainscope and a hotwire server and compares the behavior — the
cheap standing guard against the two backends' steering semantics drifting
apart.
