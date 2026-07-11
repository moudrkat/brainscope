# J-lens — what's on the model's mind

The logit lens asks every layer "what would you say if you stopped *now*".
The **Jacobian lens** asks a better question: "what is this activation
disposed to make the model say *later*?" — and the vocabulary patterns it
picks out light up for concepts the model is holding **silently**, before or
without saying them. Anthropic introduced the technique and the term
**J-space** in *A global workspace in language models*
([announcement](https://www.anthropic.com/research/global-workspace) ·
[paper](https://transformer-circuits.pub/2026/workspace/index.html), 2026);
their [reference implementation](https://github.com/anthropics/jacobian-lens)
is Apache-2.0. brainscope ships an **independent MIT reimplementation from
the paper** (`brainscope/jlens.py`) — the published math, none of their code.

← back to the [README](../README.md).

## The method, in one breath

For every decoder layer `l`, average — over many prompts, source positions
`t` and later positions `t' ≥ t` — the Jacobian of the final hidden state
with respect to that layer's hidden state:

    J_l = E[ ∂h_final(t') / ∂h_l(t) ]

then read any activation out through it:

    lens_l(h) = softmax(W_U · final_norm(J_l · h))

That's the logit lens with a learned linear *transport* into final-layer
space: instead of pretending layer 12's basis already matches the
unembedding, `J_l` moves the activation to where the unembedding is valid,
weighted by measured causal influence on future outputs.

brainscope estimates `J_l` with stochastic vector-Jacobian probes — one
backward pass per random probe, accumulating rank-1 samples (see the module
docstring for the derivation). Two built-in health checks:

- **identity self-test** — `∂h_final/∂h_final = I` exactly, so the fitted
  final layer must converge to the identity; `identity_error` in the
  artifact metadata should be well below 1.
- **sample budget warning** — the estimator needs comfortably more probes
  than the model's hidden size; the CLI warns below `2 × hidden`.

## Fitting a lens

One fit per model, reusable forever (`lenses/` is gitignored — artifacts
are fp16 `[n_layers, d, d]`, tens to hundreds of MB):

```bash
pip install datasets                      # for the wikitext prompt source
brainscope-jlens fit --model tiny --prompts wikitext \
    --out lenses/qwen2.5-0.5b-instruct.jlens.pt --n-texts 128 --repeats 16
```

`--prompts` takes `wikitext` (pretraining-like distribution, as in the
paper) or your own JSONL/plain file of text snippets — fitting on prompts
close to *your* traffic is a feature, not a bug. Reproducible invocations
for every model we've fitted live in
[`examples/fit_jlens.sh`](../examples/fit_jlens.sh).

Cost: `n_texts × repeats` forward+backward passes of `--max-tokens` each.
On one RTX 4070 Ti SUPER: minutes for a 0.5B model, tens of minutes for a
4B. CPU fitting works but takes hours — fit where the GPU is, copy the
artifact home. Everything downstream is cheap: serving adds one matmul per
layer per token.

## Serving with it

```bash
brainscope --model tiny --jlens lenses/qwen2.5-0.5b-instruct.jlens.pt
```

A **J-lens** tab appears next to the logit lens — same grid, same tooltips,
different question. The **◎ j-lens** header switch (or `POST /jlens
{"on": false}`) turns the per-token readout off without a restart, same
spirit as ◉ capture. The server refuses a lens whose shape doesn't match
the loaded model and warns when the fit came from a different model id.

Reading the panel: a word in a J-lens cell is *on the model's mind* at that
layer — disposed to be said later, not necessarily next, not necessarily
ever. Watch it during a `<think>` block: concepts surface here many tokens
before they are verbalized (that's the workspace effect the paper is
about), and the [traces](traces.md) emergence chart turns that impression
into a curve.

## Steering × J-lens

Every vocabulary token has a J-space direction: `J_lᵀ · W_U[token]` — the
per-layer activation pattern that, to first order, raises the token's
*future* logit. brainscope materializes it as a normal `[n_layers, hidden]`
steering direction:

```bash
curl -X POST localhost:8010/jlens/direction -H "Content-Type: application/json" \
     -d '{"text": "cake"}'        # → direction "j:cake", ready to steer
```

or type the word into the header box (**word… → vec**). The direction plugs
into everything steering already does — slider, stacks, per-request
steering, policies — and the J-lens panel doubles as the readout: nudge
what the model is thinking about, then watch whether it took. Injection and
verification in one instrument.

Rows are unit-normalized, so start around the prefilled strength and treat
it like any other direction (see [steering.md](steering.md) for the
over-steering lessons — they all apply).

## A-lens (experimental, ours)

Anthropic's J-lens averages influence over *all* future tokens. For
reasoning models we care about a sharper question: which activations
causally shape **the eventual answer**, ignoring the verbal reasoning in
between? `--mode answer` fits the same estimator with target positions
restricted to tokens after `--marker` (default `</think>`):

```bash
# 1. collect reasoning traces from a running brainscope server
brainscope-jlens gen-traces --base-url http://localhost:8010/v1 --out traces.jsonl
# 2. fit the answer lens on them
brainscope-jlens fit --model qwen3-8b --prompts traces.jsonl --mode answer \
    --out lenses/qwen3-8b.alens.pt
```

Serve with it exactly like a J-lens and compare both in the emergence
chart: does the answer surface earlier or cleaner in A-space?

**Honesty note.** The A-lens is a brainscope experiment, not a published
technique. Its risks are knowable: fewer target positions per text means a
noisier estimate (feed it more traces), and fitting on your own model's
traces narrows the distribution. The emergence view exists precisely so the
variant has to earn its keep against the real J-lens before you believe
anything it shows.

## Limitations (the paper's and ours)

- **Single-token concepts only** — the lens vocabulary is the model's
  vocabulary; multi-token concepts don't get a row. (Paper limitation.)
- **First-order** — a Jacobian is a linearization; strongly nonlinear
  routes to an output are invisible to it.
- **Estimator variance** — our stochastic fit trades exactness for
  simplicity; check `identity_error`, and refit with more texts/repeats if
  it's high.
- **Illustrates vs measures** — the live panel *illustrates* the
  global-workspace effect on your traffic; the careful causal measurements
  (patching, injection, selectivity) are in Anthropic's paper. When the two
  disagree, believe the paper.
