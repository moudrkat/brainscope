# Reasoning traces — persist, replay, and ask when the answer emerged

Chat responses strip the `<think>…</think>` block; the interesting part of
a reasoning model's generation is exactly the part your app never sees.
With `--traces DIR` brainscope keeps it all: every generation is persisted
with everything the instruments saw, replayable token by token, and
analyzable after the fact.

← back to the [README](../README.md).

## Turning it on

```bash
brainscope --model qwen3-8b --traces traces/ --jlens lenses/qwen3-8b.jlens.pt
```

Each finished generation writes one JSON to the directory: tokens, per-layer
norms, logit-lens and [J-lens](jlens.md) readouts, steering state, tags, and
the `<think>` segmentation. The oldest traces are dropped beyond
`--keep-traces` (default 200). A **traces** tab appears in the viz.

Two runtime switches (`POST /traces/config`, also buttons in the tab):

- `{"save": false}` — pause persistence without restarting.
- `{"hidden": true}` — **the heavy option**: store every captured step's
  full `[n_layers, hidden]` residual (fp16) next to the trace. Tens of MB
  per trace for small models, hundreds for 9B-class — but it's what makes
  post-hoc analytics *exact* instead of a lower bound. Off by default;
  capped at `HIDDEN_MAX_STEPS` (env, default 2048) steps per trace.

## Replay

Click a trace in the list: the text renders with the think block dimmed,
and a scrubber walks the generation token by token — at every step you see
the logit-lens column ("would say now") and the J-lens column ("pushed
toward later") for that exact moment. `{"raw": true}` in a chat request keeps the
unstripped text in the API response too (`message.raw_content`), so
programmatic clients can line responses up with traces.

## Answer emergence — when did it actually decide?

The chart under the replay answers the question that makes reasoning traces
worth inspecting: **for the token that opens the final answer, how probable
was it at every step of the think block?** Best layer per step, one curve
per lens:

- **amber — logit lens**: the model would *say* it now.
- **green — J-lens**: the model is *holding* it (the global-workspace
  effect: this curve typically rises while the text is still reasoning).
- **dashed** — top-k lower bound, used when hidden states weren't stored
  (the token only counts when it made the stored top-5). Flip
  `hidden: on` and the curves become exact.
- **│** — where `</think>` ends.

Click any word of the trace to track that word instead — "when did *Paris*
enter the picture?" is one click. The API returns the raw series:

```
GET /traces/{id}/emergence            # first answer token
GET /traces/{id}/emergence?token=Paris
```

`emerge_step` in the response is the first step each curve crosses p = 0.1
— a number you can compare across lenses (which saw it first?), across
steering settings, or between the J-lens and the experimental
[A-lens](jlens.md#a-lens-experimental-ours).

## API summary

| endpoint | what |
|---|---|
| `GET /traces` | index (id, time, preview, badges) |
| `GET /traces/{id}` | full trace JSON |
| `DELETE /traces/{id}` | remove trace + hidden states |
| `POST /traces/config` | `{"save": bool, "hidden": bool}` |
| `GET /traces/{id}/emergence` | answer-emergence series (`?token=` to override) |

## Honesty note

A visible chain-of-thought is not a faithful account of the computation —
models can reason toward an answer they already hold, or verbalize reasons
they didn't use. That's precisely why this inspector reads *activations*
alongside the text. Still: the emergence chart **illustrates** where the
answer surfaces in the residual stream of your model on your prompts;
measuring workspace effects properly — with interventions and controls — is
what Anthropic's [global workspace paper](https://www.anthropic.com/research/global-workspace)
does, and the faithfulness question has its own literature. When a curve
surprises you, the next step is an intervention (steer the concept, rerun,
compare traces), not a conclusion.
