# brainscope

Watch your model think while your app talks to it.

An OpenAI-compatible chat server over any Hugging Face causal LM that streams
per-token, per-layer residual-stream activity to a live browser visualization.
No changes to your app — just point its OpenAI `base_url` here and open the
viz in a window next to it.

```bash
pip install -e .
python -m brainscope.server --model Qwen/Qwen3-4B-Instruct-2507 --port 8010
# bigger models on a 16 GB card:
# python -m brainscope.server --model Qwen/Qwen3.5-9B --quantize 8bit
# app  -> http://<host>:8010/v1  (chat completions incl. tool calls, hermes format)
# eyes -> http://<host>:8010/
```

Optional: `--directions dirs.json` (`{"name": [hidden_size floats]}`) adds
per-layer cosine projections of every generated token onto named directions —
watch a steering vector engage, layer by layer. (Applying vectors, not just
watching them, is next.)

## Steering (v0)

Extract a direction from contrast pairs, load it, and drive it live with the
slider in the viz header — activation addition on real traffic:

```bash
python -m brainscope.extract --model tiny --pairs examples/czech_pairs.jsonl --layer 12 --name czech
python -m brainscope.server --model tiny --directions dirs.json
```

Built-ins worth knowing: chat box on the viz page (no app needed), ● záznam
(WebM of the heatmap), PNG snapshot, hover for token/layer detail, model
presets (`--model tiny|qwen3-4b|qwen3.5-9b|gemma-e4b`).
