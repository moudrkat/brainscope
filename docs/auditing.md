# Auditing baked personas

A 9 KB weights patch can turn a model into a covert advocate — with **no
runtime steering active**. This is the live half of a persona audit: serve the
baked model, load the persona catalogue, and watch the persona surface in the
representations, token by token.

← back to the [README](../README.md).

## The sister project

My research repo
[hidden-directions](https://github.com/moudrkat/hidden-directions) shows that
an advocate persona (say, a flat-earther) can be *baked into 9 KB* of a
model's weights — one MLP bias — and ships a catalogue of persona directions
plus the audit tool that catches such bakes on disk. brainscope is the live
half of that audit.

## Catch a bake live

```bash
brainscope --model Qwen/Qwen2.5-7B-Instruct --quantize 8bit \
    --bake hidden-directions/artifacts/example_flat_earth_7b \
    --directions hidden-directions/direction_dict/qwen2.5-7b
```

Ask it about the shape of the Earth: it answers flat — with **no runtime
steering active** — and every token reports its per-layer cosine with each
catalogued direction, so `v_pref_flat_earth` visibly lights up from the baked
layer on. Restart without `--bake` for the clean baseline.

## The dictionary

The dictionary ships 40 directions — sycophant, refusal, a dozen
contested-factual personas, an "evil" escalation ladder — each with a
verified strength/layer that prefills the controls when you pick it. The
[hidden-directions README](https://github.com/moudrkat/hidden-directions#the-qwen-25-7b-dictionary)
catalogues them and lists starting points (e.g. `v_pref_sycophant` +1.5,
`v_refusal` +2).

## Composing the pieces

`--directions` takes any `direction_dict/` folder
(per-layer matrices are applied row-per-layer; tensors load with
`weights_only`, so a dict from the internet cannot execute code), the
manifest's model is checked and mismatches are warned about, and the
manifest's `recommended_layer`/`recommended_alpha` prefill the strength and
layer controls when you pick a direction. Several vectors apply at once —
the ＋ button stacks them in the UI, and the API takes
`{"stack": [spec, ...]}` wherever a single steering spec goes — so the exact
bake recipe can be re-created live:

```bash
curl -X POST localhost:8010/steer -d '{"stack": [
  {"name": "v_pref_flat_earth", "strength": 1.5, "layer_from": 17, "layer_to": 17},
  {"name": "v_refusal",        "strength": -1.0, "layer_from": 17, "layer_to": 17}]}'
```