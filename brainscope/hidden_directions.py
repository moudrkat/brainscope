"""Find a behavioural direction in the residual stream from contrast pairs.

A serious upgrade over `brainscope.extract`: instead of one mean-difference at
one hand-picked layer, this reads hidden states of the *completion* tokens for
a positive and a negative continuation of the same prompt, computes per-pair
per-layer differences, and takes the first principal component of those
differences at every layer (repeng-style). Layers are then scored by how
cleanly the direction separates positive from negative completions, so you
learn *where* in the model the behaviour lives instead of guessing.

Input JSONL, one pair per line:
    {"prompt": "...", "positive": "...", "negative": "...", "system": "..."}
(`system` optional; `positive`/`negative` are assistant completions.)

    python -m brainscope.hidden_directions --model qwen3-4b \
        --pairs pairs.jsonl --name discuss-no-tasks --out dirs.json

The winning layer's direction is saved into dirs.json (brainscope --directions
format); the printed table tells you which layer range to steer.
"""

import argparse
import json
from pathlib import Path

import torch

from .server import PRESETS


@torch.inference_mode()
def completion_hiddens(model, tok, device, system: str | None, prompt: str,
                       completion: str) -> torch.Tensor:
    """Mean hidden state of the completion tokens, per layer -> [L, H]."""
    messages = ([{"role": "system", "content": system}] if system else []) + [
        {"role": "user", "content": prompt}
    ]
    prefix_ids = tok.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    )
    full_ids = tok.apply_chat_template(
        messages + [{"role": "assistant", "content": completion}],
        add_generation_prompt=False, return_tensors="pt",
    )
    n_prefix = prefix_ids.shape[1]
    out = model(input_ids=full_ids.to(device), output_hidden_states=True)
    # hidden_states: (n_layers+1) tensors [1, T, H]; skip the embedding layer
    # and average over completion positions only (the behaviour lives there,
    # the shared prompt would just dilute the contrast).
    span = slice(n_prefix, full_ids.shape[1])
    return torch.stack(
        [h[0, span].float().mean(0) for h in out.hidden_states[1:]]
    )  # [L, H]


def pca_direction(diffs: torch.Tensor) -> torch.Tensor:
    """First principal component of [N, H] diffs, sign-aligned with the mean."""
    mean = diffs.mean(0)
    # PCA on the raw (uncentered) diffs: we want the dominant *difference*
    # axis, and centering would remove exactly the consistent part of it.
    _, _, vt = torch.linalg.svd(diffs, full_matrices=False)
    direction = vt[0]
    if torch.dot(direction, mean) < 0:
        direction = -direction
    return direction / direction.norm()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="preset or HF model id")
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--out", type=Path, default=Path("dirs.json"))
    parser.add_argument("--device", default=None)
    parser.add_argument("--quantize", choices=["8bit", "4bit"], default=None)
    parser.add_argument("--meta-out", type=Path, default=None,
                        help="where to write the per-layer score table (JSON)")
    args = parser.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_id = PRESETS.get(args.model, args.model)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(model_id)
    kwargs: dict = {"torch_dtype": torch.bfloat16 if device == "cuda" else torch.float32}
    if args.quantize:
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_8bit=args.quantize == "8bit", load_in_4bit=args.quantize == "4bit"
        )
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    if not args.quantize:
        model = model.to(device)
    model.eval()

    pairs = [json.loads(l) for l in args.pairs.read_text().splitlines() if l.strip()]
    pos_h, neg_h = [], []
    for i, p in enumerate(pairs):
        pos_h.append(completion_hiddens(model, tok, device, p.get("system"),
                                        p["prompt"], p["positive"]))
        neg_h.append(completion_hiddens(model, tok, device, p.get("system"),
                                        p["prompt"], p["negative"]))
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(pairs)} pairs")
    pos = torch.stack(pos_h)   # [N, L, H]
    neg = torch.stack(neg_h)
    diffs = pos - neg

    n_layers = diffs.shape[1]
    table = []
    directions = []
    for layer in range(n_layers):
        d = pca_direction(diffs[:, layer])
        directions.append(d)
        proj_pos = pos[:, layer] @ d
        proj_neg = neg[:, layer] @ d
        gap = proj_pos - proj_neg
        # t-like separation: consistent gap across pairs beats a big noisy one
        score = (gap.mean() / (gap.std() + 1e-6)).item()
        acc = (gap > 0).float().mean().item()
        table.append({"layer": layer + 1, "score": round(score, 3),
                      "pair_acc": round(acc, 3)})

    best = max(table, key=lambda r: r["score"])
    good = [r["layer"] for r in table if r["score"] >= 0.9 * best["score"]]
    print(f"\n{'layer':>5} {'score':>8} {'pair_acc':>9}")
    for r in table:
        marker = " <-- best" if r["layer"] == best["layer"] else (
            " *" if r["layer"] in good else "")
        print(f"{r['layer']:>5} {r['score']:>8.3f} {r['pair_acc']:>9.3f}{marker}")
    print(f"\nbest layer: {best['layer']} (score {best['score']}, "
          f"pair_acc {best['pair_acc']})")
    print(f"suggested steering range: layers {min(good)}-{max(good)}")

    existing = json.loads(args.out.read_text()) if args.out.exists() else {}
    existing[args.name] = [round(x, 6) for x in directions[best["layer"] - 1].tolist()]
    args.out.write_text(json.dumps(existing))
    print(f"saved direction '{args.name}' (layer {best['layer']}) -> {args.out}")

    meta_out = args.meta_out or args.out.with_suffix(".meta.json")
    meta_out.write_text(json.dumps(
        {"name": args.name, "model": model_id, "n_pairs": len(pairs),
         "best_layer": best["layer"], "suggested_layers": [min(good), max(good)],
         "table": table}, indent=2))
    print(f"layer table -> {meta_out}")


if __name__ == "__main__":
    main()
