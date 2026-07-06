"""Extract a steering direction from contrast prompt pairs (ActAdd/repeng style).

Input: JSONL with {"positive": "...", "negative": "..."} per line — texts that
differ in exactly the property you want a direction for (language, tone,
formality…). The direction is the mean difference of last-token hidden states
at --layer, normalized. Save several under different names and load them all:

    python -m brainscope.extract --model tiny --pairs pairs.jsonl \\
        --layer 12 --name czech --out dirs.json
    python -m brainscope.server --model tiny --directions dirs.json
"""

import argparse
import json
from pathlib import Path

import torch

from .server import PRESETS


@torch.inference_mode()
def extract(model_name: str, pairs: list[dict], layer: int, device: str | None) -> torch.Tensor:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16 if dev == "cuda" else torch.float32).to(dev).eval()

    def last_hidden(text: str) -> torch.Tensor:
        ids = tok(text, return_tensors="pt").input_ids.to(dev)
        out = model(input_ids=ids, output_hidden_states=True)
        return out.hidden_states[layer][0, -1].float()

    diffs = [last_hidden(p["positive"]) - last_hidden(p["negative"]) for p in pairs]
    direction = torch.stack(diffs).mean(0)
    return direction / direction.norm()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--layer", type=int, required=True,
                        help="which hidden layer to read (≈40-70%% of depth works well)")
    parser.add_argument("--name", required=True)
    parser.add_argument("--out", type=Path, default=Path("dirs.json"))
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    pairs = [json.loads(line) for line in args.pairs.read_text().splitlines() if line.strip()]
    direction = extract(PRESETS.get(args.model, args.model), pairs, args.layer, args.device)

    existing = json.loads(args.out.read_text()) if args.out.exists() else {}
    existing[args.name] = [round(x, 6) for x in direction.tolist()]
    args.out.write_text(json.dumps(existing))
    print(f"saved direction '{args.name}' ({len(direction)} dims, {len(pairs)} pairs) -> {args.out}")


if __name__ == "__main__":
    main()
