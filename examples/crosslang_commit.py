"""Cross-language commit-depth — a companion to brainscope's logit lens.

brainscope's logit lens shows *where each word's prediction settles* in the residual
stream. This example asks a follow-up: **does that settling happen later for
typologically more distant languages?** — i.e. is "meaning before language"
(Wendler et al. 2024) stronger the harder the target language is?

For a shared concept (e.g. `cat`) given a few English→L exemplars, we find the
earliest layer from which the target-language token stays top-1 over the English
token (the "commit depth", normalized 0..1), then correlate mean commit-depth with
a typological-difficulty rank across a ladder: German < French < Spanish < Czech <
Finnish < Icelandic.

    python examples/crosslang_commit.py --model tiny
    python examples/crosslang_commit.py --model Qwen/Qwen2.5-3B-Instruct --plot

Data: examples/crosslang_commit_ladder.jsonl  (one concept per line, translations per lang).
Pure Hugging Face + the same logit-lens idea brainscope already visualizes — no server needed.
"""
import argparse
import json
import statistics
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# difficulty ladder (rough typological distance from English, low → high)
LADDER = ["de", "fr", "es", "cs", "fi", "is"]
LANG_NAME = {"de": "German", "fr": "French", "es": "Spanish",
             "cs": "Czech", "fi": "Finnish", "is": "Icelandic"}
HERE = Path(__file__).resolve().parent


def first_tok(tok, word):
    ids = tok(" " + word, add_special_tokens=False).input_ids
    return ids[0] if ids else None


@torch.inference_mode()
def commit_depth(model, tok, lang, concept, rows, device):
    """Earliest normalized layer from which the L-token stays top-1 over the EN-token."""
    row = next(r for r in rows if r["en"] == concept)
    tgt, eng = first_tok(tok, row[lang]), first_tok(tok, concept)
    if tgt is None or eng is None:
        return None
    exemplars = [r for r in rows if r["en"] != concept][:6]
    lines = [f"English: {r['en']} = {LANG_NAME[lang]}: {r[lang]}" for r in exemplars]
    lines.append(f"English: {concept} = {LANG_NAME[lang]}:")
    ids = tok("\n".join(lines), return_tensors="pt").to(device)
    hs = model(**ids, output_hidden_states=True).hidden_states
    norm, head, n = model.model.norm, model.lm_head, len(hs) - 1
    wins = [bool(head(norm(hs[i][:, -1, :]))[0][tgt] > head(norm(hs[i][:, -1, :]))[0][eng])
            for i in range(1, len(hs))]
    for i in range(n):
        if all(wins[i:]):
            return (i + 1) / n
    return 1.0


def corr(xs, ys):
    pairs = [(a, b) for a, b in zip(xs, ys) if a is not None and b is not None]
    if len(pairs) < 3:
        return None
    xs, ys = zip(*pairs)
    mx, my = statistics.mean(xs), statistics.mean(ys)
    cov = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    vx = sum((a - mx) ** 2 for a in xs)
    vy = sum((b - my) ** 2 for b in ys)
    return round(cov / ((vx * vy) ** 0.5), 3) if vx and vy else None


PRESETS = {"tiny": "Qwen/Qwen2.5-0.5B-Instruct"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="tiny")
    ap.add_argument("--data", default=str(HERE / "crosslang_commit_ladder.jsonl"))
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()

    name = PRESETS.get(args.model, args.model)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dt = torch.bfloat16 if device == "cuda" else torch.float32
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=dt).to(device).eval()
    rows = [json.loads(l) for l in open(args.data, encoding="utf-8")]
    concepts = [r["en"] for r in rows]

    print(f"model={name}  concepts={len(concepts)}  ladder={LADDER}\n")
    depths = []
    for lang in LADDER:
        ds = [d for c in concepts if (d := commit_depth(model, tok, lang, c, rows, device)) is not None]
        m = round(statistics.mean(ds), 3) if ds else None
        depths.append(m)
        print(f"  {LANG_NAME[lang]:10}  commit_depth = {m}")

    c = corr(depths, list(range(len(LADDER))))
    print(f"\ncommit_depth vs difficulty rank:  {c}")
    print("(positive → harder languages are assembled later — 'meaning before language' scales with distance)")

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(7, 4))
        plt.plot([LANG_NAME[l] for l in LADDER], depths, "o-", color="#c41e1e")
        plt.ylabel("mean commit depth (0..1)")
        plt.title(f"Assembly depth vs language difficulty (corr={c})")
        plt.tight_layout()
        out = HERE / "crosslang_commit.png"
        plt.savefig(out, dpi=120)
        print(f"plot → {out}")


if __name__ == "__main__":
    main()
