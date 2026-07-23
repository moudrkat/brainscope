# Cross-language commit depth

A small companion to brainscope's **logit lens** (the "where each word's prediction
settled" view). It turns that single-word observation into a comparison across a
**difficulty ladder** of languages, to test a claim from
[Wendler et al. 2024, *"Do Llamas Work in English?"*](https://arxiv.org/abs/2402.10588):
that non-English meaning is assembled in the *later* layers, routed through an
English-ish concept space in the middle.

**Question:** does the layer at which a translation "commits" (stays top-1 over the
English token) get **later** as the target language gets typologically more distant?

**Ladder (low → high distance from English):**
German < French < Spanish < Czech < Finnish < Icelandic

**Run:**
```bash
python examples/crosslang_commit.py --model tiny            # CPU, 0.5B
python examples/crosslang_commit.py --model Qwen/Qwen2.5-3B-Instruct --plot
```

**What it does:** for each shared concept (e.g. `cat`), a few English→L exemplars prime
the model; the logit lens is read at every layer; commit depth = earliest normalized
layer from which the L-token beats the English token and stays ahead. Mean commit depth
per language is then correlated with the difficulty rank.

**Observed (Qwen family + Mistral-7B, 14 concepts):** commit depth rises with difficulty
— correlation ≈ **+0.85**, with **Icelandic assembled latest** and Spanish/German earliest.
Directionally consistent with Wendler's "meaning before language", now measured across
typological distance rather than a single language.

Data: [`examples/crosslang_commit_ladder.jsonl`](../examples/crosslang_commit_ladder.jsonl)
— one concept per line, translations per language; easy to extend with more concepts or languages.

*Contributed as a companion lens — pairs naturally with the live logit-lens view. Happy to
follow up with a UI panel (per-layer curve per language) if useful.*
