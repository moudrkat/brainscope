"""Parity check: the same steering spec against brainscope AND hotwire-vLLM.

Both backends accept the canonical (hotwire) spec dialect — brainscope via
``steering`` / ``vllm_xargs``, hotwire via ``vllm_xargs``. This script sends
identical prompts + spec to both and reports whether the *behavior* agrees:
exact-match rate, output-length stats, and per-prompt first-divergence. Exact
token equality across backends is NOT expected (different kernels, dtypes);
what should agree is the shape of the effect — if it stops agreeing, one
side's steering semantics drifted (that is how a regime mismatch is caught
by a script instead of a production conversation).

    python -m brainscope.parity \
        --brainscope-url http://localhost:8010 --hotwire-url http://host:8001 \
        --model-b tiny --model-h qwen3-4b \
        --spec '{"id": "calm", "layer": 2, "scale": 4, "decode_only": true}'
"""

import argparse
import json
import urllib.request

NEUTRAL_PROMPTS = [
    "Describe your ideal weekend in two sentences.",
    "A friend is late again. What do you tell them?",
    "Summarize why people keep houseplants.",
    "What should I cook tonight? I have eggs and rice.",
    "Explain rain to a five-year-old.",
    "Write a two-line note canceling a meeting politely.",
]


def ask(base_url: str, model: str, prompt: str, spec, timeout: int = 300) -> str:
    body = {"model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 120, "temperature": 0}
    if spec is not None:
        body["vllm_xargs"] = {"hotwire": json.dumps(spec)}
    req = urllib.request.Request(
        base_url.rstrip("/") + "/v1/chat/completions",
        json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.loads(r.read())
    return d["choices"][0]["message"].get("content") or ""


def compare(pairs: list[tuple[str, str]]) -> dict:
    """Pure comparison of (backend_a_text, backend_b_text) pairs."""
    n = len(pairs)
    exact = sum(a == b for a, b in pairs)
    len_a = [len(a.split()) for a, _ in pairs]
    len_b = [len(b.split()) for _, b in pairs]
    diverge = []
    for i, (a, b) in enumerate(pairs):
        if a == b:
            diverge.append(None)
            continue
        j = next((k for k, (ca, cb) in enumerate(zip(a, b)) if ca != cb),
                 min(len(a), len(b)))
        diverge.append(j)
    return {"n": n, "exact_match": exact,
            "mean_words_a": round(sum(len_a) / n, 1) if n else 0,
            "mean_words_b": round(sum(len_b) / n, 1) if n else 0,
            "length_ratio": round(sum(len_b) / max(1, sum(len_a)), 2),
            "first_divergence_chars": diverge}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--brainscope-url", required=True)
    ap.add_argument("--hotwire-url", required=True)
    ap.add_argument("--model-b", default="steered", help="model name on brainscope")
    ap.add_argument("--model-h", required=True, help="model name on hotwire vLLM")
    ap.add_argument("--spec", required=True, help="steering spec, JSON")
    ap.add_argument("--prompts", default=None,
                    help="file with one prompt per line (default: built-ins)")
    args = ap.parse_args(argv)
    spec = json.loads(args.spec)
    prompts = ([p.strip() for p in open(args.prompts) if p.strip()]
               if args.prompts else NEUTRAL_PROMPTS)

    for label, use_spec in (("UNSTEERED", None), ("STEERED", spec)):
        pairs = []
        for p in prompts:
            a = ask(args.brainscope_url, args.model_b, p, use_spec)
            b = ask(args.hotwire_url, args.model_h, p, use_spec)
            pairs.append((a, b))
        rep = compare(pairs)
        print(f"[{label}] n={rep['n']} exact={rep['exact_match']} "
              f"words brainscope={rep['mean_words_a']} hotwire={rep['mean_words_b']} "
              f"ratio={rep['length_ratio']}")
    print("parity: compare the STEERED length ratio and behavior by eye — "
          "cross-backend token equality is not expected, drifting effect is.")


if __name__ == "__main__":
    main()
