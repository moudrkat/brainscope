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


# Prompts that make the model call a tool, so `syntax_mute` has something to
# act on. The neutral prompts above can never exercise it -- it only engages
# inside a tool call, which is exactly why the drift below went unnoticed.
TOOL_PROMPTS = [
    "Book a table for four at eight tonight.",
    "Remind me to call the dentist on Tuesday morning.",
    "Add oat milk and coffee to the shopping list.",
    "Schedule a follow-up with Petra next week.",
]
TOOL_SCHEMA = [{"type": "function", "function": {
    "name": "create_item",
    "description": "Create a scheduled item for the user.",
    "parameters": {"type": "object",
                   "properties": {"title": {"type": "string"},
                                  "when": {"type": "string"}},
                   "required": ["title", "when"]}}}]


def ask(base_url: str, model: str, prompt: str, spec, timeout: int = 300,
        tools=None) -> str:
    body = {"model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 120, "temperature": 0}
    if tools:
        body["tools"] = tools
        body["tool_choice"] = "auto"
    if spec is not None:
        body["vllm_xargs"] = {"hotwire": json.dumps(spec)}
    req = urllib.request.Request(
        base_url.rstrip("/") + "/v1/chat/completions",
        json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.loads(r.read())
    msg = d["choices"][0]["message"]
    # here a tool call IS the answer, so fold it into the compared text
    calls = msg.get("tool_calls") or []
    return (json.dumps(calls, sort_keys=True, ensure_ascii=False) if calls
            else (msg.get("content") or ""))


def honours_syntax_mute(base_url: str, model: str, spec) -> tuple:
    """Does this backend act on ``syntax_mute`` at all?

    Send the same steered tool-call request twice, once with the guard on and
    once off. A backend that implements the flag answers differently; one that
    drops the key answers identically twice.

    Deliberately a self-check, not a cross-backend diff: token equality across
    backends is never expected, but a backend disagreeing with *itself* under
    two specs that differ in one boolean is unambiguous.

    Why this exists: hotwire-vLLM parses the spec into
    ``SteerSpec(vector_id, layer, scale)`` and drops every other key, so
    sending ``syntax_mute`` there is silently a no-op -- the guard that keeps
    the function name and the JSON keys out of the steered span lives only in
    brainscope, and nothing errored to say so. Found by reading wire.py; this
    turns it into a test.

    Returns (differing, total).
    """
    def with_flag(value):
        first = dict(spec[0]) if isinstance(spec, list) else dict(spec)
        first["syntax_mute"] = value
        return [first] if isinstance(spec, list) else first

    differ = 0
    for p in TOOL_PROMPTS:
        on = ask(base_url, model, p, with_flag(True), tools=TOOL_SCHEMA)
        off = ask(base_url, model, p, with_flag(False), tools=TOOL_SCHEMA)
        differ += on != off
    return differ, len(TOOL_PROMPTS)


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

    # The regime flags are the part that drifts silently, because a backend
    # that does not implement one just ignores it. Ask each side directly.
    print()
    for name, url, model in (("brainscope", args.brainscope_url, args.model_b),
                             ("hotwire   ", args.hotwire_url, args.model_h)):
        try:
            d, n = honours_syntax_mute(url, model, spec)
        except Exception as e:  # noqa: BLE001 - a dead backend is a result
            print(f"[syntax_mute] {name}: could not test — {type(e).__name__}: {e}")
            continue
        verdict = ("honoured" if d else
                   "IGNORED — the key is dropped, silently, with no error")
        print(f"[syntax_mute] {name}: {verdict}  ({d}/{n} answers differ "
              f"between guard on and off)")


if __name__ == "__main__":
    main()
