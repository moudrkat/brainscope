"""Trace persistence and reasoning-trace analytics for brainscope.

A *trace* is one finished generation with everything the instruments saw:
tokens, per-layer norms, logit-lens and J-lens readouts, steering state,
think/answer segmentation — one JSON file per generation in --traces DIR.
Optionally (heavy, off by default) the raw per-token hidden states are kept
next to it ({id}.hidden.pt, fp16 [steps, n_layers, hidden]), which unlocks
exact post-hoc analytics like the answer-emergence curve.

Answer emergence: for the token that eventually opens the final answer,
how probable was it — at every reasoning step, best layer — under each
lens? The logit lens says when the model would *say* it; the J-lens
(Anthropic, "A global workspace in language models", 2026 —
transformer-circuits.pub/2026/workspace) says when it started *holding* it;
the A-lens is brainscope's answer-targeted variant (see jlens.py). Where
hidden states were not stored, the curve falls back to the recorded top-k
readouts (a lower bound: probability counts only when the token made top-k).
"""

import json
import re
import time
import uuid
from pathlib import Path

import torch

THINK_RE = re.compile(r"<think>(.*?)</think>", re.S)


def think_span(all_tokens: list[str]) -> list[int] | None:
    """[start, end] indices into all_tokens of the <think>…</think> block
    (end = index of the token that completes </think>), or None."""
    text, starts = "", []
    for t in all_tokens:
        starts.append(len(text))
        text += t
    m = THINK_RE.search(text)
    if not m:
        return None

    def tok_at(char: int) -> int:
        for i in range(len(starts) - 1, -1, -1):
            if starts[i] <= char:
                return i
        return 0

    return [tok_at(m.start()), tok_at(m.end() - 1)]


class TraceStore:
    """One JSON per generation in a directory; in-memory index for listing."""

    def __init__(self, root: Path, keep: int = 200):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.keep = keep
        self.index: list[dict] = []
        for f in sorted(self.root.glob("*.json")):
            try:
                t = json.loads(f.read_text())
                self.index.append(self._entry(t))
            except (json.JSONDecodeError, KeyError):
                continue

    def _entry(self, t: dict) -> dict:
        text = "".join(t.get("all_tokens", []))
        return {"id": t["id"], "ts": t.get("ts"), "model": t.get("model"),
                "n_tokens": len(t.get("all_tokens", [])), "think": t.get("think"),
                "steer": t.get("steer"), "tags": t.get("tags") or {},
                "has_hidden": t.get("has_hidden", False),
                "has_lens": bool(t.get("lens")), "has_jlens": bool(t.get("jlens")),
                "preview": text.strip().replace("\n", " ")[:120]}

    def save(self, gen: dict, model: str, hidden: list | None = None) -> dict:
        all_tokens = gen.get("all_tokens", gen.get("tokens", []))
        trace = {"id": gen["id"], "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
                 "model": model, "tags": gen.get("tags") or {}, "steer": gen.get("steer"),
                 "n_prompt": gen.get("n_prompt"),
                 "prompt_tail": "".join(gen.get("prompt_tokens", [])[-400:]),
                 "all_tokens": all_tokens,
                 # captured signals lag all_tokens by (len(all_tokens) - len(tokens))
                 # steps: the first token comes straight out of prefill, uncaptured
                 "capture_offset": len(all_tokens) - len(gen.get("tokens", [])),
                 "tokens": gen.get("tokens", []), "norms": gen.get("norms", []),
                 "lens": gen.get("lens", []), "jlens": gen.get("jlens", []),
                 "think": think_span(all_tokens),
                 "has_hidden": bool(hidden)}
        (self.root / f"{trace['id']}.json").write_text(
            json.dumps(trace, ensure_ascii=False))
        if hidden:
            torch.save(torch.stack(hidden).to(torch.float16),
                       self.root / f"{trace['id']}.hidden.pt")
        self.index = [e for e in self.index if e["id"] != trace["id"]]
        self.index.append(self._entry(trace))
        while len(self.index) > self.keep:
            self.delete(self.index[0]["id"])
        return trace

    def list(self) -> list[dict]:
        return list(reversed(self.index))

    def load(self, trace_id: str) -> dict | None:
        f = self.root / f"{trace_id}.json"
        if not f.exists() or not _safe_id(trace_id):
            return None
        return json.loads(f.read_text())

    def hidden(self, trace_id: str) -> torch.Tensor | None:
        f = self.root / f"{trace_id}.hidden.pt"
        if not f.exists() or not _safe_id(trace_id):
            return None
        return torch.load(f, map_location="cpu", weights_only=True)

    def delete(self, trace_id: str) -> bool:
        if not _safe_id(trace_id):
            return False
        found = False
        for suffix in (".json", ".hidden.pt"):
            f = self.root / f"{trace_id}{suffix}"
            if f.exists():
                f.unlink()
                found = True
        self.index = [e for e in self.index if e["id"] != trace_id]
        return found


def _safe_id(trace_id: str) -> bool:
    return bool(re.fullmatch(r"[0-9a-f]{6,32}", trace_id))


def answer_token(trace: dict, tokenizer=None, override: str | None = None) -> tuple[int, str] | None:
    """(index into all_tokens, piece) of the token that opens the final
    answer: first non-whitespace token after </think> (or the first answer
    token when there is no think block). `override` picks a different piece
    to track, matched anywhere after the think block."""
    toks = trace.get("all_tokens") or []
    start = trace["think"][1] + 1 if trace.get("think") else 0
    if override:
        want = override.strip()
        for i in range(start, len(toks)):
            if want and toks[i].strip() == want:
                return i, toks[i]
        return None
    for i in range(start, len(toks)):
        if toks[i].strip():
            return i, toks[i]
    return None


@torch.no_grad()
def emergence(trace: dict, hidden: torch.Tensor | None, *, tokenizer, norm, head,
              jlens=None, override: str | None = None) -> dict:
    """Answer-emergence curve: p(answer token), max over layers, per captured
    step, under each available lens. Exact when hidden states were stored;
    top-k lower bound from the recorded readouts otherwise."""
    at = answer_token(trace, tokenizer, override)
    if at is None:
        return {"error": "no answer token found"}
    a_idx, piece = at
    ids = tokenizer(piece, add_special_tokens=False).input_ids
    if not ids:
        return {"error": f"cannot tokenize {piece!r}"}
    a_id = ids[0]
    off = trace.get("capture_offset", 1)
    n_steps = len(trace.get("tokens", []))
    series: dict[str, list] = {}

    def topk_series(readouts):   # lower bound from stored top-k entries
        out = []
        for step in readouts:
            best = 0.0
            for layer in step or []:
                for e in layer:
                    if e["t"] == piece and e["p"] > best:
                        best = e["p"]
            out.append(round(best, 4))
        return out

    if trace.get("lens"):
        series["logit_lens_topk"] = topk_series(trace["lens"])
    if trace.get("jlens"):
        series["jlens_topk"] = topk_series(trace["jlens"])

    if hidden is not None and norm is not None and head is not None:
        dtype = next(head.parameters()).dtype
        hs = hidden.to(dtype)                       # [steps, n_layers, d]
        exact = []
        for s in range(hs.shape[0]):
            probs = torch.softmax(head(norm(hs[s])).float(), dim=-1)   # [n_layers, vocab]
            exact.append(round(float(probs[:, a_id].max()), 4))
        series["logit_lens"] = exact
        if jlens is not None:
            exact_j = []
            for s in range(hs.shape[0]):
                z = jlens.transport(hs[s].float()).to(dtype)
                probs = torch.softmax(head(norm(z)).float(), dim=-1)
                exact_j.append(round(float(probs[:, a_id].max()), 4))
            series["jlens"] = exact_j

    def first_over(xs, thr=0.1):
        for i, x in enumerate(xs):
            if x >= thr:
                return i
        return None

    return {"token": piece, "token_index": a_idx, "capture_offset": off,
            "n_steps": n_steps, "think": trace.get("think"),
            "series": series, "exact": hidden is not None,
            "emerge_step": {k: first_over(v) for k, v in series.items()}}
