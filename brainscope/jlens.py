"""Jacobian lens (J-lens) and answer lens (A-lens) for brainscope.

Independent reimplementation of the Jacobian lens introduced in:
  Gurnee, Sofroniew, Pearce, Piotrowski, Kauvar, Chen, Soligo, Bogdan, Ong,
  Wang, Thompson, Abrahams, Kantamneni, Ameisen, Batson & Lindsey (Anthropic),
  "Verbalizable Representations Form a Global Workspace in Language Models",
  Transformer Circuits Thread, July 2026.
  paper      https://transformer-circuits.pub/2026/workspace/index.html
  announce   https://www.anthropic.com/research/global-workspace
  reference  https://github.com/anthropics/jacobian-lens (Apache-2.0)
  (full citation + BibTeX: docs/jlens.md#citing)
No code is copied from the reference implementation; the method is
re-derived from the paper's description so brainscope stays MIT and
dependency-free.

The idea: for every decoder layer l estimate the *averaged Jacobian*

    J_l = E[ dh_final(t') / dh_l(t) ]      over prompts, source positions t
                                           and target positions t' >= t,
then read an activation h out as

    lens_l(h) = softmax(W_U · final_norm(J_l · h))

— the logit lens with a learned linear transport into final-layer space. A
plain logit lens asks "what would the model say if it stopped HERE"; the
J-lens asks "what is this activation disposed to make the model say LATER".
The vocabulary directions it picks out are what the paper calls J-space:
representations that push toward future output before any of it is emitted.

The ANSWER lens (mode="answer") is brainscope's own experimental variant:
target positions are restricted to tokens after a marker (default
"</think>"), so J_l measures causal influence on the eventual *answer*
only, ignoring influence on the verbal reasoning in between. Fit it on
reasoning traces; compare against the J-lens in the traces panel. It
*illustrates* where the answer forms — measuring that properly needs the
validation in the emergence view.

Estimator (one backward per probe): draw u ~ N(0, I) at every target
position, backprop s = sum_t' u(t')·h_final(t'). The gradient at (l, t) is
g = sum_{t'>=t} (J^(t->t'))ᵀ u(t'); since E[u uᵀ] = I, the outer product
outer(sum u, sum_t g) is an unbiased (rank-1, high-variance) sample of the
position-summed Jacobian. Averaging over prompts × repeats converges; keep
samples comfortably above d_model. Normalization is per target position, so
J at the final layer converges to the identity — a built-in self-test.
Absolute scale is cosmetic anyway: the readout re-normalizes (RMSNorm).
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch

FORMAT_VERSION = 1


class JacobianLens:
    """A fitted lens: J [n_layers, d, d] plus fit metadata."""

    def __init__(self, J: torch.Tensor, meta: dict):
        self.J = J
        self.meta = meta

    @property
    def n_layers(self) -> int:
        return self.J.shape[0]

    @property
    def hidden(self) -> int:
        return self.J.shape[-1]

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"J": self.J.to(torch.float16).cpu(), "meta": self.meta}, path)

    @classmethod
    def load(cls, path: Path, device: str = "cpu", dtype=torch.float32) -> "JacobianLens":
        art = torch.load(Path(path), map_location="cpu", weights_only=True)
        return cls(art["J"].to(device=device, dtype=dtype), art.get("meta", {}))

    @torch.no_grad()
    def transport(self, hs: torch.Tensor) -> torch.Tensor:
        """Map per-layer activations into final-layer space.

        hs: [n_layers, d] (one position, every layer) -> [n_layers, d]."""
        return torch.einsum("lij,lj->li", self.J, hs.to(self.J.dtype))

    @torch.no_grad()
    def direction(self, token_id: int, unembed_weight: torch.Tensor) -> torch.Tensor:
        """Steering direction for one vocabulary token: the activation
        pattern at each layer whose growth most increases the token's
        *future* logit, to first order: J_lᵀ · W_U[v]. Rows are unit-
        normalized so the steer-strength slider means the same thing as for
        other directions. (The final RMSNorm between J·h and W_U is treated
        as a positive scalar — direction, not magnitude.)"""
        w = unembed_weight[token_id].to(self.J.device, self.J.dtype)  # [d]
        dirs = torch.einsum("lji,j->li", self.J, w)                   # [n_layers, d]
        return dirs / (dirs.norm(dim=-1, keepdim=True) + 1e-8)

    @torch.no_grad()
    def decompose(self, hs: torch.Tensor, layer: int, unembed_weight: torch.Tensor,
                  k: int = 16, method: str = "gp") -> list[dict]:
        """EXPERIMENTAL J-space decomposition: express each state as a sparse
        NONNEGATIVE combination of k J-lens vectors {J_lᵀ·W_U[v]/‖·‖} — "which
        word-linked patterns, added together, make up this state?" Unlike the
        top-k readout (a softmax ranking where strong components eclipse weak
        ones), every selected component gets its own additive coefficient.

        Follows the paper's recipe: "we solve for a sparse nonnegative
        combination of k J-lens vectors ... using gradient pursuit"; they use
        k ≤ 25 (16 for verbal-report decomposition) and report that the
        J-space component carries no more than ~10% of activation variance —
        `explained` in the output lets you verify that caveat yourself.

        method="gp": gradient pursuit (select by max positive correlation,
        then one exact-line-search gradient step on the active set, projected
        to c ≥ 0 — Blumensath & Davies 2008 flavour).
        method="mp": plain matching pursuit (kept for comparison).

        hs: [steps, d] residuals at `layer` (stored trace hidden states).
        Returns per step: {"components": [(token_id, coeff), ...] sorted by
        coeff desc, "explained": fraction of squared norm captured}."""
        assert method in ("gp", "mp"), method
        J = self.J[layer]                                    # [d, d]
        # big tensors stay in the unembedding's native dtype (bf16 on GPU) —
        # the fp32 copies would add ~3 GB next to a 4B model and OOM a 16 GB
        # card; per-step math below is small and runs in fp32
        W = unembed_weight.to(J.device)                      # [vocab, d]
        JW = W @ J.to(W.dtype)                               # rows = w_vᵀ J_l  [vocab, d]
        norms = JW.norm(dim=1).float().clamp_min(1e-6)
        H = hs.to(J.device, torch.float32)                   # [steps, d]
        n_steps = H.shape[0]
        R = H.clone()
        active = torch.full((n_steps, k), -1, dtype=torch.long, device=H.device)
        coeffs = torch.zeros(n_steps, k, dtype=torch.float32, device=H.device)
        for it in range(k):
            corr = (R.to(JW.dtype) @ JW.T).float() / norms   # [steps, vocab]
            for s in range(n_steps):                         # no atom twice
                prev = active[s, :it]
                corr[s, prev[prev >= 0]] = -torch.inf
            best = corr.argmax(dim=1)                        # nonneg: max positive
            active[:, it] = best
            for s in range(n_steps):
                A = active[s, : it + 1]
                D = JW[A].float() / norms[A, None]           # [it+1, d] unit atoms
                if method == "mp":
                    c = float(R[s] @ D[-1])
                    if c > 0:
                        coeffs[s, it] = c
                        R[s] -= c * D[-1]
                    continue
                # gradient pursuit: one exact-line-search step on the active set
                g = D @ R[s]                                 # gradient wrt c_A
                Dg = g @ D                                   # direction in h-space
                denom = float(Dg @ Dg)
                if denom < 1e-12:
                    continue
                alpha = float(R[s] @ Dg) / denom
                c_new = (coeffs[s, : it + 1] + alpha * g).clamp_min(0)   # project c >= 0
                coeffs[s, : it + 1] = c_new
                R[s] = H[s] - c_new @ D
        out = []
        h_sq = (H * H).sum(dim=1).clamp_min(1e-9)
        r_sq = (R * R).sum(dim=1)
        for s in range(n_steps):
            comps = [(int(active[s, i]), round(float(coeffs[s, i]), 4))
                     for i in range(k) if coeffs[s, i] > 1e-6]
            comps.sort(key=lambda x: -x[1])
            out.append({"components": comps,
                        "explained": round(float(1 - r_sq[s] / h_sq[s]), 4)})
        return out

    def identity_error(self) -> float:
        """Relative error of the final layer's J against the identity — the
        estimator's built-in self-test (should be well under 1 after a
        healthy fit; large values mean too few samples)."""
        eye = torch.eye(self.hidden, dtype=self.J.dtype)
        return float((self.J[-1].cpu() - eye).norm() / eye.norm())


# Positions before this index are excluded from the Jacobian average on both
# the source and target side — early positions act as attention sinks with
# atypical residual statistics, and the final position has no next-token
# target. Matches the reference implementation's reduction (verified against
# jlens/fitting.py by reading, 2026-07-11; their SKIP_FIRST_N_POSITIONS=16).
SKIP_FIRST_POSITIONS = 16


def _valid_mask(seq: int, skip_first: int) -> torch.Tensor:
    """Bool [seq]: positions included in the Jacobian average (source side)."""
    mask = torch.zeros(seq, dtype=torch.bool)
    mask[min(skip_first, max(0, seq - 2)): seq - 1] = True
    return mask


def _target_mask(text: str, tokenizer, ids: torch.Tensor, mode: str, marker: str,
                 skip_first: int = SKIP_FIRST_POSITIONS) -> torch.Tensor:
    """Bool [seq]: which positions count as lens targets t'."""
    seq = ids.shape[-1]
    mask = _valid_mask(seq, skip_first)
    if mode == "answer":   # only positions after the marker; last quarter fallback
        cut = text.find(marker)
        if cut >= 0:
            prefix = tokenizer(text[: cut + len(marker)], return_tensors="pt").input_ids
            start = min(seq - 1, prefix.shape[-1])
        else:
            start = int(seq * 0.75)
        mask[:start] = False
    return mask


def fit(model, tokenizer, texts, *, mode: str = "future", marker: str = "</think>",
        repeats: int = 16, max_tokens: int = 128, seed: int = 0,
        skip_first: int = SKIP_FIRST_POSITIONS, progress=None) -> JacobianLens:
    """Fit a Jacobian lens on raw texts. See module docstring for the math.

    Cost: len(texts) × repeats forward+backward passes of <= max_tokens.
    Keep len(texts) × repeats comfortably above the model's hidden size."""
    assert mode in ("future", "answer"), mode
    device = next(model.parameters()).device
    torch.manual_seed(seed)
    was_training = model.training
    model.eval()

    acc = None          # [n_layers, d, d] fp32 accumulator (on CPU to spare VRAM)
    count = 0.0         # total target positions seen
    n_samples = 0
    t0 = time.time()
    for pi, text in enumerate(texts):
        ids = tokenizer(text, return_tensors="pt", truncation=True,
                        max_length=max_tokens).input_ids.to(device)
        if ids.shape[-1] < 8:
            continue
        seq = ids.shape[-1]
        tmask = _target_mask(text, tokenizer, ids, mode, marker, skip_first).to(device)
        smask = _valid_mask(seq, skip_first).to(device)   # source-side positions
        if not bool(tmask.any()):
            continue
        with torch.enable_grad():
            out = model(input_ids=ids, output_hidden_states=True, use_cache=False)
            hs = out.hidden_states[1:]          # per-layer outputs, [1, seq, d] each
            h_final = hs[-1]
            for r in range(repeats):
                u = torch.randn_like(h_final) * tmask[None, :, None]
                s = (u * h_final).sum()
                grads = torch.autograd.grad(s, hs, retain_graph=r < repeats - 1)
                u_sum = u[0].sum(0).float().cpu()                     # [d]
                if acc is None:
                    acc = torch.zeros(len(hs), u_sum.shape[0], u_sum.shape[0])
                for l, g in enumerate(grads):
                    acc[l] += torch.outer(u_sum, g[0][smask].sum(0).float().cpu())
                count += float(tmask.sum())
                n_samples += 1
        del out, hs, h_final
        if device.type == "cuda":
            torch.cuda.empty_cache()
        if progress and (pi + 1) % 10 == 0:
            progress(f"jlens fit: {pi + 1}/{len(texts)} texts · {n_samples} samples "
                     f"· {time.time() - t0:.0f}s")

    if acc is None:
        raise ValueError("no usable texts (need >= 8 tokens each)")
    J = acc / max(count, 1.0)
    meta = {"model": getattr(getattr(model, "config", None), "_name_or_path", "?"),
            "mode": mode, "marker": marker if mode == "answer" else None,
            "n_texts": len(texts), "repeats": repeats, "n_samples": n_samples,
            "max_tokens": max_tokens, "seed": seed, "skip_first": skip_first,
            "format": FORMAT_VERSION,
            "fitted_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    lens = JacobianLens(J, meta)
    meta["identity_error"] = round(lens.identity_error(), 4)
    if was_training:
        model.train()
    return lens


# ---------------------------------------------------------------- CLI ----

def _load_texts(spec: str, n: int) -> list[str]:
    path = Path(spec)
    if path.exists():
        texts = []
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                texts.append(row["text"] if isinstance(row, dict) else str(row))
            except json.JSONDecodeError:
                texts.append(line)   # plain-text file, one snippet per line
        return texts[:n]
    if spec == "wikitext":   # pretraining-like distribution, as in the paper
        try:
            from datasets import load_dataset
        except ImportError:
            sys.exit("--prompts wikitext needs `pip install datasets` "
                     "(or pass a JSONL file of {\"text\": ...} lines)")
        ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="train", streaming=True)
        texts, buf = [], ""
        for row in ds:
            buf += row["text"]
            if len(buf) > 600:            # ~128+ tokens per snippet
                texts.append(buf)
                buf = ""
            if len(texts) >= n:
                break
        return texts
    sys.exit(f"prompts source {spec!r}: not a file and not a known dataset name")


# a handful of built-in questions for gen-traces (A-lens fitting data);
# short multi-step problems so the model produces a <think> block worth
# analyzing, with a clearly-delimited final answer
BUILTIN_QUESTIONS = [
    "A train leaves at 9:40 and arrives at 12:05. How long is the trip in minutes?",
    "I buy 3 notebooks at 45 CZK and pay with a 200 CZK note. What change do I get?",
    "Which is heavier: 2.5 kg of feathers or 2400 g of iron? Explain briefly.",
    "If today is Wednesday, what day is it in 100 days?",
    "A rectangle is 7 cm by 12 cm. What is its perimeter and area?",
    "Anna is twice as old as Ben. Together they are 36. How old is Ben?",
    "What is the next number: 2, 6, 12, 20, 30, ...?",
    "A shirt costs 800 CZK after a 20% discount. What was the original price?",
    "Three friends split a 741 CZK bill evenly. How much does each pay?",
    "Water boils at 100°C. Is 373 K hotter, colder, or the same? Why?",
    "How many legs do 7 spiders and 4 birds have in total?",
    "A car does 6.2 l/100 km. How many liters for a 250 km trip?",
    "Which fraction is larger: 3/7 or 4/9? Show the comparison.",
    "A clock shows 3:15. What is the angle between the hands?",
    "If 5 machines make 5 parts in 5 minutes, how long do 100 machines need for 100 parts?",
    "What is 17 × 23, computed step by step?",
    "A password must be 3 letters then 2 digits. How many are possible with 26 letters?",
    "I walk 4 km/h for 45 minutes, then 6 km/h for 30 minutes. Total distance?",
    "Is 2027 a prime number? Check it.",
    "A recipe for 4 people needs 300 g flour. How much for 7 people?",
]


def _cmd_fit(args) -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from .server import PRESETS
    model_id = PRESETS.get(args.model, args.model)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"jlens: loading {model_id} on {device} …", flush=True)
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    tok = AutoTokenizer.from_pretrained(model_id)
    # sdpa attention: fitting needs gradients, not attention weights — take the speed
    kwargs = {"torch_dtype": dtype, "attn_implementation": "sdpa"}
    if getattr(args, "quantize", None):  # input-grads flow through bnb linears
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = (
            BitsAndBytesConfig(load_in_8bit=True) if args.quantize == "8bit"
            else BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=dtype))
        kwargs["device_map"] = device
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    if not getattr(args, "quantize", None):
        model = model.to(device)
    texts = _load_texts(args.prompts, args.n_texts)
    hidden = model.config.hidden_size
    if len(texts) * args.repeats < 2 * hidden:
        print(f"jlens: WARNING — {len(texts)} texts × {args.repeats} repeats = "
              f"{len(texts) * args.repeats} samples for hidden size {hidden}; "
              "the estimate will be noisy (aim for >= 2×hidden)", flush=True)
    lens = fit(model, tok, texts, mode=args.mode, marker=args.marker,
               repeats=args.repeats, max_tokens=args.max_tokens, seed=args.seed,
               skip_first=args.skip_first,
               progress=lambda m: print(m, flush=True))
    lens.save(args.out)
    print(f"jlens: saved {args.out} · layers {lens.n_layers} · hidden {lens.hidden} "
          f"· identity_error {lens.meta['identity_error']} (final-layer self-test, "
          "≪ 1 is healthy)", flush=True)


def _cmd_gen_traces(args) -> None:
    """Collect reasoning traces from a running brainscope/OpenAI server into a
    JSONL ready for `fit --mode answer`. Uses <think> models' raw text."""
    import urllib.request
    questions = (Path(args.questions).read_text().splitlines()
                 if args.questions != "builtin" else BUILTIN_QUESTIONS)
    rows = []
    for i, q in enumerate([q for q in questions if q.strip()]):
        body = json.dumps({"model": "any", "messages": [{"role": "user", "content": q}],
                           "max_tokens": args.max_tokens, "raw": True}).encode()
        req = urllib.request.Request(args.base_url.rstrip("/") + "/chat/completions",
                                     data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=600) as r:
            resp = json.load(r)
        text = resp["choices"][0]["message"].get("raw_content") \
            or resp["choices"][0]["message"].get("content") or ""
        rows.append({"text": q + "\n" + text})
        print(f"gen-traces: {i + 1} done ({len(text)} chars)", flush=True)
    Path(args.out).write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows))
    print(f"gen-traces: wrote {len(rows)} traces to {args.out}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("fit", help="fit a J-lens (or A-lens) for a model")
    f.add_argument("--model", required=True, help="HF id or brainscope preset")
    f.add_argument("--prompts", default="wikitext",
                   help="JSONL/plain file of texts, or 'wikitext' (needs datasets)")
    f.add_argument("--out", type=Path, required=True)
    f.add_argument("--mode", choices=["future", "answer"], default="future",
                   help="future = J-lens (all later tokens) · answer = A-lens "
                        "(only tokens after --marker; fit on reasoning traces)")
    f.add_argument("--marker", default="</think>")
    f.add_argument("--n-texts", type=int, default=256)
    f.add_argument("--repeats", type=int, default=16, help="probes per text")
    f.add_argument("--max-tokens", type=int, default=128)
    f.add_argument("--device", default=None)
    f.add_argument("--quantize", choices=["8bit", "4bit"], default=None)
    f.add_argument("--seed", type=int, default=0)
    f.add_argument("--skip-first", type=int, default=SKIP_FIRST_POSITIONS,
                   help="exclude the first N positions from the average "
                        "(attention sinks; matches the reference implementation)")
    f.set_defaults(func=_cmd_fit)

    g = sub.add_parser("gen-traces",
                       help="collect reasoning traces from a running server "
                            "into JSONL for `fit --mode answer`")
    g.add_argument("--base-url", default="http://localhost:8010/v1")
    g.add_argument("--questions", default="builtin", help="'builtin' or a text file, one per line")
    g.add_argument("--out", type=Path, required=True)
    g.add_argument("--max-tokens", type=int, default=768)
    g.set_defaults(func=_cmd_gen_traces)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
