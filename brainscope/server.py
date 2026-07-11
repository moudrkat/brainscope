"""brainscope — watch your model think while your app talks to it.

An OpenAI-compatible chat-completions server over any Hugging Face causal LM
that streams per-token, per-layer residual-stream activity to a live browser
visualization. Point your app's OpenAI base_url at it; open http://host:port/
in a window next to your app.

    python -m brainscope.server --model Qwen/Qwen2.5-0.5B-Instruct --port 8010

Optional steering directions: --directions takes either a dirs.json mapping
{"name": vector or [n_layers, hidden] matrix, ...} or a hidden-directions
direction_dict/ folder (manifest.json + *.pt). Loaded directions can be
applied live (activation addition, globally / per request / by policy), and
every generated token reports its per-layer cosine with each named direction.
"""

import argparse
import asyncio
import base64
import gc
import math
import threading
import json
import re
import time
import uuid
from pathlib import Path

# must be set before torch initializes CUDA: lets the allocator grow segments
# instead of fragmenting fixed pools (long-prompt prefill + KV-cache churn)
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse

from .jlens import JacobianLens
from .traces import TraceStore, emergence as compute_emergence

STATIC = Path(__file__).parent / "static"

# friendly shortcuts -> HF ids (extend freely)
PRESETS = {
    "qwen3-4b": "Qwen/Qwen3-4B-Instruct-2507",
    "qwen3-8b": "Qwen/Qwen3-8B",
    "qwen3.5-9b": "Qwen/Qwen3.5-9B",
    "gemma-e4b": "google/gemma-4-E4B-it",
    "tiny": "Qwen/Qwen2.5-0.5B-Instruct",  # CPU-friendly demo
}

app = FastAPI(title="brainscope")
state: dict = {"model": None, "tokenizer": None, "directions": {}, "dir_meta": {}, "clients": set(),
               "loop": None, "device": "cpu", "model_name": "",
               "steer": None, "steer_handles": [], "policy": [], "policy_on": True,
               "gen": None, "probes": {"attn": {}, "mlp": {}}, "lens": False,
               "viz": True,
               # J-lens (Jacobian lens) — see jlens.py; loaded via --jlens,
               # per-token readout toggleable live (POST /jlens)
               "jlens": None, "jlens_on": False,
               # trace persistence — TraceStore via --traces; hidden-state
               # capture is the heavy part and stays off until asked
               "traces": None, "save_traces": True, "save_hidden": False}

# hidden-state capture safety valve: at most this many steps kept per trace
# (a 9B model's 4k-token trace would otherwise stack >1 GB of fp16)
HIDDEN_MAX_STEPS = int(os.getenv("HIDDEN_MAX_STEPS", "2048"))

# Prefill batch size: bounds eager attention's transient chunk×seq matrix.
# Eager softmax upcasts to fp32, so the transient is chunk × seq × heads × 4 B
# — at 128 that's ~0.4 GB for a 24k prompt, safe next to model + KV cache.
PREFILL_CHUNK = 128
# short prompts (<= this many tokens) get their FULL per-head seq×seq prefill
# attention captured for the matrix viz — "peek into the model". Longer prompts
# skip it (a 20k-token seq×seq×heads matrix would be gigabytes).
ATTN_MATRIX_MAX = int(os.getenv("ATTN_MATRIX_MAX", "160"))

# Tool-call output formats differ per model family; try each in order.
TOOL_CALL_PATTERNS = [
    re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S),          # qwen/hermes
    re.compile(r"```tool_(?:call|code)\s*(\{.*?\})\s*```", re.S),         # gemma-style fenced
    re.compile(r"^\s*(\{\s*\"name\".*?\"arguments\".*?\})\s*$", re.S),  # bare JSON fallback
]


def load_model(name: str, device: str | None, quantize: str | None = None) -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    # eager attention so output_attentions really returns weights (sdpa/flash
    # never materialize them) — brainscope trades speed for sight everywhere
    kwargs = {"torch_dtype": torch.bfloat16 if dev == "cuda" else torch.float32,
              "attn_implementation": "eager"}
    if quantize:  # fit bigger models on a 16 GB card at some quality cost
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = (
            BitsAndBytesConfig(load_in_8bit=True) if quantize == "8bit"
            else BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16))
        kwargs["device_map"] = "auto"
    state["tokenizer"] = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, **kwargs)
    if not quantize:
        model = model.to(dev)
    state["model"] = model.eval()
    state["device"] = dev
    state["model_name"] = name
    _install_probe_hooks()


def _install_probe_hooks() -> None:
    """Forward hooks on every attention and MLP sublayer recording the L2 norm
    of their output at the last position — how hard each sublayer worked on
    the current token. Powers the live architecture view."""
    probes = state["probes"] = {"attn": {}, "mlp": {}}

    def make(kind: str, idx: int):
        def hook(_module, _inp, out):
            t = out[0] if isinstance(out, tuple) else out
            probes[kind][idx] = float(t[0, -1].float().norm())
        return hook

    for i, layer in enumerate(_decoder_layers(state["model"])):
        if hasattr(layer, "self_attn"):
            layer.self_attn.register_forward_hook(make("attn", i))
        if hasattr(layer, "mlp"):
            layer.mlp.register_forward_hook(make("mlp", i))


def _final_norm_and_head():
    model = state["model"]
    head = model.get_output_embeddings()
    for attr in ("model", "transformer"):
        core = getattr(model, attr, None)
        if core is not None:
            for nattr in ("norm", "ln_f", "final_layernorm"):
                norm = getattr(core, nattr, None)
                if norm is not None:
                    return norm, head
    return None, head


async def broadcast(payload: dict) -> None:
    dead = []
    for ws in state["clients"]:
        try:
            await ws.send_text(json.dumps(payload, ensure_ascii=False))
        except Exception:
            dead.append(ws)
    for ws in dead:
        state["clients"].discard(ws)


def _decoder_layers(model):
    for attr in ("model", "transformer"):
        core = getattr(model, attr, None)
        if core is not None and hasattr(core, "layers"):
            return core.layers
    raise RuntimeError("cannot locate decoder layers on this architecture")


def apply_bake(path: Path) -> None:
    """Patch a hidden-directions bake artifact (advocate_bias.pt) into the
    loaded model: the saved persona bias is added to one MLP down_proj output
    via a forward hook. A hook (rather than a real bias parameter) survives
    quantization, where bitsandbytes replaces the Linear module."""
    art = torch.load(path / "advocate_bias.pt", map_location="cpu", weights_only=True)
    if art.get("base_model") and art["base_model"] != state["model_name"]:
        print(f"brainscope: WARNING — bake artifact was made for {art['base_model']}, "
              f"but the loaded model is {state['model_name']}", flush=True)
    bias = art["bias"].to(state["device"])

    def hook(_module, _inp, out):
        t = out[0] if isinstance(out, tuple) else out
        t = t + bias.to(t.dtype)
        return (t, *out[1:]) if isinstance(out, tuple) else t

    _decoder_layers(state["model"])[art["layer"]].mlp.down_proj.register_forward_hook(hook)
    print(f"brainscope: bake applied at layer {art['layer']} — {art.get('note', path)}",
          flush=True)


def _install_steer_hooks(name: str, strength: float, layer_from: int, layer_to: int) -> list:
    """Register activation-addition hooks (h += strength * direction), return handles.

    A direction is either one vector [hidden] applied to every steered layer,
    or a per-layer matrix [n_layers, hidden] (e.g. a hidden-directions dict
    entry) where each steered layer gets its own row."""
    vec = state["directions"][name]

    def make_hook(row):
        def hook(_module, _inp, out):
            if state.get("steer_mute"):   # inside a tool call: syntax over persona
                return out
            hidden = out[0] if isinstance(out, tuple) else out
            hidden = hidden + strength * row.to(hidden.dtype)
            return (hidden, *out[1:]) if isinstance(out, tuple) else hidden
        return hook

    layers = _decoder_layers(state["model"])
    layer_to = min(layer_to if layer_to >= 0 else len(layers) - 1, len(layers) - 1)
    return [layers[i].register_forward_hook(
                make_hook(vec[min(i, vec.shape[0] - 1)] if vec.dim() == 2 else vec))
            for i in range(max(0, layer_from), layer_to + 1)]


def _normalize_steer(body) -> list:
    """One steering spec, {"stack": [specs]}, or a bare list — always returns
    a list of effective specs (unknown names and zero strengths drop out).
    Stacks compose like the bake recipes: e.g. pref @ 1.5 with refusal @ -1."""
    if not body:
        return []
    specs = body.get("stack", [body]) if isinstance(body, dict) else body
    out = []
    for s in specs if isinstance(specs, list) else []:
        name, strength = s.get("name"), float(s.get("strength") or 0)
        if name and strength != 0 and name in state["directions"]:
            out.append({"name": name, "strength": strength,
                        "layer_from": int(s.get("layer_from", 0)),
                        "layer_to": int(s.get("layer_to", -1))})
    return out


def _install_steer_stack(specs: list) -> tuple[list, list]:
    """Install hooks for every spec; returns (handles, applied-description)."""
    n = len(_decoder_layers(state["model"]))
    handles, applied = [], []
    for s in specs:
        handles += _install_steer_hooks(s["name"], s["strength"],
                                        s["layer_from"], s["layer_to"])
        lt = min(s["layer_to"] if s["layer_to"] >= 0 else n - 1, n - 1)
        applied.append({"name": s["name"], "strength": s["strength"],
                        "layers": [max(0, s["layer_from"]), lt]})
    return handles, applied


def apply_steering(body) -> dict:
    """(Re)install the GLOBAL steering hooks (the viz slider). Per-request
    steering (a `steering` object in a chat completions request) temporarily
    replaces these for the duration of that one generation."""
    for h in state["steer_handles"]:
        h.remove()
    state["steer_handles"] = []
    specs = _normalize_steer(body)
    if not specs:
        state["steer"] = None
        return {"active": False}
    state["steer_handles"], state["steer"] = _install_steer_stack(specs)
    return {"active": True, "steer": state["steer"]}


def _layer_signals(hidden_states, directions):
    """Per-layer L2 norms (+ cosines with named directions) for the last token."""
    norms, cos = [], {name: [] for name in directions}
    for i, h in enumerate(hidden_states[1:]):  # skip embedding layer
        v = h[0, -1].float()
        norms.append(float(v.norm()))
        for name, d in directions.items():
            row = d[min(i, d.shape[0] - 1)] if d.dim() == 2 else d
            cos[name].append(float(torch.nn.functional.cosine_similarity(v, row.float(), dim=0)))
    return norms, cos


def _stack_last(hidden_states) -> torch.Tensor:
    """Every layer's hidden state at the newest position: [n_layers, hidden]."""
    return torch.stack([h[0, -1] for h in hidden_states[1:]])


def _topk_readout(z: torch.Tensor, top: int = 5):
    """Final norm + lm_head over a [n_layers, hidden] stack already sitting
    in (or transported into) final-layer space — shared by both lenses."""
    norm, head = _final_norm_and_head()
    if norm is None or head is None:
        return None
    dtype = next(head.parameters()).dtype
    probs = torch.softmax(head(norm(z.to(dtype))).float(), dim=-1)
    p, idx = probs.topk(top, dim=-1)
    tok = state["tokenizer"]
    return [[{"t": tok.decode(int(idx[layer, k])), "p": round(float(p[layer, k]), 4)}
             for k in range(top)] for layer in range(z.shape[0])]


def _logit_lens(hs: torch.Tensor, top: int = 5):
    """What the model would say if it stopped at each layer: every layer's
    hidden state pushed through the final norm + lm_head. Watching the answer
    crystallize with depth is the point of the exercise."""
    return _topk_readout(hs, top)


def _jlens_readout(hs: torch.Tensor, top: int = 5):
    """What each layer is disposed to make the model say LATER: the hidden
    state transported into final-layer space by the fitted averaged Jacobian
    before the usual norm + lm_head readout (Anthropic 2026, see jlens.py).
    Silent concepts — a word lighting up here is on the model's mind, not
    necessarily in its mouth."""
    return _topk_readout(state["jlens"].transport(hs.float()), top)


def _capture_matrix(attentions, gen: dict) -> None:
    """Full per-head self-attention from a SHORT prompt's prefill. Stored per
    layer as uint8 [heads, seq, seq]; each query row is scaled to its own max so
    the pattern is visible (causal ⇒ lower-triangular). Cheap when seq is small."""
    if not attentions:
        return
    mats = []
    for a in attentions:                       # a: [1, heads, seq, seq]
        w = a[0].float()                        # [heads, seq, seq]
        w = w / (w.amax(dim=-1, keepdim=True) + 1e-9)
        mats.append((w * 255).round().to(torch.uint8).cpu().numpy())
    gen["matrix"] = mats


def _attn_signals(attentions, gen: dict):
    """Digest one decode step's attention weights.

    Stores full mean-over-heads rows (uint8) + last-step per-head detail in
    the generation record for the HTTP endpoints; returns lightweight per-layer
    summaries (entropy of the mean pattern, argmax position, per-head entropy)
    for the websocket stream.
    """
    entropy, top, head_entropy, mean_rows, head_rows = [], [], [], [], []
    for a in attentions:
        w = a[0, :, -1, :].float()          # [heads, seq]
        mean = w.mean(0)                    # [seq]
        log_seq = math.log(max(2, w.shape[-1]))
        mean_rows.append((mean * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy().tobytes())
        head_rows.append((w * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy().tobytes())
        he = -(w * (w + 1e-9).log()).sum(-1) / log_seq          # [heads], 0..1
        head_entropy.append([round(float(x), 3) for x in he])
        me = -(mean * (mean + 1e-9).log()).sum() / log_seq
        entropy.append(round(float(me), 3))
        top.append(int(mean.argmax()))
    gen["attn_rows"].append(mean_rows)
    gen["last_heads"] = head_rows           # only the newest token, per layer
    return entropy, top, head_entropy


def _match_policy(tags: dict) -> dict | None:
    """First steering-policy rule whose every `match` key equals the request's
    tag (case-insensitive); missing match keys are wildcards. Lets an app tag
    requests (OpenAI `metadata`: e.g. {"agent": "assistant", "phase":
    "chat"}) and keep all steering knowledge on the brainscope side."""
    if not state["policy_on"]:
        return None
    for rule in state["policy"]:
        match = rule.get("match", {})
        if all(str(tags.get(k, "")).lower() == str(v).lower() for k, v in match.items()):
            return rule.get("steer")
    return None


_GEN_LOCK = threading.Lock()  # one generation at a time — retries/parallel agents must queue


@torch.inference_mode()
def generate_with_signals(messages, tools, max_new_tokens, temperature, notify,
                          steering: dict | None = None, tags: dict | None = None,
                          tool_choice=None):
    """Token-by-token generation, calling notify(payload) per token.

    `steering` scopes activation addition to THIS request only: it overrides
    the global slider for the duration of the generation (including an
    explicit "no steering" via strength 0), then the global state is
    restored. This is how an app steers one agent without steering everyone
    else on the server.
    """
    with _GEN_LOCK:
        request_handles = []
        active_steer = state["steer"]
        if steering is not None:
            for h in state["steer_handles"]:
                h.remove()
            state["steer_handles"] = []
            specs = _normalize_steer(steering)
            if specs:
                request_handles, active_steer = _install_steer_stack(specs)
                for a in active_steer:
                    a["scope"] = "request"
            else:
                active_steer = None
        try:
            return _generate(messages, tools, max_new_tokens, temperature, notify,
                             active_steer, tags, tool_choice)
        finally:
            for h in request_handles:
                h.remove()
            if steering is not None and state["steer"]:
                state["steer_handles"], state["steer"] = _install_steer_stack(
                    [{"name": s["name"], "strength": s["strength"],
                      "layer_from": s["layers"][0], "layer_to": s["layers"][1]}
                     for s in state["steer"]])
            gc.collect()
            if state["device"] == "cuda":
                torch.cuda.empty_cache()


def _tool_scan_new() -> dict:
    return {"buf": "", "in_call": False, "stack": [], "in_str": False, "esc": False,
            "expect": "key", "key": "", "args_depth": None, "speak": True}


def _tool_scan(st: dict, text: str) -> None:
    """Advance the tool-call scanner over newly decoded text (mutates st).

    Drives the steering mute: outside a <tool_call> block everything is
    steered; inside one only the STRING VALUES under "arguments" may be
    steered — the persona talks through the call's content while the JSON
    scaffolding, the keys and the function name stay well-formed.
    st["speak"] is True whenever steering may apply."""
    for c in text:
        st["buf"] = (st["buf"] + c)[-16:]
        if not st["in_call"]:
            if st["buf"].endswith("<tool_call>"):
                st.update(in_call=True, stack=[], in_str=False, esc=False,
                          expect="key", key="", args_depth=None)
        elif st["buf"].endswith("</tool_call>"):
            st["in_call"] = False
        elif st["in_str"]:
            if st["esc"]:
                st["esc"] = False
            elif c == "\\":
                st["esc"] = True
            elif c == '"':
                st["in_str"] = False
                if st["expect"] == "key":
                    st["expect"] = "colon"
        else:
            if c == '"':
                st["in_str"] = True
                if st["expect"] == "key":
                    st["key"] = ""
            elif c == ":":
                if st["expect"] == "colon":
                    if st["key"] == "arguments" and st["args_depth"] is None:
                        st["args_depth"] = len(st["stack"])
                    st["expect"] = "value"
            elif c in "{[":
                st["stack"].append(c)
                st["expect"] = "key" if c == "{" else "value"
            elif c in "}]":
                if st["stack"]:
                    st["stack"].pop()
                if st["args_depth"] is not None and len(st["stack"]) < st["args_depth"]:
                    st["args_depth"] = None
                st["expect"] = "key"
            elif c == ",":
                st["expect"] = "key" if (st["stack"] and st["stack"][-1] == "{") else "value"
        if st["in_call"] and st["in_str"] and st["expect"] == "key":
            st["key"] += c if c not in '"' else ""
    st["speak"] = (not st["in_call"]) or (
        st["in_str"] and st["expect"] == "value"
        and st["args_depth"] is not None and len(st["stack"]) > st["args_depth"])


def _generate(messages, tools, max_new_tokens, temperature, notify,
              active_steer=None, tags=None, tool_choice=None):
    tok, model = state["tokenizer"], state["model"]
    kwargs = {"tools": tools} if tools else {}
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **kwargs)

    # tool_choice enforcement without guided decoding: seed the generation with
    # the opening of a tool call in the model's own format, so the only way to
    # continue is to finish one ("required" picks the tool, a named choice
    # forces it). The prefix is prepended back before parsing the tool call.
    forced_prefix = ""
    if tools and tool_choice and tool_choice not in ("none", "auto"):
        forced = tool_choice.get("function", {}).get("name") \
            if isinstance(tool_choice, dict) else None
        tmpl = getattr(tok, "chat_template", "") or ""
        forced_prefix = ("<tool_call>\n{\"name\": " if "<tool_call>" in tmpl else
                         "```tool_call\n{\"name\": " if "```tool" in tmpl else
                         "{\"name\": ")
        if forced:
            # the opening brace keeps arguments a JSON object (bare values
            # like `"arguments": 3 * 7` would break the parse)
            forced_prefix += f"\"{forced}\", \"arguments\": {{"
        prompt += forced_prefix

    scan = _tool_scan_new()
    _tool_scan(scan, forced_prefix)
    state["steer_mute"] = not scan["speak"]
    ids = tok(prompt, return_tensors="pt").input_ids.to(state["device"])
    past, generated = None, []

    n_prompt = ids.shape[1]
    prompt_ids = ids[0].tolist()[-4096:]  # axis labels; very long prompts keep the tail
    gen = {"id": uuid.uuid4().hex[:12], "n_prompt": n_prompt,
           "prompt_offset": n_prompt - len(prompt_ids),
           "prompt_tokens": [tok.decode(i) for i in prompt_ids],
           "tokens": [], "all_tokens": [], "norms": [], "lens": [], "jlens": [],
           "attn_rows": [], "last_heads": [], "matrix": None, "hidden": [],
           "steer": active_steer, "tags": tags or {}, "done": False}
    # honored for the whole generation, so stored hidden states align with
    # the captured steps even if the toggle flips mid-flight
    save_hidden = bool(state["save_hidden"] and state["traces"])
    state["gen"] = gen
    state["stop"] = False   # cleared each generation; POST /stop sets it
    notify({"type": "start", "gen_id": gen["id"], "prompt_tokens": n_prompt,
            "model": state["model_name"], "steer": active_steer, "tags": tags or {}})

    # compute lm_head only for the last position — full-prompt logits of a 20k
    # prompt × 150k vocab would be ~7 GB (transformers materializes them all)
    logits_kw = {}
    import inspect
    fwd_params = inspect.signature(model.forward).parameters
    for kw in ("logits_to_keep", "num_logits_to_keep"):
        if kw in fwd_params:
            logits_kw = {kw: 1}
            break

    # prefill in chunks — with eager attention a single full-prompt forward
    # materializes the whole seq×seq attention matrix (a 24k-token agent
    # prompt ≈ 37 GB); chunked, the transient is only chunk×seq per layer
    # SHORT prompt + viz on: one full forward WITH attentions, so the viz can
    # show real per-head self-attention matrices (cheap when seq is small).
    out = None
    if state["viz"] and ids.shape[1] <= ATTN_MATRIX_MAX:
        out = model(input_ids=ids, past_key_values=past, use_cache=True,
                    output_attentions=True, **logits_kw)
        past = out.past_key_values
        _capture_matrix(out.attentions, gen)
    else:
        for i in range(0, ids.shape[1], PREFILL_CHUNK):
            out = model(input_ids=ids[:, i:i + PREFILL_CHUNK], past_key_values=past,
                        use_cache=True, **logits_kw)
            past = out.past_key_values
    if state["device"] == "cuda":
        torch.cuda.empty_cache()   # drop prefill transients before decode

    for step in range(max_new_tokens):
        if step:
            # hidden states/attentions only for DECODE steps — prefill signals
            # of a 20k prompt would eat gigabytes, we only visualize the answer.
            # viz off = dark mode: skip all capture, just generate.
            capture = state["viz"]
            out = model(input_ids=ids[:, -1:], past_key_values=past,
                        output_hidden_states=capture, output_attentions=capture,
                        use_cache=True, **logits_kw)
            past = out.past_key_values
        logits = out.logits[0, -1]
        if temperature and temperature > 0:
            probs = torch.softmax(logits / temperature, dim=-1)
            next_id = torch.multinomial(probs, 1)
        else:
            next_id = logits.argmax().reshape(1)
        piece = tok.decode(next_id)
        # steering is muted while the model writes tool-call scaffolding — a
        # persona vector strong enough to matter corrupts strict JSON syntax.
        # Inside the string values of "arguments" the persona speaks again.
        _tool_scan(scan, piece)
        state["steer_mute"] = not scan["speak"]
        payload = {"type": "token", "i": step, "text": piece, "norms": [], "cos": {}}
        gen["all_tokens"].append(piece)
        if out.hidden_states is not None:
            norms, cos = _layer_signals(out.hidden_states, state["directions"])
            payload.update({"norms": norms, "cos": cos})
            probes = state["probes"]
            n_layers = len(norms)
            payload["attn_norm"] = [round(probes["attn"].get(i, 0.0), 2) for i in range(n_layers)]
            payload["mlp_norm"] = [round(probes["mlp"].get(i, 0.0), 2) for i in range(n_layers)]
            hs = _stack_last(out.hidden_states)
            if save_hidden and len(gen["hidden"]) < HIDDEN_MAX_STEPS:
                gen["hidden"].append(hs.to(torch.float16).cpu())
            if state["lens"]:
                lens = _logit_lens(hs)
                payload["lens"] = lens
                gen["lens"].append(lens)
            if state["jlens"] is not None and state["jlens_on"]:
                jlens = _jlens_readout(hs)
                payload["jlens"] = jlens
                gen["jlens"].append(jlens)
            if out.attentions:
                entropy, top, head_entropy = _attn_signals(out.attentions, gen)
                payload.update({"attn_entropy": entropy, "attn_top": top,
                                "head_entropy": head_entropy})
            gen["tokens"].append(piece)
            gen["norms"].append(norms)
        notify(payload)
        generated.append(int(next_id))
        ids = torch.cat([ids, next_id.reshape(1, 1)], dim=1)
        if int(next_id) == tok.eos_token_id or state.get("stop"):
            break

    text = forced_prefix + tok.decode(generated, skip_special_tokens=False)
    gen["done"] = True
    state["steer_mute"] = False
    trace_id = None
    if state["traces"] and state["save_traces"]:
        try:
            state["traces"].save(gen, state["model_name"], gen["hidden"] or None)
            trace_id = gen["id"]
        except Exception as e:   # persistence must never break generation
            print(f"brainscope: trace save failed — {e}", flush=True)
    notify({"type": "done", "gen_id": gen["id"], "completion_tokens": len(generated),
            "trace_id": trace_id})
    return text


def to_openai_response(text: str, model: str, raw: bool = False) -> dict:
    tool_calls = []
    matched = None
    for pattern in TOOL_CALL_PATTERNS:
        for m in pattern.finditer(text):
            try:
                call = json.loads(m.group(1))
            except json.JSONDecodeError:
                continue
            name = call.get("name") or call.get("tool_name")
            args = call.get("arguments") or call.get("parameters") or {}
            tool_calls.append({
                "id": f"call_{uuid.uuid4().hex[:8]}", "type": "function",
                "function": {"name": name,
                             "arguments": json.dumps(args, ensure_ascii=False)}})
        if tool_calls:
            matched = pattern
            break
    content = matched.sub("", text) if matched else text
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.S)
    content = content.replace("<|im_end|>", "").strip()
    message = {"role": "assistant", "content": content or None}
    if raw:   # {"raw": true} in the request keeps the unstripped generation
        message["raw_content"] = text   # incl. <think>…</think> — reasoning-trace clients
    if tool_calls:
        message["tool_calls"] = tool_calls
    return {"id": f"chatcmpl-{uuid.uuid4().hex[:12]}", "object": "chat.completion",
            "created": int(time.time()), "model": model,
            "choices": [{"index": 0, "message": message,
                         "finish_reason": "tool_calls" if tool_calls else "stop"}]}


@app.post("/v1/chat/completions")
async def chat_completions(body: dict):
    loop = asyncio.get_running_loop()

    def notify(payload):
        asyncio.run_coroutine_threadsafe(broadcast(payload), loop)

    # Per-request steering: {"steering": {"name", "strength", "layer_from",
    # "layer_to"}} in the body (OpenAI SDKs pass it via extra_body). Scoped to
    # this request; {"strength": 0} explicitly opts out of any global steering.
    # Without an explicit steering object, request tags (OpenAI `metadata`,
    # e.g. {"agent": ..., "phase": ...}) are matched against the steering
    # policy — the app stays steering-agnostic, rules live here.
    tags = body.get("metadata") if isinstance(body.get("metadata"), dict) else {}
    steering = body.get("steering")
    if steering is None and tags:
        steering = _match_policy(tags)
    if steering is not None:
        if not isinstance(steering, (dict, list)):
            return JSONResponse({"error": "steering must be an object or a list"},
                                status_code=400)
        specs = steering.get("stack", [steering]) if isinstance(steering, dict) else steering
        for s in specs if isinstance(specs, list) else []:
            name = s.get("name")
            if name and name not in state["directions"]:
                return JSONResponse(
                    {"error": f"unknown direction {name!r}",
                     "directions": sorted(state["directions"])}, status_code=400)

    text = await asyncio.to_thread(
        generate_with_signals, body["messages"], body.get("tools"),
        int(body.get("max_tokens") or 1024), float(body.get("temperature") or 0),
        notify, steering, tags, body.get("tool_choice"))
    return JSONResponse(to_openai_response(text, state["model_name"], bool(body.get("raw"))))


@app.get("/v1/models")
async def models():
    return {"object": "list", "data": [{"id": state["model_name"], "object": "model"}]}


@app.get("/info")
async def info():
    cfg = state["model"].config
    return {"model": state["model_name"],
            "n_layers": getattr(cfg, "num_hidden_layers", None),
            "n_heads": getattr(cfg, "num_attention_heads", None),
            "n_kv_heads": getattr(cfg, "num_key_value_heads", None),
            "hidden_size": getattr(cfg, "hidden_size", None),
            "intermediate_size": getattr(cfg, "intermediate_size", None),
            "vocab_size": getattr(cfg, "vocab_size", None),
            "lens": state["lens"],
            "viz": state["viz"],
            "jlens": {"loaded": state["jlens"] is not None, "on": state["jlens_on"],
                      "mode": (state["jlens"].meta.get("mode") if state["jlens"] else None)},
            "traces": {"enabled": state["traces"] is not None,
                       "save": state["save_traces"], "hidden": state["save_hidden"]},
            "params_b": round(sum(p.numel() for p in state["model"].parameters()) / 1e9, 1)}


@app.get("/gen")
async def gen_meta():
    """Metadata + light signals of the current/last generation (for viz
    late-joiners; the heavy attention data stays behind /gen/attention)."""
    g = state["gen"]
    if not g:
        return JSONResponse({"error": "no generation yet"}, status_code=404)
    keys = ("id", "n_prompt", "prompt_offset", "prompt_tokens", "tokens",
            "all_tokens", "norms", "lens", "jlens", "steer", "tags", "done")
    return {k: g.get(k) for k in keys}


@app.get("/gen/attention")
async def gen_attention(layer: int = 0, since: int = 0):
    """Mean-over-heads attention rows for one layer, whole generation so far.
    rows[j] (base64 uint8, 255 = all attention) is the pattern while emitting
    answer token j+1, over positions 0 .. n_prompt+j (token 0 comes from
    prefill, which we don't capture). `since` skips already-fetched rows so
    live polling stays incremental instead of re-shipping megabytes."""
    g = state["gen"]
    if not g or not g["attn_rows"]:
        return JSONResponse({"error": "no attention captured yet"}, status_code=404)
    if not 0 <= layer < len(g["attn_rows"][0]):
        return JSONResponse({"error": "layer out of range"}, status_code=400)
    since = max(0, since)
    return {"id": g["id"], "layer": layer, "n_prompt": g["n_prompt"], "since": since,
            "total": len(g["attn_rows"]),
            "rows": [base64.b64encode(r[layer]).decode() for r in g["attn_rows"][since:]]}


@app.get("/gen/heads")
async def gen_heads(layer: int = 0):
    """Per-head attention of the NEWEST token for one layer (heads × seq,
    base64 uint8, row-major). Head detail is kept for the latest step only —
    storing it for every token would be gigabytes."""
    g = state["gen"]
    if not g or not g["last_heads"]:
        return JSONResponse({"error": "no attention captured yet"}, status_code=404)
    if not 0 <= layer < len(g["last_heads"]):
        return JSONResponse({"error": "layer out of range"}, status_code=400)
    data = g["last_heads"][layer]
    seq = g["n_prompt"] + len(g["attn_rows"])  # may be 1 step stale mid-generation
    if len(data) % seq:
        seq = len(data) // max(1, len(data) // seq)
    return {"id": g["id"], "layer": layer, "seq": seq,
            "heads": len(data) // seq, "data": base64.b64encode(data).decode()}


@app.get("/gen/matrix")
async def gen_matrix(layer: int = 0):
    """Full per-head self-attention matrices from a SHORT prompt's prefill:
    [heads, seq, seq] uint8, row-major, each query row scaled to its own max.
    Only present when the prompt was short enough (see ATTN_MATRIX_MAX) and viz
    was on — long prompts skip it. This is the "peek into the model" view."""
    g = state["gen"]
    if not g or g.get("matrix") is None:
        return JSONResponse({"error": "no attention matrix — prompt too long, or capture was off"},
                            status_code=404)
    if not 0 <= layer < len(g["matrix"]):
        return JSONResponse({"error": "layer out of range"}, status_code=400)
    m = g["matrix"][layer]                      # [heads, seq, seq] uint8
    heads, seq, _ = m.shape
    return {"id": g["id"], "layer": layer, "layers": len(g["matrix"]),
            "heads": int(heads), "seq": int(seq),
            "tokens": g["prompt_tokens"], "data": base64.b64encode(m.tobytes()).decode()}


@app.post("/viz")
async def set_viz(body: dict):
    """Toggle signal capture: {"on": bool}. Off = dark mode — the model just
    generates (still eager-attention slow, but no lens matmuls, no GPU→CPU
    copies, no per-token streaming)."""
    state["viz"] = bool(body.get("on", True))
    return {"on": state["viz"]}


def _piece_at(g: dict, pos: int) -> str | None:
    """Decoded token at an absolute position, if we have it: prompt tail is
    stored decoded; answer tokens start from the second one (prefill emits
    the first without capture)."""
    if pos < g["n_prompt"]:
        i = pos - g["prompt_offset"]
        return g["prompt_tokens"][i] if i >= 0 else None
    k = pos - g["n_prompt"]
    return g["tokens"][k - 1] if k >= 1 else None


@app.get("/gen/sources")
async def gen_sources(step: int = -1, top: int = 8):
    """Which positions fed answer step `step` (default latest): the step's
    mean-over-heads attention rows averaged across ALL layers, top positions
    returned with decoded context snippets — text, not pixels."""
    g = state["gen"]
    if not g or not g["attn_rows"]:
        return JSONResponse({"error": "no attention captured yet"}, status_code=404)
    if step < 0:
        step = len(g["attn_rows"]) - 1
    if not 0 <= step < len(g["attn_rows"]):
        return JSONResponse({"error": "step out of range"}, status_code=400)
    layers = [np.frombuffer(r, dtype=np.uint8).astype(np.float32)
              for r in g["attn_rows"][step]]
    mean = np.mean(layers, axis=0)
    order = np.argsort(mean)[::-1][:top]
    seq_end = g["n_prompt"] + len(g["tokens"]) + 1
    sources = []
    for pos in order:
        pos = int(pos)
        if mean[pos] <= 2:      # below quantization noise
            continue
        before = "".join(p for p in (_piece_at(g, i) for i in range(max(0, pos - 6), pos)) if p)
        after = "".join(p for p in (_piece_at(g, i) for i in range(pos + 1, min(seq_end, pos + 7))) if p)
        sources.append({"pos": pos, "w": round(float(mean[pos]) / 255, 4),
                        "token": _piece_at(g, pos), "before": before, "after": after,
                        "side": "prompt" if pos < g["n_prompt"] else "answer"})
    return {"step": step, "sources": sources}


@app.get("/directions")
async def directions():
    return {"directions": sorted(state["directions"]), "steer": state["steer"],
            "meta": state["dir_meta"]}


def _hidden_size() -> int:
    cfg = state["model"].config
    return getattr(cfg, "hidden_size", None) or cfg.text_config.hidden_size


def _persist_directions() -> None:
    """dirs.json is the vector library — keep it in sync with the live state."""
    path = state.get("dirs_path")
    if not path:
        return
    raw = {k: [round(float(x), 6) for x in v.tolist()] if v.dim() == 1 else
              [[round(float(x), 6) for x in row] for row in v.tolist()]
           for k, v in state["directions"].items()}
    Path(path).write_text(json.dumps(raw))


def load_direction_dict(path: Path) -> dict:
    """Load a direction_dict/ folder in the hidden-directions layout
    (github.com/moudrkat/hidden-directions): manifest.json naming the source
    model plus one .pt tensor per direction, shaped [hidden] or
    [n_layers, hidden]. Tensors are loaded with weights_only=True, so a dict
    from the internet cannot execute code here."""
    hidden = _hidden_size()
    manifest_path = path / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        src_model = manifest.get("model")
        if src_model and src_model != state["model_name"]:
            print(f"brainscope: WARNING — direction dict was extracted on {src_model}, "
                  f"but the loaded model is {state['model_name']}; foreign directions "
                  "produce noise, not steering", flush=True)
        # optional per-direction steering presets in the manifest
        for e in manifest.get("directions", []):
            hint = {}
            if e.get("recommended_layer") is not None:
                hint["layer_from"] = hint["layer_to"] = int(e["recommended_layer"])
            if e.get("recommended_alpha") is not None:
                hint["strength"] = float(e["recommended_alpha"])
            if hint:
                state["dir_meta"][e["name"]] = hint
    dirs = {}
    for f in sorted(path.glob("*.pt")):
        t = torch.load(f, map_location="cpu", weights_only=True)
        if not torch.is_tensor(t) or t.dim() > 2 or t.shape[-1] != hidden:
            shape = tuple(t.shape) if torch.is_tensor(t) else type(t).__name__
            print(f"brainscope: skipping {f.name} ({shape} does not fit hidden size {hidden})", flush=True)
            continue
        dirs[f.stem] = t.float().to(state["device"])
    print(f"brainscope: loaded {len(dirs)} direction(s) from {path}", flush=True)
    return dirs


@app.post("/capture")
async def capture(body: dict):
    """Read a prompt's residual-stream state at one layer — the raw material for
    agent-to-agent 'telepathy': capture agent A's state here, POST it to
    /directions, then steer agent B by it (per-request or /steer).

    {"messages": [...], "layer": int?, "pool": "mean"|"last"?} ->
    {"layer", "pool", "hidden": int, "vector": [floats]}.

    `layer` is a decoder-layer index (default: mid-stack); the vector returned is
    that layer's OUTPUT residual, so injecting it back at the same layer aligns
    with how the steer hooks add (h += strength * vector at that layer's output).
    `pool` averages over the prompt tokens ("mean", default) or takes the last
    position ("last")."""
    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        return JSONResponse({"error": "expected {messages: [...], layer?, pool?}"},
                            status_code=400)
    model = state["model"]
    n_layers = len(_decoder_layers(model))
    layer = int(body.get("layer", n_layers // 2))
    if not 0 <= layer < n_layers:
        return JSONResponse({"error": f"layer out of range 0..{n_layers - 1}"},
                            status_code=400)
    pool = body.get("pool", "mean")
    if pool not in ("mean", "last"):
        return JSONResponse({"error": "pool must be 'mean' or 'last'"}, status_code=400)

    def run():
        tok = state["tokenizer"]
        prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        ids = tok(prompt, return_tensors="pt").input_ids.to(state["device"])
        with torch.no_grad():
            out = model(input_ids=ids, output_hidden_states=True, use_cache=False)
        h = out.hidden_states[layer + 1][0]   # [seq, hidden] — layer's output residual
        vec = h[-1] if pool == "last" else h.mean(0)
        return vec.float().cpu().tolist()

    vector = await asyncio.to_thread(run)
    return {"layer": layer, "pool": pool, "hidden": len(vector), "vector": vector}


@app.post("/directions")
async def add_direction(body: dict):
    name, vector = body.get("name"), body.get("vector")
    if not name or not isinstance(vector, list):
        return JSONResponse({"error": "expected {name, vector: [floats] or [[floats] per layer]}"},
                            status_code=400)
    hidden = _hidden_size()
    tensor = torch.tensor(vector).float()
    if tensor.dim() > 2 or tensor.shape[-1] != hidden:
        return JSONResponse({"error": f"vector rows must have {hidden} dims"},
                            status_code=400)
    state["directions"][name] = tensor.to(state["device"])
    if isinstance(body.get("meta"), dict):   # optional steering preset
        state["dir_meta"][name] = {k: body["meta"][k] for k in
                                   ("strength", "layer_from", "layer_to")
                                   if body["meta"].get(k) is not None}
    _persist_directions()
    return {"directions": sorted(state["directions"])}


@app.delete("/directions/{name}")
async def delete_direction(name: str):
    if name not in state["directions"]:
        return JSONResponse({"error": "unknown direction"}, status_code=404)
    del state["directions"][name]
    state["dir_meta"].pop(name, None)
    if state["steer"] and any(s["name"] == name for s in state["steer"]):
        apply_steering(None)
    _persist_directions()
    return {"directions": sorted(state["directions"])}


def _persist_policy() -> None:
    path = state.get("policy_path")
    if path:
        Path(path).write_text(json.dumps(
            {"enabled": state["policy_on"], "policy": state["policy"]},
            ensure_ascii=False, indent=1))


@app.get("/policy")
async def get_policy():
    return {"policy": state["policy"], "enabled": state["policy_on"]}


@app.post("/policy")
async def set_policy(body: dict):
    """Replace the steering policy and/or flip it on and off:
    {"policy": [{"match": {tag: value, ...},
                 "steer": {"name", "strength", "layer_from", "layer_to"}}, ...],
     "enabled": bool} — both keys optional, so {"enabled": false} pauses the
    rules without losing them. First matching rule wins; missing match keys
    are wildcards."""
    rules = body.get("policy")
    if rules is None and "enabled" not in body:
        return JSONResponse({"error": "expected {policy: [...]} and/or {enabled: bool}"},
                            status_code=400)
    if rules is not None:
        if not isinstance(rules, list):
            return JSONResponse({"error": "expected {policy: [{match, steer}, ...]}"},
                                status_code=400)
        for rule in rules:
            if not isinstance(rule, dict):
                return JSONResponse({"error": "each rule must be an object"}, status_code=400)
            name = (rule.get("steer") or {}).get("name")
            if name and name not in state["directions"]:
                return JSONResponse({"error": f"unknown direction {name!r}",
                                     "directions": sorted(state["directions"])},
                                    status_code=400)
        state["policy"] = rules
    if "enabled" in body:
        state["policy_on"] = bool(body["enabled"])
    _persist_policy()
    return {"policy": state["policy"], "enabled": state["policy_on"]}


# ------------------------------------------------------------- J-lens ----

@app.get("/jlens")
async def jlens_info():
    jl = state["jlens"]
    if jl is None:
        return {"loaded": False, "on": False}
    return {"loaded": True, "on": state["jlens_on"], "meta": jl.meta,
            "n_layers": jl.n_layers, "hidden": jl.hidden}


@app.post("/jlens")
async def jlens_toggle(body: dict):
    """{"on": bool} — flip the per-token J-lens readout. This is the heavy
    switch: one [n_layers, d, d] transport per generated token when on."""
    if state["jlens"] is None:
        return JSONResponse({"error": "no J-lens loaded — start with --jlens LENS.pt "
                             "(fit one: python -m brainscope.jlens fit …)"},
                            status_code=400)
    state["jlens_on"] = bool(body.get("on", True))
    return {"on": state["jlens_on"]}


@app.post("/jlens/direction")
async def jlens_direction(body: dict):
    """Steering × J-lens: {"text": "cake", "name"?} materializes the J-space
    steering direction for a vocabulary token — the per-layer activation
    pattern that, to first order, makes the model more likely to say the
    token LATER — and registers it as a normal [n_layers, hidden] direction
    for the existing steer stack / policies. Nudge what's on the model's
    mind, then watch the J-lens panel to see whether it took."""
    jl, text = state["jlens"], (body.get("text") or "").strip()
    if jl is None:
        return JSONResponse({"error": "no J-lens loaded"}, status_code=400)
    if not text:
        return JSONResponse({"error": "expected {text: word}"}, status_code=400)
    tok = state["tokenizer"]
    ids = tok(" " + text, add_special_tokens=False).input_ids or \
        tok(text, add_special_tokens=False).input_ids
    if not ids:
        return JSONResponse({"error": f"cannot tokenize {text!r}"}, status_code=400)
    piece = tok.decode(ids[0])
    head = state["model"].get_output_embeddings()
    dirs = jl.direction(ids[0], head.weight.detach().float().cpu())
    name = body.get("name") or f"j:{text}"
    state["directions"][name] = dirs.to(state["device"])
    # rows are unit-normalized: gentle strength over a NARROW mid-stack band —
    # ~1.5 @ 3 layers reads naturally, 4+ over a wide band collapses small
    # models into chanting the word (measured on Qwen2.5-0.5B)
    n = jl.n_layers
    state["dir_meta"][name] = {"layer_from": round(n * 0.42), "layer_to": round(n * 0.54),
                               "strength": 1.5}
    _persist_directions()
    return {"name": name, "token": piece, "multi_token": len(ids) > 1,
            "directions": sorted(state["directions"])}


# ------------------------------------------------------------- traces ----

@app.get("/traces")
async def traces_list():
    if state["traces"] is None:
        return JSONResponse({"error": "trace persistence is off — start with --traces DIR"},
                            status_code=404)
    return {"traces": state["traces"].list(),
            "save": state["save_traces"], "hidden": state["save_hidden"]}


@app.post("/traces/config")
async def traces_config(body: dict):
    """{"save": bool?, "hidden": bool?} — `hidden` is the heavy one: keeps
    every captured step's full [n_layers, hidden] residual next to the trace
    (fp16), which is what exact post-hoc analytics (emergence) need."""
    if state["traces"] is None:
        return JSONResponse({"error": "trace persistence is off — start with --traces DIR"},
                            status_code=404)
    if "save" in body:
        state["save_traces"] = bool(body["save"])
    if "hidden" in body:
        state["save_hidden"] = bool(body["hidden"])
    return {"save": state["save_traces"], "hidden": state["save_hidden"]}


@app.get("/traces/{trace_id}")
async def trace_get(trace_id: str):
    t = state["traces"].load(trace_id) if state["traces"] else None
    if t is None:
        return JSONResponse({"error": "unknown trace"}, status_code=404)
    return t


@app.delete("/traces/{trace_id}")
async def trace_delete(trace_id: str):
    if not (state["traces"] and state["traces"].delete(trace_id)):
        return JSONResponse({"error": "unknown trace"}, status_code=404)
    return {"deleted": trace_id}


@app.get("/traces/{trace_id}/emergence")
async def trace_emergence(trace_id: str, token: str | None = None):
    """When did the answer surface? p(answer token) per reasoning step, best
    layer, under each lens — exact where hidden states were stored, top-k
    lower bound otherwise. `token` overrides which piece to track."""
    store = state["traces"]
    t = store.load(trace_id) if store else None
    if t is None:
        return JSONResponse({"error": "unknown trace"}, status_code=404)
    norm, head = _final_norm_and_head()

    def run():
        return compute_emergence(t, store.hidden(trace_id), tokenizer=state["tokenizer"],
                                 norm=norm, head=head, jlens=state["jlens"],
                                 override=token)

    out = await asyncio.to_thread(run)
    status = 400 if "error" in out else 200
    return JSONResponse(out, status_code=status)


@app.post("/stop")
async def stop():
    # cooperative cancel: the decode loop checks state["stop"] each token, so
    # this returns the partial answer instead of killing the whole server
    state["stop"] = True
    return {"stopped": True}


@app.post("/steer")
async def steer(body: dict):
    # one spec {"name", "strength", "layer_from", "layer_to"} — or a composed
    # {"stack": [spec, ...]} applying several vectors at once (bake-recipe style)
    return apply_steering(body)


@app.get("/")
async def index():
    return FileResponse(STATIC / "index.html")


@app.websocket("/ws")
async def ws(websocket: WebSocket):
    await websocket.accept()
    state["clients"].add(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        state["clients"].discard(websocket)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--port", type=int, default=8010)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--device", default=None)
    parser.add_argument("--directions", type=Path, default=None,
                        help="JSON {name: vector or [n_layers, hidden] matrix}, or a "
                             "hidden-directions direction_dict/ folder (manifest.json + *.pt)")
    parser.add_argument("--bake", type=Path, default=None,
                        help="hidden-directions bake artifact folder (advocate_bias.pt) "
                             "to patch into the model — serve a baked persona and audit it")
    parser.add_argument("--policy", type=Path, default=None,
                        help="JSON steering-policy rules matched against request metadata tags")
    parser.add_argument("--lens", choices=["auto", "on", "off"], default="auto",
                        help="logit lens (per-layer next-token readout); auto = on for CUDA, "
                             "off for CPU (one lm_head matmul per layer per token)")
    parser.add_argument("--jlens", type=Path, default=None,
                        help="fitted Jacobian-lens artifact (python -m brainscope.jlens fit …) "
                             "— per-token J-space readout, toggleable live via POST /jlens")
    parser.add_argument("--traces", type=Path, default=None,
                        help="directory for trace persistence — every generation saved for "
                             "replay & reasoning-trace analytics (hidden-state capture stays "
                             "off until enabled via POST /traces/config)")
    parser.add_argument("--keep-traces", type=int, default=200,
                        help="max stored traces before the oldest are dropped")
    parser.add_argument("--quantize", choices=["8bit", "4bit"], default=None,
                        help="bitsandbytes quantization to fit bigger models on 16 GB")
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    model_id = PRESETS.get(args.model, args.model)
    print(f"brainscope: loading {model_id} …")
    load_model(model_id, args.device, args.quantize)
    if args.bake:
        apply_bake(args.bake)
    if args.directions:
        if args.directions.is_dir():   # hidden-directions direction_dict/ (read-only)
            state["directions"] = load_direction_dict(args.directions)
        else:
            state["dirs_path"] = args.directions
            if args.directions.exists():
                raw = json.loads(args.directions.read_text())
                state["directions"] = {k: torch.tensor(v).float().to(state["device"])
                                       for k, v in raw.items()}
            # pca_directions writes suggested layers next to the vector library
            meta_path = args.directions.with_suffix(".meta.json")
            if meta_path.exists():
                raw = json.loads(meta_path.read_text())
                if "name" in raw:   # legacy single-direction meta file
                    raw = {raw["name"]: raw}
                for k, m in raw.items():
                    if k in state["directions"] and m.get("suggested_layers"):
                        state["dir_meta"][k] = {"layer_from": m["suggested_layers"][0],
                                                "layer_to": m["suggested_layers"][1]}
    if args.policy:
        state["policy_path"] = args.policy
        if args.policy.exists():
            raw = json.loads(args.policy.read_text())
            if isinstance(raw, dict):    # {"enabled": bool, "policy": [...]}
                state["policy"] = raw.get("policy", [])
                state["policy_on"] = bool(raw.get("enabled", True))
            else:                        # legacy format: bare list of rules
                state["policy"] = raw
    state["lens"] = args.lens == "on" or (args.lens == "auto" and state["device"] == "cuda")
    if args.jlens:
        jl = JacobianLens.load(args.jlens, device=state["device"])
        n_layers, hidden = len(_decoder_layers(state["model"])), _hidden_size()
        if (jl.n_layers, jl.hidden) != (n_layers, hidden):
            raise SystemExit(f"brainscope: J-lens {args.jlens} was fitted for "
                             f"{jl.meta.get('model')} ({jl.n_layers} layers × {jl.hidden}), "
                             f"but the loaded model has {n_layers} × {hidden}")
        if jl.meta.get("model") not in ("?", None, state["model_name"]):
            print(f"brainscope: WARNING — J-lens fitted on {jl.meta['model']}, "
                  f"serving {state['model_name']}", flush=True)
        state["jlens"], state["jlens_on"] = jl, True
        print(f"brainscope: J-lens loaded ({jl.meta.get('mode', 'future')} mode, "
              f"identity_error {jl.meta.get('identity_error')})", flush=True)
    if args.traces:
        state["traces"] = TraceStore(args.traces, keep=args.keep_traces)
        print(f"brainscope: traces → {args.traces} "
              f"({len(state['traces'].index)} existing)", flush=True)
    if not args.no_browser:
        import threading
        import webbrowser
        threading.Timer(1.0, lambda: webbrowser.open(f"http://localhost:{args.port}/")).start()
    print(f"brainscope: app endpoint http://0.0.0.0:{args.port}/v1 · viz http://localhost:{args.port}/")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
