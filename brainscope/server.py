"""brainscope — watch your model think while your app talks to it.

An OpenAI-compatible chat-completions server over any Hugging Face causal LM
that streams per-token, per-layer residual-stream activity to a live browser
visualization. Point your app's OpenAI base_url at it; open http://host:port/
in a window next to your app.

    python -m brainscope.server --model Qwen/Qwen2.5-0.5B-Instruct --port 8010

Optional steering-direction projections: --directions dirs.json where the file
maps {"name": [hidden_size floats], ...}; each generated token then also
reports its per-layer cosine with every named direction (the hook for watching
steering vectors work — and, later, for applying them).
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
state: dict = {"model": None, "tokenizer": None, "directions": {}, "clients": set(),
               "loop": None, "device": "cpu", "model_name": "",
               "steer": None, "steer_handles": [], "policy": [], "policy_on": True,
               "gen": None, "probes": {"attn": {}, "mlp": {}}, "lens": False,
               "viz": True}

# Prefill batch size: bounds eager attention's transient chunk×seq matrix.
# Eager softmax upcasts to fp32, so the transient is chunk × seq × heads × 4 B
# — at 128 that's ~0.4 GB for a 24k prompt, safe next to model + KV cache.
PREFILL_CHUNK = 128

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


def _install_steer_hooks(name: str, strength: float, layer_from: int, layer_to: int) -> list:
    """Register activation-addition hooks (h += strength * direction), return handles.

    A direction is either one vector [hidden] applied to every steered layer,
    or a per-layer matrix [n_layers, hidden] (e.g. a hidden-directions dict
    entry) where each steered layer gets its own row."""
    vec = state["directions"][name]

    def make_hook(row):
        def hook(_module, _inp, out):
            hidden = out[0] if isinstance(out, tuple) else out
            hidden = hidden + strength * row.to(hidden.dtype)
            return (hidden, *out[1:]) if isinstance(out, tuple) else hidden
        return hook

    layers = _decoder_layers(state["model"])
    layer_to = min(layer_to if layer_to >= 0 else len(layers) - 1, len(layers) - 1)
    return [layers[i].register_forward_hook(
                make_hook(vec[min(i, vec.shape[0] - 1)] if vec.dim() == 2 else vec))
            for i in range(max(0, layer_from), layer_to + 1)]


def apply_steering(name: str | None, strength: float, layer_from: int, layer_to: int) -> dict:
    """(Re)install the GLOBAL steering hooks (the viz slider). Per-request
    steering (a `steering` object in a chat completions request) temporarily
    replaces these for the duration of that one generation."""
    for h in state["steer_handles"]:
        h.remove()
    state["steer_handles"] = []
    if not name or strength == 0:
        state["steer"] = None
        return {"active": False}
    state["steer_handles"] = _install_steer_hooks(name, strength, layer_from, layer_to)
    layers = _decoder_layers(state["model"])
    layer_to = min(layer_to if layer_to >= 0 else len(layers) - 1, len(layers) - 1)
    state["steer"] = {"name": name, "strength": strength,
                      "layers": [max(0, layer_from), layer_to]}
    return {"active": True, **state["steer"]}


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


def _logit_lens(hidden_states, top: int = 5):
    """What the model would say if it stopped at each layer: every layer's
    hidden state pushed through the final norm + lm_head. Watching the answer
    crystallize with depth is the point of the exercise."""
    norm, head = _final_norm_and_head()
    if norm is None or head is None:
        return None
    hs = torch.stack([h[0, -1] for h in hidden_states[1:]])  # [n_layers, hidden]
    dtype = next(head.parameters()).dtype
    probs = torch.softmax(head(norm(hs.to(dtype))).float(), dim=-1)
    p, idx = probs.topk(top, dim=-1)
    tok = state["tokenizer"]
    return [[{"t": tok.decode(int(idx[layer, k])), "p": round(float(p[layer, k]), 4)}
             for k in range(top)] for layer in range(hs.shape[0])]


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
    requests (OpenAI `metadata`: e.g. {"agent": "super_agent", "phase":
    "DISCUSS"}) and keep all steering knowledge on the brainscope side."""
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
                          steering: dict | None = None, tags: dict | None = None):
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
            name = steering.get("name")
            strength = float(steering.get("strength") or 0)
            if name and strength != 0:
                layer_from = int(steering.get("layer_from", 0))
                layer_to = int(steering.get("layer_to", -1))
                request_handles = _install_steer_hooks(name, strength, layer_from, layer_to)
                active_steer = {"name": name, "strength": strength,
                                "layers": [layer_from, layer_to], "scope": "request"}
            else:
                active_steer = None
        try:
            return _generate(messages, tools, max_new_tokens, temperature, notify,
                             active_steer, tags)
        finally:
            for h in request_handles:
                h.remove()
            if steering is not None and state["steer"]:
                s = state["steer"]
                state["steer_handles"] = _install_steer_hooks(
                    s["name"], s["strength"], s["layers"][0], s["layers"][1])
            gc.collect()
            if state["device"] == "cuda":
                torch.cuda.empty_cache()


def _generate(messages, tools, max_new_tokens, temperature, notify,
              active_steer=None, tags=None):
    tok, model = state["tokenizer"], state["model"]
    kwargs = {"tools": tools} if tools else {}
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **kwargs)
    ids = tok(prompt, return_tensors="pt").input_ids.to(state["device"])
    past, generated = None, []

    n_prompt = ids.shape[1]
    prompt_ids = ids[0].tolist()[-4096:]  # axis labels; very long prompts keep the tail
    gen = {"id": uuid.uuid4().hex[:12], "n_prompt": n_prompt,
           "prompt_offset": n_prompt - len(prompt_ids),
           "prompt_tokens": [tok.decode(i) for i in prompt_ids],
           "tokens": [], "norms": [], "lens": [], "attn_rows": [], "last_heads": [],
           "steer": active_steer, "tags": tags or {}, "done": False}
    state["gen"] = gen
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
    out = None
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
        payload = {"type": "token", "i": step, "text": piece, "norms": [], "cos": {}}
        if out.hidden_states is not None:
            norms, cos = _layer_signals(out.hidden_states, state["directions"])
            payload.update({"norms": norms, "cos": cos})
            probes = state["probes"]
            n_layers = len(norms)
            payload["attn_norm"] = [round(probes["attn"].get(i, 0.0), 2) for i in range(n_layers)]
            payload["mlp_norm"] = [round(probes["mlp"].get(i, 0.0), 2) for i in range(n_layers)]
            if state["lens"]:
                lens = _logit_lens(out.hidden_states)
                payload["lens"] = lens
                gen["lens"].append(lens)
            if out.attentions:
                entropy, top, head_entropy = _attn_signals(out.attentions, gen)
                payload.update({"attn_entropy": entropy, "attn_top": top,
                                "head_entropy": head_entropy})
            gen["tokens"].append(piece)
            gen["norms"].append(norms)
        notify(payload)
        generated.append(int(next_id))
        ids = torch.cat([ids, next_id.reshape(1, 1)], dim=1)
        if int(next_id) == tok.eos_token_id:
            break

    text = tok.decode(generated, skip_special_tokens=False)
    gen["done"] = True
    notify({"type": "done", "gen_id": gen["id"], "completion_tokens": len(generated)})
    return text


def to_openai_response(text: str, model: str) -> dict:
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
        if not isinstance(steering, dict):
            return JSONResponse({"error": "steering must be an object"}, status_code=400)
        name = steering.get("name")
        if name and name not in state["directions"]:
            return JSONResponse(
                {"error": f"unknown direction {name!r}",
                 "directions": sorted(state["directions"])}, status_code=400)

    text = await asyncio.to_thread(
        generate_with_signals, body["messages"], body.get("tools"),
        int(body.get("max_tokens") or 1024), float(body.get("temperature") or 0),
        notify, steering, tags)
    return JSONResponse(to_openai_response(text, state["model_name"]))


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
            "params_b": round(sum(p.numel() for p in state["model"].parameters()) / 1e9, 1)}


@app.get("/gen")
async def gen_meta():
    """Metadata + light signals of the current/last generation (for viz
    late-joiners; the heavy attention data stays behind /gen/attention)."""
    g = state["gen"]
    if not g:
        return JSONResponse({"error": "no generation yet"}, status_code=404)
    keys = ("id", "n_prompt", "prompt_offset", "prompt_tokens", "tokens",
            "norms", "lens", "steer", "tags", "done")
    return {k: g[k] for k in keys}


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
    return {"directions": sorted(state["directions"]), "steer": state["steer"]}


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
        src_model = json.loads(manifest_path.read_text()).get("model")
        if src_model and src_model != state["model_name"]:
            print(f"brainscope: WARNING — direction dict was extracted on {src_model}, "
                  f"but the loaded model is {state['model_name']}; foreign directions "
                  "produce noise, not steering", flush=True)
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
    _persist_directions()
    return {"directions": sorted(state["directions"])}


@app.delete("/directions/{name}")
async def delete_direction(name: str):
    if name not in state["directions"]:
        return JSONResponse({"error": "unknown direction"}, status_code=404)
    del state["directions"][name]
    if state["steer"] and state["steer"]["name"] == name:
        apply_steering(None, 0, 0, -1)
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


@app.post("/steer")
async def steer(body: dict):
    return apply_steering(body.get("name"), float(body.get("strength") or 0),
                          int(body.get("layer_from", 0)), int(body.get("layer_to", -1)))


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
    parser.add_argument("--policy", type=Path, default=None,
                        help="JSON steering-policy rules matched against request metadata tags")
    parser.add_argument("--lens", choices=["auto", "on", "off"], default="auto",
                        help="logit lens (per-layer next-token readout); auto = on for CUDA, "
                             "off for CPU (one lm_head matmul per layer per token)")
    parser.add_argument("--quantize", choices=["8bit", "4bit"], default=None,
                        help="bitsandbytes quantization to fit bigger models on 16 GB")
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    model_id = PRESETS.get(args.model, args.model)
    print(f"brainscope: loading {model_id} …")
    load_model(model_id, args.device, args.quantize)
    if args.directions:
        if args.directions.is_dir():   # hidden-directions direction_dict/ (read-only)
            state["directions"] = load_direction_dict(args.directions)
        else:
            state["dirs_path"] = args.directions
            if args.directions.exists():
                raw = json.loads(args.directions.read_text())
                state["directions"] = {k: torch.tensor(v).float().to(state["device"])
                                       for k, v in raw.items()}
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
    if not args.no_browser:
        import threading
        import webbrowser
        threading.Timer(1.0, lambda: webbrowser.open(f"http://localhost:{args.port}/")).start()
    print(f"brainscope: app endpoint http://0.0.0.0:{args.port}/v1 · viz http://localhost:{args.port}/")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
