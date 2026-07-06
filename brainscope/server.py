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
import gc
import threading
import json
import re
import time
import uuid
from pathlib import Path

import torch
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse

STATIC = Path(__file__).parent.parent / "static"

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
               "steer": None, "steer_handles": []}

# Tool-call output formats differ per model family; try each in order.
TOOL_CALL_PATTERNS = [
    re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S),          # qwen/hermes
    re.compile(r"```tool_(?:call|code)\s*(\{.*?\})\s*```", re.S),         # gemma-style fenced
    re.compile(r"^\s*(\{\s*\"name\".*?\"arguments\".*?\})\s*$", re.S),  # bare JSON fallback
]


def load_model(name: str, device: str | None, quantize: str | None = None) -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    kwargs = {"torch_dtype": torch.bfloat16 if dev == "cuda" else torch.float32}
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


def apply_steering(name: str | None, strength: float, layer_from: int, layer_to: int) -> dict:
    """(Re)install activation-addition hooks: h += strength * direction."""
    for h in state["steer_handles"]:
        h.remove()
    state["steer_handles"] = []
    if not name or strength == 0:
        state["steer"] = None
        return {"active": False}
    vec = state["directions"][name]

    def hook(_module, _inp, out):
        hidden = out[0] if isinstance(out, tuple) else out
        hidden = hidden + strength * vec.to(hidden.dtype)
        return (hidden, *out[1:]) if isinstance(out, tuple) else hidden

    layers = _decoder_layers(state["model"])
    layer_to = min(layer_to if layer_to >= 0 else len(layers) - 1, len(layers) - 1)
    for i in range(max(0, layer_from), layer_to + 1):
        state["steer_handles"].append(layers[i].register_forward_hook(hook))
    state["steer"] = {"name": name, "strength": strength,
                      "layers": [max(0, layer_from), layer_to]}
    return {"active": True, **state["steer"]}


def _layer_signals(hidden_states, directions):
    """Per-layer L2 norms (+ cosines with named directions) for the last token."""
    norms, cos = [], {name: [] for name in directions}
    for h in hidden_states[1:]:  # skip embedding layer
        v = h[0, -1].float()
        norms.append(float(v.norm()))
        for name, d in directions.items():
            cos[name].append(float(torch.nn.functional.cosine_similarity(v, d, dim=0)))
    return norms, cos


_GEN_LOCK = threading.Lock()  # one generation at a time — retries/parallel agents must queue


@torch.inference_mode()
def generate_with_signals(messages, tools, max_new_tokens, temperature, notify):
    """Token-by-token generation, calling notify(payload) per token."""
    with _GEN_LOCK:
        try:
            return _generate(messages, tools, max_new_tokens, temperature, notify)
        finally:
            gc.collect()
            if state["device"] == "cuda":
                torch.cuda.empty_cache()


def _generate(messages, tools, max_new_tokens, temperature, notify):
    tok, model = state["tokenizer"], state["model"]
    kwargs = {"tools": tools} if tools else {}
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **kwargs)
    ids = tok(prompt, return_tensors="pt").input_ids.to(state["device"])
    past, generated = None, []
    notify({"type": "start", "prompt_tokens": ids.shape[1], "model": state["model_name"]})

    for step in range(max_new_tokens):
        # hidden states only for DECODE steps — prefill hidden states of a 20k
        # prompt would eat gigabytes and we only visualize the answer
        out = model(input_ids=ids if past is None else ids[:, -1:],
                    past_key_values=past, output_hidden_states=past is not None, use_cache=True)
        past = out.past_key_values
        logits = out.logits[0, -1]
        if temperature and temperature > 0:
            probs = torch.softmax(logits / temperature, dim=-1)
            next_id = torch.multinomial(probs, 1)
        else:
            next_id = logits.argmax().reshape(1)
        if out.hidden_states is not None:
            norms, cos = _layer_signals(out.hidden_states, state["directions"])
            piece = tok.decode(next_id)
            notify({"type": "token", "i": step, "text": piece, "norms": norms, "cos": cos})
        else:
            piece = tok.decode(next_id)
            notify({"type": "token", "i": step, "text": piece,
                    "norms": [], "cos": {}})
        generated.append(int(next_id))
        ids = torch.cat([ids, next_id.reshape(1, 1)], dim=1)
        if int(next_id) == tok.eos_token_id:
            break

    text = tok.decode(generated, skip_special_tokens=False)
    notify({"type": "done", "completion_tokens": len(generated)})
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

    text = await asyncio.to_thread(
        generate_with_signals, body["messages"], body.get("tools"),
        int(body.get("max_tokens") or 1024), float(body.get("temperature") or 0), notify)
    return JSONResponse(to_openai_response(text, state["model_name"]))


@app.get("/v1/models")
async def models():
    return {"object": "list", "data": [{"id": state["model_name"], "object": "model"}]}


@app.get("/directions")
async def directions():
    return {"directions": sorted(state["directions"]), "steer": state["steer"]}


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
                        help="JSON {name: [hidden_size floats]} to project layers onto")
    parser.add_argument("--quantize", choices=["8bit", "4bit"], default=None,
                        help="bitsandbytes quantization to fit bigger models on 16 GB")
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    model_id = PRESETS.get(args.model, args.model)
    print(f"brainscope: loading {model_id} …")
    load_model(model_id, args.device, args.quantize)
    if args.directions:
        raw = json.loads(args.directions.read_text())
        state["directions"] = {k: torch.tensor(v).float().to(state["device"])
                               for k, v in raw.items()}
    if not args.no_browser:
        import threading
        import webbrowser
        threading.Timer(1.0, lambda: webbrowser.open(f"http://localhost:{args.port}/")).start()
    print(f"brainscope: app endpoint http://0.0.0.0:{args.port}/v1 · viz http://localhost:{args.port}/")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
