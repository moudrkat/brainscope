"""Shared fixtures: a tiny random Qwen2 model (real architecture, random
weights, ~10 M params) so the whole capture/lens/trace pipeline runs in
seconds on CPU without downloading model weights. Only the tokenizer comes
from the Hub (Qwen2.5-0.5B-Instruct — chat template included)."""

import pytest
import torch
from transformers import AutoTokenizer, Qwen2Config, Qwen2ForCausalLM

from brainscope import jlens as jl
from brainscope import server as bs
from brainscope.traces import TraceStore

TOKENIZER_ID = "Qwen/Qwen2.5-0.5B-Instruct"

FIT_TEXTS = [
    f"Snippet {i}: the little tram climbs the hill above the river while two "
    f"engineers argue about voltage, rain, and the price of coffee in Prague. "
    f"Later that day, test number {i} finally passes and everyone goes home."
    for i in range(12)
]


@pytest.fixture(scope="session")
def tok():
    return AutoTokenizer.from_pretrained(TOKENIZER_ID)


@pytest.fixture(scope="session")
def model(tok):
    cfg = Qwen2Config(hidden_size=64, intermediate_size=128, num_hidden_layers=4,
                      num_attention_heads=4, num_key_value_heads=2,
                      vocab_size=len(tok), max_position_embeddings=1024,
                      tie_word_embeddings=False)
    cfg._attn_implementation = "eager"   # server relies on real attention weights
    torch.manual_seed(0)
    return Qwen2ForCausalLM(cfg).eval()


@pytest.fixture(scope="session")
def fitted_lens(model, tok):
    return jl.fit(model, tok, FIT_TEXTS, repeats=8, max_tokens=48)


def make_state(model, tok, traces_dir):
    bs.state.update({
        "model": model, "tokenizer": tok, "device": "cpu",
        "model_name": "tiny-random-qwen2", "directions": {}, "dir_meta": {},
        "clients": set(), "steer": None, "steer_handles": [], "steer_mute": False,
        "policy": [], "policy_on": True, "gen": None, "lens": True, "viz": True,
        "jlens": None, "jlens_on": False,
        "traces": TraceStore(traces_dir), "save_traces": True, "save_hidden": False,
        "probes": {"attn": {}, "mlp": {}},
    })
    bs.state.pop("dirs_path", None)
    bs.state.pop("policy_path", None)
    bs._install_probe_hooks()
    return bs.state


@pytest.fixture
def app_state(model, tok, tmp_path):
    return make_state(model, tok, tmp_path / "traces")


@pytest.fixture
def client(app_state):
    from fastapi.testclient import TestClient
    with TestClient(bs.app) as c:
        yield c


def chat(client, text="Say something.", max_tokens=10, **extra):
    body = {"messages": [{"role": "user", "content": text}],
            "max_tokens": max_tokens, **extra}
    r = client.post("/v1/chat/completions", json=body)
    assert r.status_code == 200, r.text
    return r.json()
