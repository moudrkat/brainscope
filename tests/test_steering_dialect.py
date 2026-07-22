"""Canonical (hotwire) steering dialect, regime flags, passport export, parity.

The spec dialect people deploy with (hotwire's ``{"id", "layer", "scale",
"decode_only"}``) must work here unchanged — including the full wire format
(``vllm_xargs.hotwire`` as a JSON string), so a request built for a
hotwire-vLLM server steers a brainscope server identically.
"""

import json

import pytest
import torch

from brainscope import server as bs
from brainscope.export_hotwire import export
from brainscope.parity import compare
from tests.conftest import chat


@pytest.fixture
def direction(app_state):
    torch.manual_seed(7)
    vec = torch.randn(64)
    bs.state["directions"]["vec"] = vec
    return vec


def test_normalize_accepts_hotwire_dialect(app_state, direction):
    specs = bs._normalize_steer({"id": "vec", "layer": 2, "scale": 3.0,
                                 "decode_only": True})
    assert specs == [{"name": "vec", "strength": 3.0,
                      "layer_from": 2, "layer_to": 2,
                      "prefill": False, "syntax_mute": True}]


def test_normalize_accepts_json_string_and_list(app_state, direction):
    wire = json.dumps([{"id": "vec", "layer": 1, "scale": 2},
                       {"id": "vec", "layer": 3, "scale": 2}])
    specs = bs._normalize_steer(wire)
    assert [s["layer_from"] for s in specs] == [1, 3]
    assert all(s["prefill"] for s in specs)  # default: legacy behavior


def test_legacy_dialect_unchanged(app_state, direction):
    specs = bs._normalize_steer({"name": "vec", "strength": 4.0,
                                 "layer_from": 1, "layer_to": 2})
    assert specs[0]["name"] == "vec" and specs[0]["layer_to"] == 2
    assert specs[0]["prefill"] and specs[0]["syntax_mute"]


def test_vllm_xargs_request_steers(client, direction):
    base = chat(client, "Tell me something.", max_tokens=12)
    spec = json.dumps({"id": "vec", "layer": 1, "scale": 60.0})
    steered = chat(client, "Tell me something.", max_tokens=12,
                   vllm_xargs={"hotwire": spec})
    assert bs.state["gen"]["steer"][0]["scope"] == "request"
    a = base["choices"][0]["message"]
    b = steered["choices"][0]["message"]
    assert (a.get("content"), a.get("tool_calls")) != \
           (b.get("content"), b.get("tool_calls"))
    # global steering untouched afterwards
    assert bs.state["steer"] is None


def test_vllm_xargs_bad_json_400(client, direction):
    r = client.post("/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "hi"}],
        "vllm_xargs": {"hotwire": "{broken"}})
    assert r.status_code == 400


def test_unknown_id_400(client, direction):
    r = client.post("/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "hi"}],
        "vllm_xargs": {"hotwire": json.dumps({"id": "ghost", "layer": 1,
                                              "scale": 2})}})
    assert r.status_code == 400
    assert "ghost" in r.text


def test_decode_only_skips_prefill_forward(app_state, direction):
    """With in_prefill set, a decode_only hook must be a no-op; the same
    hook must fire once in_prefill clears."""
    model = bs.state["model"]
    ids = torch.tensor([[1, 2, 3, 4]])
    with torch.no_grad():
        clean = model(input_ids=ids).logits

    handles = bs._install_steer_hooks("vec", 60.0, 1, 1, prefill=False)
    try:
        bs.state["in_prefill"] = True
        with torch.no_grad():
            prefill_logits = model(input_ids=ids).logits
        bs.state["in_prefill"] = False
        with torch.no_grad():
            decode_logits = model(input_ids=ids).logits
    finally:
        for h in handles:
            h.remove()
        bs.state["in_prefill"] = False

    assert torch.allclose(clean, prefill_logits), "prefill must stay unsteered"
    assert not torch.allclose(clean, decode_logits), "decode must be steered"


def test_generation_resets_in_prefill_flag(client, direction):
    chat(client, max_tokens=4)
    assert bs.state.get("in_prefill") is False


def test_export_passport(tmp_path):
    dirs = {"calm": [0.1] * 8, "layered": [[0.2] * 8] * 3}
    p = tmp_path / "dirs.json"
    p.write_text(json.dumps(dirs))
    manifest = export(str(p), str(tmp_path / "out"), model="tiny-model")
    by_name = {e["name"]: e for e in manifest["vectors"]}
    assert by_name["calm"]["shape"] == [8]
    assert by_name["layered"]["shape"] == [3, 8]
    assert by_name["calm"]["model"] == "tiny-model"
    assert by_name["calm"]["calibration_regime"]["decode_only_recommended"]
    t = torch.load(tmp_path / "out" / "layered.pt")
    assert t.shape == (3, 8)
    saved = json.loads((tmp_path / "out" / "manifest.json").read_text())
    assert len(saved["vectors"]) == 2


def test_export_unknown_name_raises(tmp_path):
    p = tmp_path / "dirs.json"
    p.write_text(json.dumps({"a": [0.1] * 4}))
    with pytest.raises(KeyError):
        export(str(p), str(tmp_path / "out"), names=["missing"])


def test_parity_compare_report():
    rep = compare([("same text", "same text"),
                   ("one two three four", "one two"),
                   ("abc", "abd")])
    assert rep["n"] == 3 and rep["exact_match"] == 1
    assert rep["first_divergence_chars"] == [None, 7, 2]
    assert rep["mean_words_a"] > rep["mean_words_b"]


def test_replay_ab_with_cos_summary(client, direction):
    spec = {"id": "vec", "layer": 1, "scale": 60.0, "decode_only": True}
    r = client.post("/replay", json={
        "messages": [{"role": "user", "content": "Tell me a story."}],
        "steering": spec, "max_tokens": 8})
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["baseline"]["text"] and d["steered"]["text"]
    assert d["baseline"]["text"] != d["steered"]["text"]
    assert d["steered"]["steer"][0]["scope"] == "request"
    cos = d["steered"]["cos_by_layer"]["vec"]
    assert len(cos) >= 1 and all(0.0 <= c <= 1.0 for c in cos)


def test_replay_unknown_direction_400(client, direction):
    r = client.post("/replay", json={
        "messages": [{"role": "user", "content": "hi"}],
        "steering": {"id": "ghost", "layer": 1, "scale": 2}})
    assert r.status_code == 400


def test_trace_replay_roundtrip(client, direction):
    chat(client, "Original conversation.", max_tokens=6)
    trace_id = bs.state["gen"]["id"]
    r = client.post(f"/traces/{trace_id}/replay", json={
        "steering": {"id": "vec", "layer": 1, "scale": 60.0},
        "max_tokens": 6})
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["baseline"]["text"] and d["steered"]["text"]


def test_trace_replay_predating_messages_409(client, direction, tmp_path):
    chat(client, "Old trace.", max_tokens=4)
    trace_id = bs.state["gen"]["id"]
    # simulate a pre-persistence trace: strip the replay context on disk
    import json as _json
    path = bs.state["traces"].root / f"{trace_id}.json"
    t = _json.loads(path.read_text())
    t.pop("replay", None)
    path.write_text(_json.dumps(t))
    r = client.post(f"/traces/{trace_id}/replay",
                    json={"steering": {"id": "vec", "layer": 1, "scale": 2}})
    assert r.status_code == 409


def test_replay_jlens_suppressed_words(client, direction, fitted_lens):
    bs.state["jlens"] = fitted_lens
    bs.state["jlens_on"] = True
    try:
        r = client.post("/replay", json={
            "messages": [{"role": "user", "content": "Tell me a story."}],
            "steering": {"id": "vec", "layer": 1, "scale": 60.0},
            "max_tokens": 8})
        assert r.status_code == 200, r.text
        d = r.json()
        assert "jlens_suppressed" in d
        for e in d["jlens_suppressed"]:
            assert e["baseline_count"] >= 1 and e["word"]
    finally:
        bs.state["jlens_on"] = False


def test_export_passport_extras(tmp_path):
    p = tmp_path / "dirs.json"
    p.write_text(json.dumps({"calm": [0.1] * 8}))
    extras = {"calm": {"calibrated": {"layer": 2, "scale": 3, "decode_only": True},
                       "eval": {"violations": "0/16"},
                       "recipe": "recipes/calm.json"}}
    manifest = export(str(p), str(tmp_path / "out"), passport=extras)
    e = manifest["vectors"][0]
    assert e["calibrated"]["scale"] == 3 and e["eval"]["violations"] == "0/16"
    assert e["recipe"] == "recipes/calm.json"


def test_forced_replay_zero_strength_is_identical(client, direction):
    """Teacher-forced diff with strength ~0 must suppress NOTHING and
    change nothing — the two passes are literally the same computation."""
    r = client.post("/replay", json={
        "messages": [{"role": "user", "content": "Say a few words."}],
        "steering": {"id": "vec", "layer": 1, "scale": 1e-9},
        "forced": True, "max_tokens": 6})
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["suppressed_positional"] == []
    assert len(d["positions"]) == len(d["tokens"]) > 0


def test_forced_replay_steered_shows_direct_effect(client, direction, fitted_lens):
    bs.state["jlens"] = fitted_lens
    bs.state["jlens_on"] = True
    try:
        r = client.post("/replay", json={
            "messages": [{"role": "user", "content": "Say a few words."}],
            "steering": {"id": "vec", "layer": 1, "scale": 60.0},
            "forced": True, "max_tokens": 6})
        assert r.status_code == 200, r.text
        d = r.json()
        assert d["suppressed_positional"], "huge vector must displace dispositions"
        p = d["positions"][0]
        assert "cos" in p and len(p["cos"]) >= 1
        assert all(-1.0 <= c <= 1.0 for c in p["cos"])
    finally:
        bs.state["jlens_on"] = False


def test_direction_unembed(client, direction):
    r = client.get("/directions/vec/unembed", params={"layer": 0, "top": 10})
    assert r.status_code == 200, r.text
    d = r.json()
    assert len(d["top_up"]) > 0 and len(d["top_down"]) > 0
    assert d["top_up"] != d["top_down"]
    assert client.get("/directions/ghost/unembed").status_code == 404
