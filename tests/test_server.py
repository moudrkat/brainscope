"""API tests over the real generation loop (tiny random model, CPU)."""

import torch

from brainscope import server as bs
from tests.conftest import chat


def test_chat_completion_and_capture(client):
    resp = chat(client, "Hello there")
    assert resp["choices"][0]["finish_reason"] in ("stop", "tool_calls")
    g = bs.state["gen"]
    assert g["done"]
    assert len(g["all_tokens"]) >= len(g["tokens"]) > 0
    assert len(g["lens"]) == len(g["tokens"])      # --lens on in fixture
    r = client.get("/gen").json()
    assert r["all_tokens"] == g["all_tokens"]


def test_final_lens_row_matches_sampled_token(client):
    # the top logit-lens row is the sampling distribution itself: under
    # greedy decoding its argmax must be the token that was actually emitted
    # (double-norming the post-norm final hidden state used to flip this
    # at contested tokens)
    chat(client, "Hello there", temperature=0, max_tokens=16)
    g = bs.state["gen"]
    for tok_str, readout in zip(g["tokens"], g["lens"]):
        assert readout[-1][0]["t"] == tok_str


def test_raw_content_keeps_think_block(client):
    resp = chat(client, "Hi", raw=True)
    assert "raw_content" in resp["choices"][0]["message"]


def test_trace_persisted_and_served(client):
    chat(client)
    trace_id = bs.state["gen"]["id"]
    lst = client.get("/traces").json()
    assert [t["id"] for t in lst["traces"]] == [trace_id]
    t = client.get(f"/traces/{trace_id}").json()
    assert t["all_tokens"] and t["capture_offset"] == \
        len(t["all_tokens"]) - len(t["tokens"])
    assert client.delete(f"/traces/{trace_id}").status_code == 200
    assert client.get(f"/traces/{trace_id}").status_code == 404


def test_traces_config_toggles_hidden(client):
    r = client.post("/traces/config", json={"hidden": True}).json()
    assert r["hidden"] is True
    chat(client)
    trace_id = bs.state["gen"]["id"]
    assert client.get(f"/traces/{trace_id}").json()["has_hidden"] is True
    em = client.get(f"/traces/{trace_id}/emergence").json()
    assert em["exact"] is True
    assert "logit_lens" in em["series"]
    assert len(em["series"]["logit_lens"]) == em["n_steps"]
    # save off = nothing persisted
    client.post("/traces/config", json={"save": False})
    chat(client)
    assert len(client.get("/traces").json()["traces"]) == 1


def test_jlens_toggle_requires_lens(client):
    assert client.post("/jlens", json={"on": True}).status_code == 400
    assert client.get("/jlens").json() == {"loaded": False, "on": False}


def test_jlens_readout_streams_and_lands_in_trace(client, fitted_lens):
    bs.state["jlens"], bs.state["jlens_on"] = fitted_lens, True
    info = client.get("/info").json()
    assert info["jlens"] == {"loaded": True, "on": True, "mode": "future"}
    chat(client)
    g = bs.state["gen"]
    assert len(g["jlens"]) == len(g["tokens"])
    assert g["jlens"][0][0][0].keys() == {"t", "p"}    # [step][layer][k]
    t = client.get(f"/traces/{g['id']}").json()
    assert t["jlens"]
    # live off-switch stops the readout
    client.post("/jlens", json={"on": False})
    chat(client)
    assert bs.state["gen"]["jlens"] == []


def test_jlens_direction_feeds_steering(client, fitted_lens):
    bs.state["jlens"] = fitted_lens
    r = client.post("/jlens/direction", json={"text": "cake"})
    assert r.status_code == 200, r.text
    name = r.json()["name"]
    assert name == "j:cake" and name in bs.state["directions"]
    vec = bs.state["directions"][name]
    assert vec.shape == (fitted_lens.n_layers, fitted_lens.hidden)
    s = client.post("/steer", json={"name": name, "strength": 4.0,
                                    "layer_from": 1, "layer_to": 2}).json()
    assert s["active"] and s["steer"][0]["name"] == name
    chat(client)                                       # generates while steered
    assert bs.state["gen"]["steer"][0]["name"] == name
    client.post("/steer", json={})                     # detach

    assert client.post("/jlens/direction", json={"text": ""}).status_code == 400


def test_workspace_decomposition_endpoint(client, fitted_lens):
    bs.state["jlens"] = fitted_lens
    client.post("/traces/config", json={"hidden": True})
    chat(client, max_tokens=6)
    trace_id = bs.state["gen"]["id"]
    r = client.get(f"/traces/{trace_id}/workspace", params={"k": 6, "method": "gp"})
    assert r.status_code == 200, r.text
    ws = r.json()
    assert ws["k"] == 6 and ws["method"] == "gp"
    step = ws["steps"][0]
    assert 0 < step["explained"] <= 1.0
    assert all(c["c"] > 0 for c in step["components"])
    assert all(c["said"] in ("now", "future", "past", "unsaid")
               for c in step["components"])
    # without hidden states → helpful 400
    client.post("/traces/config", json={"hidden": False})
    chat(client, max_tokens=4)
    r = client.get(f"/traces/{bs.state['gen']['id']}/workspace")
    assert r.status_code == 400 and "hidden" in r.json()["error"]


def test_emergence_tracks_overridden_token(client):
    client.post("/traces/config", json={"hidden": True})
    chat(client, max_tokens=8)
    trace_id = bs.state["gen"]["id"]
    piece = next(t for t in bs.state["gen"]["all_tokens"] if t.strip())
    em = client.get(f"/traces/{trace_id}/emergence",
                    params={"token": piece.strip()}).json()
    assert piece.strip() in em["token"]
