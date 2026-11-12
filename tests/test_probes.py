"""Cheap activation probes: hook-based per-token scalar readouts that stay
alive with viz off, and the trip that flips the full capture on when one
fires (the probe → deep-dive cascade)."""

import torch

from brainscope import server as bs
from tests.conftest import chat


def _arm(client, **spec):
    torch.manual_seed(1)
    bs.state["directions"]["d"] = torch.randn(4, 64)
    r = client.post("/probes", json={"probes": [{"direction": "d", "layer": 2, **spec}]})
    assert r.status_code == 200, r.text
    return r.json()


def test_probe_scores_streamed_and_traced(client):
    _arm(client)
    chat(client, max_tokens=8)
    g = bs.state["gen"]
    assert g["probe"], "probe series recorded"
    assert all(isinstance(e["scores"]["d"], float) for e in g["probe"])
    # step 0 comes straight out of prefill — uncaptured, like norms/lens
    assert g["probe"][0]["i"] >= 1
    trace = bs.state["traces"].load(g["id"])
    assert trace["probe"] == g["probe"]


def test_probe_runs_in_dark_mode(client):
    _arm(client)
    bs.state["viz"] = False
    chat(client, max_tokens=8)
    g = bs.state["gen"]
    assert not g["norms"], "heavy capture stayed off"
    assert g["probe"], "cheap probe still on"


def test_probe_trip_flips_viz(client):
    # threshold below any possible score -> fires on the first decode step
    _arm(client, threshold=-1e9, trip="viz")
    bs.state["viz"] = False
    chat(client, max_tokens=10)
    g = bs.state["gen"]
    assert g["probe"] and g["probe"][0]["fired"] == ["d"]
    assert bs.state["viz"] is True
    assert g["norms"], "full capture kicked in after the trip"


def test_probe_disabled_is_silent(client):
    _arm(client)
    client.post("/probes", json={"enabled": False})
    chat(client, max_tokens=8)
    assert not bs.state["gen"]["probe"]


def test_probe_ema_smoothed_series(client):
    _arm(client, ema=0.2, threshold=-1e9, trip="viz")
    chat(client, max_tokens=10)
    g = bs.state["gen"]
    assert g["probe"], "series recorded"
    # every recorded step carries both the raw score and the smoothed level
    assert all("ema" in e and "d" in e["ema"] for e in g["probe"])
    # the answer's EMA continues from the PROMPT's last smoothed level (one
    # monitor over the whole exchange), so on the first step it lags the raw
    # score toward that level instead of equalling it
    first = g["probe"][0]
    prompt_level = g["probe_prompt"]["d"]["e"][-1]
    expect = prompt_level * (1 - 0.2) + first["scores"]["d"] * 0.2
    assert abs(first["ema"]["d"] - expect) < 1e-3
    assert first["fired"] == ["d"]   # threshold -1e9 fires on the smoothed level too


def test_probe_unknown_direction_rejected(client):
    r = client.post("/probes", json={"probes": [{"direction": "nope", "layer": 1}]})
    assert r.status_code == 400


def test_probe_train_factory(client):
    # the probe factory: two contrast personas -> direction + report + armed
    r = client.post("/probes/train", json={
        "name": "testdim",
        "pos_system": "Always answer with great enthusiasm!",
        "neg_system": "Answer calmly and factually.",
        "prompts": ["Hi there.", "How are you?", "Tell me a fact.", "What now?"],
        "max_tokens": 6, "layer": 2,
        "arm": {"ema": 0.2, "trip": "viz"},
    })
    assert r.status_code == 200, r.text
    rep = r.json()
    assert rep["n"] == 8 and rep["layer"] == 2
    assert "auc_holdout" in rep and "threshold" in rep
    assert rep.get("armed") is True
    assert "testdim" in bs.state["directions"]
    assert any(s["name"] == "testdim" for s in bs.state["vprobes"])


def test_probe_train_validates_input(client):
    r = client.post("/probes/train", json={"name": "x", "prompts": ["a"]})
    assert r.status_code == 400


def test_probe_default_layer_from_meta(client):
    torch.manual_seed(2)
    bs.state["directions"]["m"] = torch.randn(4, 64)
    bs.state["dir_meta"]["m"] = {"layer_from": 1, "layer_to": 3}
    r = client.post("/probes", json={"probes": [{"direction": "m"}]})
    assert r.json()["probes"][0]["layer"] == 1


def test_probe_reads_the_prompt_before_token_zero(client):
    """An input monitor: the prompt's positions are scored during prefill and
    reported before any token is decoded, with the position where the
    smoothed level first crossed the threshold."""
    _arm(client, ema=0.5, threshold=-1e9)   # threshold so low that the first position fires
    chat(client, text="Write a loop that counts to ten.", max_tokens=4)
    g = bs.state["gen"]
    pp = g.get("probe_prompt")
    assert pp and "d" in pp, "the prompt reading is kept with the generation"
    n = len(pp["d"]["s"])
    assert n >= 8, f"one score per prompt position, got {n}"
    assert len(pp["d"]["e"]) == n
    assert pp["d"]["fired_at"] == 0
    # decode-only series is untouched by the prompt positions
    assert g["probe"] and g["probe"][0]["i"] >= 1
    assert len(g["probe"]) < n


def test_prompt_reading_is_the_users_message_not_the_template(client):
    """The span is the last user message: the template around it and the start-of-turn
    sink at position 0 are not read, and the positions say where in the prompt it sits."""
    _arm(client, ema=0.5, threshold=1e9)
    text = "Write a loop that counts to ten."
    chat(client, text=text, max_tokens=2)
    pp = bs.state["gen"]["probe_prompt"]["d"]
    tok = bs.state["tokenizer"]
    assert pp["span"] == "user"
    assert 0 < pp["start"] < pp["end"] <= pp["n"]
    assert pp["end"] - pp["start"] == len(tok(text, add_special_tokens=False).input_ids) == len(pp["s"])
    assert pp["fired_at"] is None   # threshold out of reach
    # one text per read position, so the page can colour the message itself
    assert len(pp["tokens"]) == len(pp["s"]) and "".join(pp["tokens"]) == text


def test_prompt_reading_without_a_user_message_skips_the_sink(client):
    _arm(client, threshold=1e9)
    r = client.post("/v1/chat/completions", json={
        "max_tokens": 2, "messages": [{"role": "system", "content": "You count things."}]})
    assert r.status_code == 200
    pp = bs.state["gen"]["probe_prompt"]["d"]
    assert pp["span"] == "prompt" and pp["start"] == 1 and len(pp["s"]) == pp["n"] - 1


def test_prompt_reading_is_kept_in_the_trace_and_in_gen(client):
    _arm(client, ema=0.5, threshold=-1e9)
    chat(client, text="Count to ten.", max_tokens=2)
    g = client.get("/gen").json()
    assert g["probe_prompt"]["d"]["fired_at"] == 0
    tid = bs.state["traces"].list()[0]["id"]
    assert bs.state["traces"].load(tid)["probe_prompt"]["d"]["s"] == g["probe_prompt"]["d"]["s"]


def test_prompt_firing_trips_the_instruments_before_the_answer(client):
    bs.state["viz"] = False
    _arm(client, threshold=-1e9, trip="viz")
    chat(client, text="Count to ten.", max_tokens=3)
    assert bs.state["viz"] is True
    assert bs.state["gen"]["probe_prompt"]["d"]["fired_at"] == 0


def test_a_probe_trained_elsewhere_comes_in_as_it_is(client):
    """vector + bias + hidden_state, the way a linear probe leaves sklearn or torch:
    the layer is hidden_state - 1, the score is h·v + bias, and it equals the unit
    projection scaled back by |v| and shifted by the bias."""
    torch.manual_seed(3)
    v = torch.randn(64)
    r = client.post("/probes", json={"probes": [
        {"name": "lin", "vector": v.tolist(), "bias": -1.5, "hidden_state": 3, "ema": 0.5},
        {"name": "unit", "vector": v.tolist(), "hidden_state": 3, "metric": "proj"}]})
    assert r.status_code == 200, r.text
    armed = {p["name"]: p for p in r.json()["probes"]}
    assert armed["lin"]["layer"] == 2 and armed["lin"]["metric"] == "logit"
    assert "vector" not in armed["lin"]                      # the response stays readable
    chat(client, text="Count to ten.", max_tokens=4)
    g = bs.state["gen"]
    norm = float(v.norm())
    for rec in g["probe"]:
        assert abs(rec["scores"]["lin"] - (rec["scores"]["unit"] * norm - 1.5)) < 1e-2
    pp = g["probe_prompt"]
    for a, b in zip(pp["lin"]["s"], pp["unit"]["s"]):
        assert abs(a - (b * norm - 1.5)) < 1e-2


def test_probe_spec_errors(client):
    v = [0.1] * 64
    assert client.post("/probes", json={"probes": [{"name": "x", "vector": v, "hidden_state": 0}]}).status_code == 400
    assert client.post("/probes", json={"probes": [{"name": "x", "vector": v, "hidden_state": 2, "layer": 1}]}).status_code == 400
    assert client.post("/probes", json={"probes": [{"name": "x", "vector": v, "metric": "nope"}]}).status_code == 400
