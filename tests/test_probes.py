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
    # EMA equals raw on the first step, then lags it
    first = g["probe"][0]
    assert first["ema"]["d"] == first["scores"]["d"]
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
