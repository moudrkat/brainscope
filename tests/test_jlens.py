"""Unit tests for the Jacobian-lens estimator — the math, not the plumbing."""

import torch

from brainscope import jlens as jl
from tests.conftest import FIT_TEXTS


def test_final_layer_converges_to_identity(fitted_lens):
    """dh_final/dh_final = I exactly, so the estimator's final-layer output
    must line up with the identity — the built-in self-test of the whole
    VJP sampling scheme."""
    d = fitted_lens.hidden
    eye = torch.eye(d)
    cos = torch.nn.functional.cosine_similarity(
        fitted_lens.J[-1].flatten(), eye.flatten(), dim=0)
    assert cos > 0.75, f"J_last vs identity cosine {cos:.3f} — estimator broken"
    assert fitted_lens.meta["identity_error"] < 1.0


def test_shapes_and_metadata(fitted_lens, model):
    n_layers = model.config.num_hidden_layers
    d = model.config.hidden_size
    assert fitted_lens.J.shape == (n_layers, d, d)
    assert fitted_lens.meta["mode"] == "future"
    assert fitted_lens.meta["n_samples"] > 0


def test_save_load_roundtrip(fitted_lens, tmp_path):
    p = tmp_path / "lens.pt"
    fitted_lens.save(p)
    loaded = jl.JacobianLens.load(p)
    assert loaded.J.shape == fitted_lens.J.shape
    assert loaded.meta["mode"] == "future"
    # fp16 storage: close in relative terms, not bit-identical
    assert torch.allclose(loaded.J, fitted_lens.J, rtol=1e-2, atol=1e-2)


def test_transport_and_direction(fitted_lens, model):
    h = torch.randn(fitted_lens.n_layers, fitted_lens.hidden)
    z = fitted_lens.transport(h)
    assert z.shape == h.shape
    w = model.get_output_embeddings().weight.detach().float()
    dirs = fitted_lens.direction(1234, w)
    assert dirs.shape == (fitted_lens.n_layers, fitted_lens.hidden)
    assert torch.allclose(dirs.norm(dim=-1), torch.ones(fitted_lens.n_layers), atol=1e-4)


def test_position_masks_match_reference_reduction(tok):
    """First skip_first positions (attention sinks) and the final position are
    excluded, as in the reference implementation; answer mode additionally
    drops everything before the marker."""
    text = "Question here. <think>some reasoning</think> The answer is 42."
    ids = tok(text, return_tensors="pt").input_ids
    seq = ids.shape[-1]
    m_future = jl._target_mask(text, tok, ids, "future", "</think>", skip_first=4)
    m_answer = jl._target_mask(text, tok, ids, "answer", "</think>", skip_first=4)
    assert not m_future[:4].any() and not m_future[-1]
    assert bool(m_future[4:seq - 1].all())
    assert 0 < int(m_answer.sum()) < int(m_future.sum())
    assert bool(m_answer[-2])          # the answer tail is targeted
    assert not bool(m_answer[0])       # the question is not


def test_decompose_workspace(fitted_lens, model):
    torch.manual_seed(1)
    d = fitted_lens.hidden
    W = model.get_output_embeddings().weight.detach().float()
    # plant two known atoms with positive coefficients, plus noise
    a1 = fitted_lens.direction(777, W)[1] * 5.0
    a2 = fitted_lens.direction(4242, W)[1] * 3.0
    hs = (a1 + a2).unsqueeze(0) + 0.05 * torch.randn(1, d)
    for method in ("gp", "mp"):
        res = fitted_lens.decompose(hs, layer=1, unembed_weight=W, k=6, method=method)
        comps = res[0]["components"]
        ids = [v for v, c in comps]
        assert len(set(ids)) == len(ids)              # no atom twice
        assert 777 in ids, method                     # planted components recovered
        assert 4242 in ids, method
        assert all(c > 0 for v, c in comps), method   # nonnegative
        assert 0 < res[0]["explained"] <= 1.0
    # gp explains at least as much as mp on the same input
    gp = fitted_lens.decompose(hs, 1, W, k=6, method="gp")[0]["explained"]
    mp = fitted_lens.decompose(hs, 1, W, k=6, method="mp")[0]["explained"]
    assert gp >= mp - 0.05


def test_answer_mode_fit_runs(model, tok):
    texts = [t + " </think> The final answer is forty two." for t in FIT_TEXTS[:6]]
    lens = jl.fit(model, tok, texts, mode="answer", repeats=4, max_tokens=64,
                  skip_first=4)
    assert lens.meta["mode"] == "answer"
    assert lens.meta["marker"] == "</think>"
    assert lens.meta["skip_first"] == 4
    assert lens.J.shape[0] == model.config.num_hidden_layers
