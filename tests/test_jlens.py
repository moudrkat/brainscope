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


def test_answer_mode_masks_targets(tok):
    text = "Question here. <think>some reasoning</think> The answer is 42."
    ids = tok(text, return_tensors="pt").input_ids
    m_future = jl._target_mask(text, tok, ids, "future", "</think>")
    m_answer = jl._target_mask(text, tok, ids, "answer", "</think>")
    assert bool(m_future.all())
    assert 0 < int(m_answer.sum()) < int(m_future.sum())
    assert bool(m_answer[-1])          # the answer tail is targeted
    assert not bool(m_answer[0])       # the question is not


def test_answer_mode_fit_runs(model, tok):
    texts = [t + " </think> The final answer is forty two." for t in FIT_TEXTS[:6]]
    lens = jl.fit(model, tok, texts, mode="answer", repeats=4, max_tokens=64)
    assert lens.meta["mode"] == "answer"
    assert lens.meta["marker"] == "</think>"
    assert lens.J.shape[0] == model.config.num_hidden_layers
