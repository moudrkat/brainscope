"""Unit tests for trace segmentation, storage and emergence analytics."""

import torch

from brainscope import traces as tr


def test_think_span_across_token_boundaries():
    pieces = ["Sure", "<th", "ink>", "let", " me", " see", "</th", "ink>", " The", " answer"]
    span = tr.think_span(pieces)
    assert span == [1, 7]
    assert tr.think_span(["no", " think", " here"]) is None


def test_answer_token_after_think():
    trace = {"all_tokens": ["<think>", "hmm", "</think>", "\n", " Paris", " is"],
             "think": [0, 2]}
    idx, piece = tr.answer_token(trace)
    assert (idx, piece) == (4, " Paris")
    idx, piece = tr.answer_token(trace, override="is")
    assert (idx, piece) == (5, " is")


def test_store_roundtrip(tmp_path):
    store = tr.TraceStore(tmp_path, keep=2)
    gen = {"id": "abc123def456", "n_prompt": 3, "prompt_tokens": ["a", "b", "c"],
           "tokens": ["x", "y"], "all_tokens": ["w", "x", "y"],
           "norms": [[1.0], [2.0]], "lens": [], "jlens": [], "tags": {}, "steer": None}
    hidden = [torch.randn(4, 8), torch.randn(4, 8)]
    store.save(gen, "m", hidden)
    assert len(store.list()) == 1
    t = store.load("abc123def456")
    assert t["capture_offset"] == 1
    assert t["has_hidden"]
    h = store.hidden("abc123def456")
    assert h.shape == (2, 4, 8) and h.dtype == torch.float16
    # keep=2 evicts the oldest
    for i in range(3):
        store.save({**gen, "id": f"{i:012x}"}, "m")
    assert len(store.list()) == 2
    assert store.load("abc123def456") is None
    # ids are sanitized — no path traversal
    assert store.load("../evil") is None
    assert not store.delete("../evil")


def test_store_reload_orders_by_time_not_id(tmp_path):
    store = tr.TraceStore(tmp_path)
    base = {"n_prompt": 1, "prompt_tokens": ["a"], "tokens": ["x"],
            "all_tokens": ["x"], "norms": [], "lens": [], "jlens": [],
            "tags": {}, "steer": None}
    # id "ffffff..." saved FIRST (oldest), id "000000..." saved LAST (newest)
    import time as _t
    store.save({**base, "id": "f" * 12}, "m"); _t.sleep(1.1)
    store.save({**base, "id": "0" * 12}, "m")
    reloaded = tr.TraceStore(tmp_path)
    assert [e["id"] for e in reloaded.list()] == ["0" * 12, "f" * 12]  # newest first
    """Fabricated trace: the answer token sits in the stored top-k of the
    later steps only — emergence must find the first step it crosses 0.1."""
    piece = " Paris"
    lens_hist = []
    for step in range(4):
        p = 0.02 if step < 2 else 0.6
        lens_hist.append([[{"t": piece, "p": p}, {"t": " x", "p": 0.01}]
                          for _ in range(model.config.num_hidden_layers)])
    trace = {"all_tokens": ["<think>", "hm", "</think>", piece, " is"],
             "think": [0, 2], "capture_offset": 1,
             "tokens": ["hm", "</think>", piece, " is"], "lens": lens_hist}
    norm, head = model.model.norm, model.get_output_embeddings()
    out = tr.emergence(trace, None, tokenizer=tok, norm=norm, head=head)
    assert out["token"] == piece
    assert out["series"]["logit_lens_topk"] == [0.02, 0.02, 0.6, 0.6]
    assert out["emerge_step"]["logit_lens_topk"] == 2
    assert out["exact"] is False
    # with hidden states, exact series appear (values are model-dependent)
    hidden = torch.randn(4, model.config.num_hidden_layers, model.config.hidden_size)
    out = tr.emergence(trace, hidden, tokenizer=tok, norm=norm, head=head)
    assert out["exact"] is True
    assert len(out["series"]["logit_lens"]) == 4
