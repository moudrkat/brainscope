"""`honours_syntax_mute` — offline tests, no server needed.

The live parity check (test_parity_live.py) compares two backends against each
other. This one is the part that catches a *silently dropped* regime flag,
which a cross-backend diff cannot: hotwire-vLLM parses the spec into
SteerSpec(vector_id, layer, scale) and ignores every other key, so sending
`syntax_mute` there is a no-op with no error. The probe asks one backend
whether it disagrees with itself when only that boolean changes.
"""

import brainscope.parity as parity


def _stub(backend_honours_flag: bool, seen: list):
    """Stand in for a backend that either implements syntax_mute or drops it."""
    def ask(url, model, prompt, spec, timeout=300, tools=None):
        seen.append((spec, tools))
        entry = spec[0] if isinstance(spec, list) else spec
        if backend_honours_flag and entry.get("syntax_mute"):
            return "guarded output"
        return "open output"
    return ask


def test_backend_that_implements_the_flag_is_reported_as_honouring(monkeypatch):
    seen = []
    monkeypatch.setattr(parity, "ask", _stub(True, seen))
    differ, total = parity.honours_syntax_mute(
        "http://x", "m", {"id": "v", "layer": 20, "scale": 3})
    assert (differ, total) == (total, len(parity.TOOL_PROMPTS))
    assert differ > 0


def test_backend_that_drops_the_flag_is_reported_as_ignoring(monkeypatch):
    seen = []
    monkeypatch.setattr(parity, "ask", _stub(False, seen))
    differ, total = parity.honours_syntax_mute(
        "http://x", "m", {"id": "v", "layer": 20, "scale": 3})
    assert differ == 0 and total == len(parity.TOOL_PROMPTS)


def test_probe_sends_the_flag_both_ways_and_always_with_tools(monkeypatch):
    seen = []
    monkeypatch.setattr(parity, "ask", _stub(True, seen))
    parity.honours_syntax_mute("http://x", "m", {"id": "v", "layer": 20, "scale": 3})
    flags = [(s[0] if isinstance(s, list) else s).get("syntax_mute") for s, _ in seen]
    assert flags[:2] == [True, False]
    # without tools the flag has nothing to act on and the probe would be a lie
    assert all(t == parity.TOOL_SCHEMA for _, t in seen)


def test_list_and_dict_specs_both_work(monkeypatch):
    seen = []
    monkeypatch.setattr(parity, "ask", _stub(True, seen))
    parity.honours_syntax_mute("http://x", "m", [{"id": "v", "layer": 20, "scale": 3}])
    assert all(isinstance(s, list) for s, _ in seen)
    seen.clear()
    parity.honours_syntax_mute("http://x", "m", {"id": "v", "layer": 20, "scale": 3})
    assert all(isinstance(s, dict) for s, _ in seen)


def test_callers_spec_is_not_mutated(monkeypatch):
    monkeypatch.setattr(parity, "ask", _stub(True, []))
    spec = {"id": "v", "layer": 20, "scale": 3}
    parity.honours_syntax_mute("http://x", "m", spec)
    assert "syntax_mute" not in spec
