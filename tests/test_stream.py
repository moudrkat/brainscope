"""`stream: true` must add up to the non-streamed reply: no end-of-turn marker in the
text, reasoning apart from the answer, and a tool call as `tool_calls`, not as text.
The generation is scripted (a fake generate_with_signals feeding the server the tokens
a real model would), so each case is exact; one test also streams the tiny model."""

import json

import pytest

from brainscope import server as bs
from tests.conftest import chat


def stream(client, **body):
    body.setdefault("messages", [{"role": "user", "content": "hi"}])
    with client.stream("POST", "/v1/chat/completions", json={"stream": True, **body}) as r:
        lines = [l for l in r.iter_lines() if l.startswith("data: ")]
    assert lines[-1] == "data: [DONE]"
    chunks = [json.loads(l[6:]) for l in lines[:-1]]
    deltas = [c["choices"][0]["delta"] for c in chunks]
    return {
        "content": "".join(d.get("content") or "" for d in deltas),
        "reasoning": "".join(d.get("reasoning_content") or "" for d in deltas),
        "tool_calls": [tc for d in deltas for tc in d.get("tool_calls") or []],
        "finish": chunks[-1]["choices"][0]["finish_reason"],
        "chunks": chunks,
    }


@pytest.fixture
def scripted(client, monkeypatch):
    """Make the server 'generate' a given list of token pieces (and return their text,
    with the seeded tool-call opening in front when the request forces a tool)."""
    def use(pieces, prefix=""):
        def fake(messages, tools, max_new_tokens, temperature, notify, *rest):
            bs.state["last_generated"] = list(range(len(pieces)))
            for i, p in enumerate(pieces):
                notify({"type": "token", "i": i, "text": p,
                        "lp": {"logprob": -0.5, "top": [[p, -0.5], ["x", -2.0]]}})
            notify({"type": "done"})
            return prefix + "".join(pieces)
        monkeypatch.setattr(bs, "generate_with_signals", fake)
    return use


TOOLS = [{"type": "function", "function": {"name": "get_weather", "parameters": {
    "type": "object", "properties": {"city": {"type": "string"}}}}}]


def test_end_of_turn_marker_is_not_text(client, scripted):
    scripted(["Red", ",", " Yellow", ",", " Blue", "<|im_end|>"])
    s = stream(client)
    assert s["content"] == "Red, Yellow, Blue"
    assert s["finish"] == "stop"


def test_stream_equals_plain_reply(client, scripted):
    pieces = ["\n", "Sure", ":", " one", "\n", "two", "\n\n", "<|im_end|>"]
    scripted(pieces)
    plain = chat(client, "hi", max_tokens=50)["choices"][0]["message"]["content"]
    scripted(pieces)
    assert stream(client, max_tokens=50)["content"] == plain == "Sure: one\ntwo"


def test_reasoning_goes_apart(client, scripted):
    scripted(["<think>", "\nlet me", " see", "\n", "</think>", "\n\n", "Paris", "<|im_end|>"])
    s = stream(client)
    assert s["content"] == "Paris"
    assert s["reasoning"] == "\nlet me see\n"


def test_auto_tool_call_streams_as_tool_calls(client, scripted):
    scripted(["<tool_call>", "\n", '{"name": "get_weather", ', '"arguments": {"city": "Prague"}}',
              "\n", "</tool_call>", "<|im_end|>"])
    s = stream(client, tools=TOOLS)
    assert s["content"] == ""
    assert s["finish"] == "tool_calls"
    (call,) = s["tool_calls"]
    assert call["index"] == 0 and call["type"] == "function" and call["id"]
    assert call["function"]["name"] == "get_weather"
    assert json.loads(call["function"]["arguments"]) == {"city": "Prague"}


def test_forced_tool_call_without_its_seeded_opening(client, scripted):
    # tool_choice "required": the server seeds '<tool_call>\\n{"name": ' into the prompt,
    # so the model only writes the rest; the opening never arrives as a token
    scripted([' "get_weather"', ', "arguments": {"city": "Prague"}}', "\n", "</tool_call>", "<|im_end|>"],
             prefix='<tool_call>\n{"name":')
    s = stream(client, tools=TOOLS, tool_choice="required")
    assert s["content"] == ""
    assert [c["function"]["name"] for c in s["tool_calls"]] == ["get_weather"]
    assert s["finish"] == "tool_calls"


def test_text_then_tool_call(client, scripted):
    scripted(["Let me", " check", ".", "\n", "<tool", "_call>", '\n{"name": "get_weather", "arguments": {}}',
              "\n</tool_call>", "<|im_end|>"])
    s = stream(client, tools=TOOLS)
    assert s["content"] == "Let me check."          # the half-written '<tool' never shows
    assert s["tool_calls"][0]["function"]["name"] == "get_weather"


def test_text_answer_with_tools_offered_still_streams(client, scripted):
    scripted(["It", " is", " sunny", ".", "<|im_end|>"])
    s = stream(client, tools=TOOLS)
    assert s["content"] == "It is sunny."
    assert s["tool_calls"] == [] and s["finish"] == "stop"
    assert sum(1 for c in s["chunks"] if c["choices"][0]["delta"].get("content")) > 1   # really streamed


def test_logprobs_follow_the_text(client, scripted):
    scripted(["A", " B", "<|im_end|>"])
    s = stream(client, logprobs=True, top_logprobs=2)
    toks = [lp["token"] for c in s["chunks"] for lp in (c["choices"][0].get("logprobs") or {}).get("content", [])]
    assert toks == ["A", " B"]                       # none for the end-of-turn marker
    tops = [len(lp["top_logprobs"]) for c in s["chunks"] for lp in (c["choices"][0].get("logprobs") or {}).get("content", [])]
    assert set(tops) == {2}


def test_tiny_model_stream_matches_plain(client):
    plain = chat(client, "Hello there", max_tokens=12)["choices"][0]["message"]["content"] or ""
    s = stream(client, messages=[{"role": "user", "content": "Hello there"}], max_tokens=12)
    assert s["content"] == plain
    assert "<|im_end|>" not in s["content"]
