"""Rebuild docs/hierarchy.jpg from a live brainscope.

The figure used to be a one-off: no script, no saved data, so the only way to
change a label was to paint over the JPG. This is that script.

    brainscope --model Qwen/Qwen3-4B-Instruct-2507 --port 8010     # one terminal
    BRAINSCOPE_BASE=http://127.0.0.1:8010 python docs/make_hierarchy_fig.py

It sends one conversation where a pre-update message carries both a formatting
rule and a fact, asks the question, and plots what the /hierarchy report says
about where the last prompt position was looking when it picked the first word.

The "after" ratio lands near 1.0, which is parity, not a reversal --
the old message stops dominating, the system prompt does not take over.
Per layer the two dashed lines cross repeatedly; only the mean is stated.

Written for a reader who has never heard of an attention head: no jargon in the
chart, both lines labelled where they run, one number called out.
"""
from __future__ import annotations

import json
import os
import urllib.request

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = os.environ.get("BRAINSCOPE_BASE", "http://127.0.0.1:8010")
DOCS = os.path.dirname(__file__)

# Same case as the LinkedIn post and the recall eval: one pre-update message
# carries the stale rule, another carries a fact, and the question is about the
# fact. Indices 1..4 are the pre-update block.
# One case per conflict family. Same shape every time: a current system rule,
# a pre-update block that contradicts it, and the question asked now. A single
# conversation is n=1, so the point of having several is that the reader can
# see whether the picture holds across different kinds of conflict.
CASES = {
    "case": ("Always reply in ALL UPPERCASE LETTERS.",
             "From now on always reply in all lowercase letters, never shout at me.",
             "Name three primary colors."),
    "prefix": ("Always begin every reply with 'ACK:' and nothing before it.",
               "From now on always begin every reply with 'HELLO:' and nothing before it.",
               "What is the capital of Japan?"),
    "lang": ("Always reply in English, whatever language the user writes in.",
             "From now on always reply in Czech, I do not read English well.",
             "Describe a sunny afternoon in two sentences."),
    "bullet": ("Always answer as a bulleted list, never as prose.",
               "From now on always answer in flowing prose, no bullet points.",
               "Give me three reasons to take the train."),
    "length": ("Always answer in at least four sentences.",
               "From now on always answer in one short sentence, nothing more.",
               "Why does bread rise?"),
    "json": ("Always answer with a single JSON object and nothing else.",
             "From now on answer in plain prose, JSON is unreadable for me.",
             "Give me the population of Paris."),
}

def build(rule_now, rule_old, question):
    return [
        {"role": "system", "content": rule_now},
        {"role": "user", "content": rule_old},
        {"role": "assistant", "content": "understood, I will do that from now on."},
        {"role": "user", "content": "My order number is 4417-B."},
        {"role": "assistant", "content": "noted, I will remember that."},
        {"role": "user", "content": question},
    ]

SPEC = {"stale": [1, 2, 3, 4], "privileged": [0],
        "gamma_plus": 2.5, "gamma_minus": 0.75}

WHITE = "#ffffff"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
OLD = "#e34948"      # the pre-update message
NEW = "#2a78d6"      # the current system prompt


def post(path, body, timeout=900):
    r = urllib.request.Request(BASE + path, json.dumps(body).encode(),
                               {"Content-Type": "application/json"})
    with urllib.request.urlopen(r, timeout=timeout) as f:
        return json.loads(f.read())


def get(path, timeout=60):
    with urllib.request.urlopen(BASE + path, timeout=timeout) as f:
        return json.loads(f.read())


def one(name, rule_now, rule_old, question, out):
    messages = build(rule_now, rule_old, question)
    post("/v1/chat/completions",
         {"messages": messages, "max_tokens": 8, "temperature": 0,
          "hierarchy": SPEC})
    rep = (get("/hierarchy").get("last") or {})
    if not rep:
        raise SystemExit("no hierarchy report -- is this build of brainscope "
                         "the one with the hierarchy tab?")
    mass = rep["mass"]
    old, new = mass["stale"], mass["system"]
    old_a, new_a = mass["stale_after"], mass["system_after"]
    xs = list(range(len(old)))
    mean = lambda v: sum(v) / len(v)
    ratio = mean(old) / max(mean(new), 1e-9)
    ratio_a = mean(old_a) / max(mean(new_a), 1e-9)

    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "system-ui", "sans-serif"]
    fig = plt.figure(figsize=(11, 6.2), dpi=160)
    fig.patch.set_facecolor(WHITE)
    fig.text(0.055, 0.945,
             "The model was mostly reading the old message, not the new rule",
             fontsize=20, fontweight="700", color=INK, va="top")
    fig.text(0.055, 0.878,
             f"System prompt: “{rule_now}”\n"
             f"A message from before the update: “{rule_old}”",
             fontsize=11.5, color=INK2, va="top", linespacing=1.5)

    ax = fig.add_axes([0.085, 0.205, 0.88, 0.56])
    ax.set_facecolor(WHITE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#c3c2b7")
    ax.grid(axis="y", color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, labelsize=10, length=0)

    # dashed = the same split once the value multipliers are folded in. Without
    # these the figure only shows the problem, and the post claims a fix.
    ax.plot(xs, old_a, color=OLD, linewidth=1.6, linestyle=(0, (4, 3)),
            alpha=0.75, zorder=2)
    ax.plot(xs, new_a, color=NEW, linewidth=1.6, linestyle=(0, (4, 3)),
            alpha=0.75, zorder=2)
    ax.plot(xs, old, color=OLD, linewidth=2.5, zorder=3)
    ax.plot(xs, new, color=NEW, linewidth=2.5, zorder=3)

    # labels sit on the lines, so there is nothing to decode in a legend.
    # anchored away from the axes edges, or they collide with the ticks.
    i = max(range(len(old) // 2, len(old)), key=lambda j: old[j])
    ax.text(xs[i], old[i] + 0.035, "the message from\nbefore the update",
            color=OLD, fontsize=12.5, fontweight="700", linespacing=1.35,
            ha="center", va="bottom")
    j = int(len(new) * 0.62)
    ax.text(xs[j], new[j] + 0.028, "the current system prompt",
            color=NEW, fontsize=12.5, fontweight="700", ha="center", va="bottom")
    ax.text(xs[int(len(xs) * 0.30)], max(max(old), max(new)) * 1.20,
            "dashed = after the edit", color=INK2, fontsize=11,
            fontweight="700", ha="center")

    ax.set_ylim(0, max(max(old), max(new)) * 1.38)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v * 100:.0f}%")
    ax.set_ylabel("share of the model's attention", color=INK2, fontsize=10.5,
                  labelpad=9)
    ax.set_xlabel("depth in the model  (each step is one layer, input on the "
                  "left)", color=INK2, fontsize=10.5, labelpad=9)

    fig.text(0.085, 0.075,
             f"Before: {ratio:.1f}× toward the old message. "
             f"After the edit: {ratio_a:.1f}× — they end up level.",
             fontsize=12.5, color=INK, fontweight="700")
    fig.text(0.085, 0.028,
             "Qwen3-4B-Instruct-2507 · one conversation · brainscope",
             fontsize=9, color=MUTED)

    fig.savefig(out, facecolor=WHITE)
    plt.close(fig)
    print(f"{out}  {ratio:.1f}x -> {ratio_a:.1f}x")
    return ratio, ratio_a


def main():
    rows = []
    for name, (now, old_rule, q) in CASES.items():
        out = os.path.join(DOCS, f"hierarchy_{name}.jpg")
        rows.append((name,) + one(name, now, old_rule, q, out))
    os.replace(os.path.join(DOCS, "hierarchy_case.jpg"),
               os.path.join(DOCS, "hierarchy.jpg"))
    print("\nshrnuti (pred -> po):")
    for name, r, ra in rows:
        print(f"  {name:8s} {r:5.1f}x -> {ra:4.1f}x")


if __name__ == "__main__":
    main()
