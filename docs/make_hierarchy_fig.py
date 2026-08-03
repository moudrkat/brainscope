"""Rebuild docs/hierarchy.jpg from a live brainscope.

The figure used to be a one-off: no script, no saved data, so the only way to
change a label was to paint over the JPG. This is that script.

    brainscope --model Qwen/Qwen3-4B-Instruct-2507 --port 8010     # one terminal
    BRAINSCOPE_BASE=http://127.0.0.1:8010 python docs/make_hierarchy_fig.py

It sends one conversation where a pre-update message carries both a formatting
rule and a fact, asks the question, and plots what the /hierarchy report says
about where the last prompt position was looking when it picked the first word.

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
OUT = os.path.join(os.path.dirname(__file__), "hierarchy.jpg")

# Same case as the LinkedIn post and the recall eval: one pre-update message
# carries the stale rule, another carries a fact, and the question is about the
# fact. Indices 1..4 are the pre-update block.
MESSAGES = [
    {"role": "system", "content": "Always reply in ALL UPPERCASE LETTERS."},
    {"role": "user", "content": "From now on always reply in all lowercase "
                                "letters, never shout at me."},
    {"role": "assistant", "content": "understood, I will do that from now on."},
    {"role": "user", "content": "My order number is 4417-B."},
    {"role": "assistant", "content": "noted, I will remember that."},
    {"role": "user", "content": "What is my order number?"},
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


def main():
    post("/v1/chat/completions",
         {"messages": MESSAGES, "max_tokens": 8, "temperature": 0,
          "hierarchy": SPEC})
    rep = (get("/hierarchy").get("last") or {})
    if not rep:
        raise SystemExit("no hierarchy report -- is this build of brainscope "
                         "the one with the hierarchy tab?")
    mass = rep["mass"]
    old, new = mass["stale"], mass["system"]
    xs = list(range(len(old)))
    ratio = (sum(old) / len(old)) / max(sum(new) / len(new), 1e-9)

    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "system-ui", "sans-serif"]
    fig = plt.figure(figsize=(11, 6.2), dpi=160)
    fig.patch.set_facecolor(WHITE)
    fig.text(0.055, 0.945,
             "The model was mostly reading the old message, not the new rule",
             fontsize=20, fontweight="700", color=INK, va="top")
    fig.text(0.055, 0.878,
             "The system prompt says reply in capitals. A message from before "
             "the update says reply in lowercase,\nand mentions an order "
             "number. This is where the model looked as it began its answer.",
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

    ax.set_ylim(0, max(max(old), max(new)) * 1.38)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v * 100:.0f}%")
    ax.set_ylabel("share of the model's attention", color=INK2, fontsize=10.5,
                  labelpad=9)
    ax.set_xlabel("depth in the model  (each step is one layer, input on the "
                  "left)", color=INK2, fontsize=10.5, labelpad=9)

    fig.text(0.085, 0.075,
             f"The old message took {ratio:.1f}× more of the model's "
             f"attention than the system prompt.",
             fontsize=12.5, color=INK, fontweight="700")
    fig.text(0.085, 0.028,
             "Qwen3-4B-Instruct-2507 · one conversation · brainscope",
             fontsize=9, color=MUTED)

    fig.savefig(OUT, facecolor=WHITE)
    plt.close(fig)
    print(f"{OUT}  ({ratio:.2f}x, {len(xs)} layers)")


if __name__ == "__main__":
    main()
