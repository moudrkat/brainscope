"""Instruction-hierarchy steering: rescale cached V for the spans that outrank.

Different animal from the direction steering in the rest of the server. There,
a vector is added to the residual stream at a chosen layer, every step. Here
nothing is added anywhere — after the prompt is prefilled we ask, per attention
head, whether it took its cue from the privileged span (the system prompt) or
from a demoted one (conversation left over from before a prompt change), and
for the heads that got it backwards we multiply their *cached value vectors* at
those prompt positions. The edit lives in the KV cache, so decoding is untouched
and costs nothing extra.

Method: V-Steer, arXiv:2607.26228 (Zeng, Lee, Zhao & Hockenmaier, COLM 2026).

Use it when a system prompt keeps losing to something a user said earlier in the
same conversation — the "we changed the rule but the transcript didn't" case.
Not a prompt-injection defence; the messages here are benign, just old.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

GAMMA_PLUS = 2.5    # paper default; end-of-answer rules often need 5-8
GAMMA_MINUS = 0.75


def head_dim(cfg) -> int:
    """Per-head width. NOT always hidden_size // n_heads — Qwen3-4B is hidden
    2560 over 32 heads with head_dim 128, and the naive division fails
    silently."""
    d = getattr(cfg, "head_dim", None)
    return int(d) if d else cfg.hidden_size // cfg.num_attention_heads


def _layers(model):
    for path in (("model", "layers"), ("model", "language_model", "layers"),
                 ("language_model", "model", "layers")):
        obj = model
        try:
            for a in path:
                obj = getattr(obj, a)
        except AttributeError:
            continue
        if hasattr(obj, "__len__") and len(obj):
            return obj
    raise AttributeError("cannot locate decoder layers")


@dataclass
class Plan:
    """Which token positions are privileged, which are demoted, and by how much."""

    boost: list[int] = field(default_factory=list)
    suppress: list[int] = field(default_factory=list)
    gamma_plus: float = GAMMA_PLUS
    gamma_minus: float = GAMMA_MINUS
    eps: float = 0.0
    group_rule: str = "max"
    edit_suppress: bool = True   # False = compare against it, don't rescale it

    def ok(self) -> bool:
        return bool(self.boost and self.suppress)


def plan(tok, messages: list, spec: dict, prompt: str, prompt_ids) -> Plan:
    """Locate the spans named by `spec` in the rendered prompt.

    spec:
      {"stale": [2, 3],            # message indices that lost authority
       "privileged": [0],          # defaults to every system message
       "gamma_plus": 2.5, "gamma_minus": 0.75, "eps": 0}

    or, for a conflict inside a single message:

      {"boost_text": ["YOU CANNOT CREATE TASKS ..."],
       "suppress_text": ["### Character: taskie ..."]}

    Only message *content* is marked; role headers and template scaffolding are
    left alone, which is the paper's V-Simple span strategy (whole message, no
    extraction) and the one that held up in testing.
    """
    stale = set(spec.get("stale") or [])
    priv = set(spec.get("privileged") or
               [i for i, m in enumerate(messages) if m.get("role") == "system"])
    # Substring spans, for conflicts that live *inside* one message. A 12k-token
    # agent system prompt can carry both a rule and a block arguing against it,
    # and message-level marking cannot express that. Given either list, the
    # message-level defaults are dropped -- you asked for something narrower.
    boost_text = [s for s in (spec.get("boost_text") or []) if s]
    suppress_text = [s for s in (spec.get("suppress_text") or []) if s]
    if boost_text or suppress_text:
        priv, stale = set(), set()

    enc = tok(prompt, add_special_tokens=False, return_offsets_mapping=True)
    offsets = list(enc["offset_mapping"])
    # the cache holds the tokenizer's own ids for the prompt; the offsets come
    # from re-encoding the same string, so they line up position for position
    n = min(len(offsets), len(prompt_ids))

    boost: list[int] = []
    suppress: list[int] = []
    cursor = 0
    for i, m in enumerate(messages):
        body = m.get("content") or ""
        if not isinstance(body, str) or not body:
            continue
        start = prompt.find(body, cursor)
        if start < 0:
            continue
        end = start + len(body)
        cursor = end
        idx = [j for j in range(n)
               if offsets[j][1] > start and offsets[j][0] < end
               and offsets[j][1] > offsets[j][0]]
        if i in stale:
            suppress += idx
        elif i in priv:
            boost += idx

    def _spans(needles: list[str]) -> list[int]:
        out: list[int] = []
        for needle in needles:
            start = prompt.find(needle)
            while start >= 0:
                end = start + len(needle)
                out += [j for j in range(n)
                        if offsets[j][1] > start and offsets[j][0] < end
                        and offsets[j][1] > offsets[j][0]]
                start = prompt.find(needle, end)
        return out

    boost += _spans(boost_text)
    suppress += _spans(suppress_text)

    # Boost-only: you name the rule, nothing else. Head selection still needs
    # something to compare against, so the comparison span becomes "everything
    # else in the prompt" -- which is the honest reading of "the rest of the
    # context is outvoting this instruction", and avoids trying to enumerate
    # every span that argues the other way (in a 12k agent prompt there are
    # dozens, and the list is never complete). Only the named span is rescaled.
    if boost and not suppress:
        marked = set(boost)
        suppress = [j for j in range(n) if j not in marked
                    and offsets[j][1] > offsets[j][0]]

    return Plan(boost=boost, suppress=suppress,
                edit_suppress=not (boost_text and not suppress_text),
                gamma_plus=float(spec.get("gamma_plus", GAMMA_PLUS)),
                gamma_minus=float(spec.get("gamma_minus", GAMMA_MINUS)),
                eps=float(spec.get("eps", 0.0)),
                group_rule=spec.get("group_rule", "max"))


@torch.no_grad()
def apply(model, ids, past, plan_: Plan, logits_kw=None):
    """Attribute, edit the cached V, and hand back a fresh forward of the last
    token. Returns (out, report) or (None, reason).

    `past` must already hold the whole prompt. Two extra single-token forwards:
    one under eager attention to read the final position's attention row, one
    after the edit because scaling V changes that position's output too.
    """
    if not plan_.ok():
        return None, {"skipped": "need both a privileged and a demoted span"}

    logits_kw = logits_kw or {}
    device = ids.device
    T = int(ids.shape[1])
    layers = _layers(model)
    cfg = model.config
    H_q, H_kv = cfg.num_attention_heads, cfg.num_key_value_heads
    n_rep, d = H_q // H_kv, head_dim(cfg)

    if not hasattr(past, "layers") or len(past.layers) != len(layers):
        return None, {"skipped": "cache is not one addressable V per layer "
                                 "(layer-shared KV?)"}
    if any(getattr(l, "is_sliding", False) for l in past.layers):
        return None, {"skipped": "sliding-window layers — demoted spans may sit "
                                 "outside the window"}

    def last_forward(want_attn):
        past.crop(T - 1)
        prev = getattr(model.config, "_attn_implementation", None)
        if want_attn and prev != "eager":
            try:
                model.set_attn_implementation("eager")
            except Exception:
                pass
        try:
            return model(input_ids=ids[:, -1:], past_key_values=past,
                         cache_position=torch.tensor([T - 1], device=device),
                         use_cache=True, output_attentions=want_attn, **logits_kw)
        finally:
            if want_attn and prev and prev != "eager":
                try:
                    model.set_attn_implementation(prev)
                except Exception:
                    pass

    probe = last_forward(True)
    if not getattr(probe, "attentions", None) or probe.attentions[0] is None:
        return None, {"skipped": "no attention weights available"}

    # direct logit attribution of the token the model is about to write, split
    # per head into "came from the privileged span" vs "came from the demoted one"
    # the bare unembedding row, matching the paper ("ignoring layer
    # normalization") and their code: r_dirs = lm_head.weight[pred_ids]
    y = int(probe.logits[0, -1].argmax())
    r = model.lm_head.weight[y].detach().float()

    b_idx = torch.tensor(plan_.boost, device="cpu")
    s_idx = torch.tensor(plan_.suppress, device="cpu")
    delta = torch.zeros(len(layers), H_q)
    for l, layer in enumerate(layers):
        alpha = probe.attentions[l][0, :, -1, :].detach().float().cpu()   # [H_q, T]
        W_o = layer.self_attn.o_proj.weight.detach().float()
        u = (W_o.t() @ r).view(H_q, d).cpu()
        v = past.layers[l].values[0].detach().float().cpu()
        v = v.repeat_interleave(n_rep, dim=0)                              # [H_q, T, d]
        c = alpha * (v * u.unsqueeze(1)).sum(-1)                           # [H_q, T]
        delta[l] = c[:, s_idx].sum(-1) - c[:, b_idx].sum(-1)

    # the cache stores one V per KV head, shared by n_rep query heads, so the
    # edit cannot be finer than a group — reduce the group's scores first
    grouped = delta.view(len(layers), H_kv, n_rep)
    # union rule, paper Eq. (9): a KV head is steered if ANY query head in its
    # group is bad. With eps=0 that is exactly max(group) > 0.
    score = grouped.mean(-1) if plan_.group_rule == "mean" else grouped.max(-1).values
    mask = score > plan_.eps

    touched = 0
    for l in range(len(layers)):
        heads = torch.nonzero(mask[l], as_tuple=True)[0]
        if not heads.numel():
            continue
        v = past.layers[l].values
        edits = [(plan_.boost, 1.0 + plan_.gamma_plus)]
        if plan_.edit_suppress:
            edits.append((plan_.suppress, 1.0 - plan_.gamma_minus))
        for idx, m in edits:
            if m == 1.0 or not idx:
                continue
            pos = torch.tensor(idx, device=v.device)
            v[0, heads[:, None], pos[None, :], :] *= m
            touched += len(idx) * int(heads.numel())

    # What the edit actually moves. Scaling V does NOT touch the softmax — the
    # attention weights are identical before and after, which is the whole point
    # of doing it in the cache. What changes is each position's *effective*
    # contribution, alpha * m (paper Eq. 6). So we report both: how the model's
    # attention is split between the two spans per layer, and how that split
    # looks once the multipliers are folded in. That difference is the
    # intervention, made visible.
    mass = {"system": [], "stale": [], "system_after": [], "stale_after": []}
    m_hi, m_lo = 1.0 + plan_.gamma_plus, 1.0 - plan_.gamma_minus
    for l in range(len(layers)):
        a = probe.attentions[l][0, :, -1, :].detach().float().cpu()     # [H_q, T]
        sel = mask[l].repeat_interleave(n_rep)                          # [H_q] bool
        tot = a.sum(-1).clamp(min=1e-9)
        b_raw, s_raw = a[:, b_idx].sum(-1), a[:, s_idx].sum(-1)
        mass["system"].append(round(float((b_raw / tot).mean()), 4))
        mass["stale"].append(round(float((s_raw / tot).mean()), 4))
        # only the selected head groups are rescaled; the rest keep m = 1
        bm = torch.where(sel, torch.tensor(m_hi), torch.tensor(1.0))
        sm = torch.where(sel, torch.tensor(m_lo), torch.tensor(1.0))
        b_eff, s_eff = b_raw * bm, s_raw * sm
        rest = tot - b_raw - s_raw
        tot_eff = (b_eff + s_eff + rest).clamp(min=1e-9)
        mass["system_after"].append(round(float((b_eff / tot_eff).mean()), 4))
        mass["stale_after"].append(round(float((s_eff / tot_eff).mean()), 4))

    out = last_forward(False)          # position T-1 read stale values; redo it
    report = {
        "heads_edited": int(mask.sum()), "heads_total": int(mask.numel()),
        "n_rep": n_rep,
        "boost_tokens": len(plan_.boost), "suppress_tokens": len(plan_.suppress),
        "values_touched": touched,
        "gamma_plus": plan_.gamma_plus,
        "gamma_minus": plan_.gamma_minus if plan_.edit_suppress else 0.0,
        "edit_suppress": plan_.edit_suppress,
        "delta": [[round(x, 4) for x in row] for row in delta.tolist()],
        "mass": mass,
    }
    return out, report
