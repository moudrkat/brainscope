#!/usr/bin/env python3
"""Don't trust the violet cells — audit them.

For a stored trace (or the server's last generation), replay the exact
criterion the UI uses and verify every would-be violet cell against the
text that was actually generated: a cell counts only if the word it
DISPLAYS (its top-1 readout) was emitted later in the answer. "Strong"
hits precede the word's first occurrence; "echo" hits are recurrences of
a word already said once. Top-5 evidence (the "saw coming" strip) is
listed separately.

    python examples/audit_jlens_hits.py traces/<id>.json
    python examples/audit_jlens_hits.py http://localhost:8010   # last gen
"""

import json
import sys
import urllib.request
from pathlib import Path


def load(source: str) -> dict:
    if source.startswith("http"):
        with urllib.request.urlopen(source.rstrip("/") + "/gen", timeout=30) as r:
            return json.load(r)
    return json.loads(Path(source).read_text())


def audit(t: dict) -> None:
    toks = t["all_tokens"]
    jlens = t.get("jlens") or []
    if not jlens:
        sys.exit("no J-lens data in this trace (serve with --jlens and the ◎ switch on)")
    off = len(toks) - len(t["tokens"])
    print("answer:", "".join(toks).replace("\n", " ")[:120])
    cell_hits, strip_hits = [], []
    for c in range(len(jlens)):
        later = toks[c + off + 1:]
        past = toks[: c + off + 1]
        for l, layer in enumerate(jlens[c]):
            for rank, e in enumerate(layer):
                w = e["t"]
                if not w.strip() or w == t["tokens"][c]:
                    continue
                if rank > 0 and len(w.strip()) < 3:
                    continue
                if w in later:
                    row = (c, t["tokens"][c], l + 1, w, e["p"], w not in past)
                    (cell_hits if rank == 0 else strip_hits).append(row)

    for name, rows in (("violet cells (displayed word arrived later)", cell_hits),
                       ("strip/tooltip evidence (word was in a cell's top-5)", strip_hits)):
        strong = [r for r in rows if r[5]]
        print(f"\n{name}: {len(rows)} · strong (before first occurrence): {len(strong)}")
        seen = set()
        for c, emitted, layer, w, p, first in rows:
            key = (emitted, w, first)
            if key in seen:
                continue
            seen.add(key)
            tag = "STRONG" if first else "echo  "
            print(f"  {tag} writing {emitted!r:12} L{layer:2} saw {w!r} (p={p})")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    audit(load(sys.argv[1]))
