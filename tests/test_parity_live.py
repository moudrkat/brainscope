"""Standing parity check — runs only when both backend URLs are provided.

    PARITY_BRAINSCOPE_URL=http://localhost:8010 \
    PARITY_HOTWIRE_URL=http://gpu-box:8001 PARITY_HOTWIRE_MODEL=qwen3-4b \
    pytest tests/test_parity_live.py -q

Skipped in normal runs; wire it to a cron/CI job pointing at live servers
to catch the two backends' steering semantics drifting apart.
"""

import os

import pytest

from brainscope.parity import NEUTRAL_PROMPTS, ask, compare

B_URL = os.environ.get("PARITY_BRAINSCOPE_URL")
H_URL = os.environ.get("PARITY_HOTWIRE_URL")

pytestmark = pytest.mark.skipif(
    not (B_URL and H_URL), reason="set PARITY_*_URL env vars to run live parity")


def test_backends_agree_on_unsteered_shape():
    prompts = NEUTRAL_PROMPTS[:3]
    pairs = [(ask(B_URL, os.environ.get("PARITY_BRAINSCOPE_MODEL", "steered"), p, None),
              ask(H_URL, os.environ["PARITY_HOTWIRE_MODEL"], p, None))
             for p in prompts]
    rep = compare(pairs)
    assert rep["n"] == 3
    assert all(a and b for a, b in pairs), "both backends must answer"
    # behavior guard, not token equality: outputs within 4x length of each other
    assert 0.25 <= rep["length_ratio"] <= 4.0
