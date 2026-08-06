"""Watch a microcontroller's LLM think - no hardware needed.

Serves the 28.9M-param model from an ESP32-S3 in brainscope. The weights are
the exact int4 artifact the chip runs, dequantized and verified against its C
runtime to ~1e-5. Six layers, four heads: the WHOLE model fits on one screen.

    python examples/esp32/serve.py                     # the storyteller
    python examples/esp32/serve.py --flash-unplugged   # 25M flash params zeroed

Things to try once it opens:
  - type a story opening ("Once upon a time") and watch the logit lens
  - drag the "dark" steering slider and darken the story's mood
  - run --flash-unplugged and watch the storyteller collapse to a loop -
    that is what the flash chip contributes, made visible

Extra arguments pass through to brainscope (--port, --no-browser, ...).
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import ple_tinylm  # noqa: F401  (registers ple-tinylm with transformers)

from brainscope import server

WEIGHTS = "unt1l1f1nd/esp32-tinylm"
WEIGHTS_UNPLUGGED = "unt1l1f1nd/esp32-tinylm-flash-unplugged"

extra = sys.argv[1:]
unplugged = "--flash-unplugged" in extra
if unplugged:
    extra.remove("--flash-unplugged")

argv = ["brainscope", "--model", WEIGHTS_UNPLUGGED if unplugged else WEIGHTS]
if "--lens" not in extra:
    argv += ["--lens", "on"]  # 6 layers x 96 dims: the lens is free, keep it on
if "--directions" not in extra and not unplugged:
    argv += ["--directions", str(HERE / "dirs.json")]
sys.argv = argv + extra
server.main()
