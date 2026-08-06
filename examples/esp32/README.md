# Watch a microcontroller's LLM think


This model normally lives on an **ESP32-S3 microcontroller** - a $5 chip with
512 KB of RAM - where it writes children's stories at 9.88 tokens/s on a
matchbox-sized board ([slvDev/esp32-ai](https://github.com/slvDev/esp32-ai)).
It fits because 25M of its 28.9M parameters sit in the chip's **flash memory**
(Per-Layer Embeddings, the Gemma 3n trick) and are read ~450 bytes per token.

Here you get those exact weights - the int4 artifact the chip runs, dequantized
and verified against its C runtime to ~1e-5 - under brainscope's microscope.
Six layers, four heads: **the whole model fits on one screen.** No cherry-picked
attention heads, no truncated views. It is the perfect glass-box model for
learning what the logit lens, attention maps and steering actually show.

## Run it (no hardware needed)

```bash
pip install brainscope
python examples/esp32/serve.py
```

Weights download from the Hub on first run (~120 MB fp32).

## Three experiments

1. **Watch a story crystallize.** Type `Once upon a time` in the chat box and
   watch each word surface through the six layers in the logit lens.
2. **Steer the mood.** Turn steering on, pick the bundled `dark` direction and
   drag the slider - the story clouds over: storms, tears, night. Extracted
   from 10 contrast pairs at layer 3 (`brainscope.extract`).
3. **Unplug the flash.** `python examples/esp32/serve.py --flash-unplugged`
   zeroes the 25M flash-resident parameters. The storyteller collapses into
   `time there time there time...` - a direct picture of what a memory chip
   contributes to a model's thinking.

## Credit

Model and hardware story: [slvDev/esp32-ai](https://github.com/slvDev/esp32-ai)
(MIT). The conversion and the verification gate against the device's C runtime
live there in `brainscope_adapter/`. Per-Layer Embeddings are Google's design
from [Gemma 3n](https://ai.google.dev/gemma/docs/gemma-3n); the model trains on
[TinyStories](https://arxiv.org/abs/2305.07759).
