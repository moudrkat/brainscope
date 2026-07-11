#!/usr/bin/env bash
# Reproducible J-lens fits — the exact invocations used for the published
# artifacts. Fit where the GPU is; artifacts land in lenses/ (gitignored,
# fp16, reusable forever). See docs/jlens.md.
#
# Timing reference (one 16 GB consumer GPU): tiny ≈ 2 min, qwen3-4b ≈ 25 min.
# CPU works but takes hours.
set -euo pipefail
mkdir -p lenses

pip show datasets >/dev/null 2>&1 || pip install -q datasets   # wikitext source

# -- J-lens: Qwen2.5-0.5B-Instruct (the `tiny` preset) ----------------------
# 256 texts × 24 probes = 6144 samples ≫ 2×896 hidden — healthy budget
brainscope-jlens fit --model tiny --prompts wikitext \
    --out lenses/qwen2.5-0.5b-instruct.jlens.pt --n-texts 256 --repeats 24

# -- J-lens: Qwen3-4B-Instruct-2507 (the `qwen3-4b` preset) -----------------
# hidden 2560 → aim well above 2×hidden samples; 640×24 = 15360 gave
# identity_error 0.41 (the built-in self-test — see docs/jlens.md)
brainscope-jlens fit --model qwen3-4b --prompts wikitext \
    --out lenses/qwen3-4b-instruct-2507.jlens.pt --n-texts 640 --repeats 24
