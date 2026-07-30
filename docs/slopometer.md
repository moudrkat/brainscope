# Slop-o-meter — concept & validation note

*A linear probe on LLM activations that scores "AI slop" register per
token. Built in a day; validated with the structure of a bank's model
validation, at toy scale. Exploratory — where the numbers below look
perfect, that means the test classes were easy, not that the meter
is. Short on purpose.*

## 1. Purpose & intended use

Detect, per token and in real time, when a model's output drifts into
generic filler register ("In today's fast-paced world…"), on your own
model and traffic. Lab instrument for observability and demos — **not** a
content-moderation or truth detector. The same cascade frontier labs run in
production for misuse (probe screens everything, expensive checks on hits;
Anthropic ~1% overhead, DeepMind in user-facing Gemini) — pointed at a
harmless register.

## 2. Methodology

Logistic-regression probe on the mean layer-20 residual of the answer
(Qwen3-4B-Instruct-2507, hidden 2560). Deployed as a direction + threshold
in brainscope (`POST /probes`): per-token score h·ŵ, EMA smoothing (0.15),
signed threshold, optional trip. Trained via `POST /probes/train`-style
pipeline: paired prompts answered under two personas differing ONLY in the
register instruction (slop pos seeded with canonical clichés; neg = dense,
concrete, numbers). Diff-of-means gives near-identical directions
(cos ≈ 0.94) — the register is a linear direction, not a classifier trick.

## 3. Development data

16 paired everyday prompts × {slop, concrete} personas, English (n=26
kept after trace eviction; Czech variants trained separately). Same-model
synthetic labels — the persona defines the class. Grouped 4-fold CV (both
members of a pair stay in one fold).

## 4. Validation

**Development (held-out folds):** out-of-fold AUC 1.000; class separation
+6.6 / −6.5 (σ ≈ 1).

**Out-of-sample (20 unseen topics × natural / sloppy-request /
concrete-request + 8 false-positive probes):**

- Discrimination: AUC 1.000 (95% bootstrap CI 1.0–1.0), Gini 1.0,
  KS 1.0 — zero overlap (sloppy +1.73 ± 1.00, concrete −5.48 ± 0.72).
- **Calibration drift found:** the dev-calibrated answer-mean threshold
  (3.3) sits above the OOS sloppy distribution → 19/20 FN at that
  operating point; the OOS midpoint ≈ −1.9 gives 19–20/20 TP at 0/20
  FP. Streaming (per-token EMA, peaks +13) keeps a higher threshold —
  two operating points for two uses. Recalibrate on the target
  population, as ever.
- False-positive probes: technical bullets (−7.8) and sober summaries
  (−7.1) score lowest of everything — bullets are not slop; plain
  email −0.3. Two fired: a generic congratulations message (+6.8,
  arguably genre-true) and an excited-but-factual post (+4.6) — the
  hype/slop register correlation, a real limitation.
- Population note: natural LinkedIn-post answers score +3.14 ± 1.07,
  above even the explicit sloppy requests — the model's default
  LinkedIn register is the register.
- Human-text spot check (teacher-forced): the announcement post for
  this meter scores −9.8, below the concrete class — with the caveat
  that the probe was trained on the model's own generations, not
  human text.

**Qualitative token attribution (out-of-sample):** highlights land on
"Let's be real:", headline bait, hashtags, "Hope this helps!"-style
closers; concrete sentences inside the same answer stay unhighlighted.
See fig_slop_pair (same topic, two requests: one answer glows, one
doesn't).

**Cross-lingual check:** the Czech-trained probe separates English
answers at AUC 1.0 and vice versa, while cos(w_cz, w_en) ≈ 0.47 — the
discriminative component transfers across languages.

## 5. Limitations

One model, one layer, synthetic register elicitation (persona-requested
slop, not wild human-flagged slop), n in the tens, no adversarial
hardening, threshold calibrated on answer means (streaming EMA is
noisier). Correlated registers exist (a cheerful emoji-spam answer also
raises hype/sycophancy probes) — the probe reads register, not intent.

## 6. Monitoring & governance

Probe is model-version-specific: re-run `POST /probes/train` after any
base-model change (minutes). Holdout AUC is reported by the endpoint on
every retrain — do not deploy a meter whose AUC you have not seen.
Traces persist per-token scores for drift review.
