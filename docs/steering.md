# Steering: field notes & lessons learned

What we learned shipping a real steering vector with brainscope — extracting a
behaviour direction from a working agent application and using it to switch
off one concrete misbehaviour of the model.

## The case study

An assistant had one conversational behaviour it was never supposed to show in
a particular mode — violated often enough that a dedicated validator LLM ran
after every response just to catch it. Goal: replace that check with an
activation-steering vector.

- **Data**: contrast pairs `{prompt, positive, negative}` — same prompt, a
  compliant completion vs. one showing the unwanted behaviour. Two datasets:
  one built from real traffic, one fully synthetic (templated topics, the
  exact forbidden phrasings injected).
- **Extraction**: `brainscope.hidden_directions` — per-layer PCA over
  completion-token hidden-state diffs + a per-layer separation score.
- **Eval**: held-out real prompts, generated baseline vs. steered at several
  strengths; a deterministic violation checker mirroring the validator's rules
  plus coherence guards (repetition, language intact). The eval's baseline
  violation rate matched what the validator saw in practice — which is how you
  know the eval measures the real thing.

Result: with the synthetic vector applied to a narrow band of mid-depth
layers, violations on held-out prompts dropped by roughly an order of
magnitude — and the rare survivor was also the rare incoherent output. Among
coherent outputs: zero violations.

## The lessons

1. **Clean templated contrast beats realistic-but-noisy pairs — by a lot.**
   The synthetic direction scored ~10× higher layer separation and steered
   cleanly. The real-traffic direction was toxic at every dose: below the
   effective strength it did nothing, above it the model collapsed. Real
   diffs carry format, length and topic noise, so the principal component
   isn't purely the behaviour. If you must use real data, pair same-length
   completions on the same prompt and topic.

2. **Steer a narrow window near the score peak.** The per-layer table showed a
   broad plateau; steering all plateau layers at once meant a huge total
   injection — the model degraded before the behaviour fully disappeared.
   A few layers around the peak tolerated ~3× the strength with fewer side
   effects. Rule of thumb: total injection ≈ strength × number of layers.

3. **Low layers amplify.** The real-data vector peaked in the first quarter of
   the stack and destroyed the model at any effective strength — an early
   perturbation cascades through everything downstream. Mid-depth (40–70 %)
   is the safe playground, exactly where the literature puts behaviour
   directions.

4. **Over-steering has a signature.** Repetition loops first (a short phrase
   echoing over and over), then language switching, then token salad. Sweep
   strengths and score *coherence* alongside your target metric — a checker
   that only counts violations will happily report a broken model as perfect.

5. **Evaluating the agent ≠ evaluating the deployment.** `/steer` is global
   server state: every request through the server generates under the vector.
   Our eval covered the target agent only; in the full pipeline a *different*
   agent (different prompt, same server) collapsed into repetition on first
   contact. Scope steering to the calls that want it — per-request steering in
   the chat payload, a tag-matched steering policy, or route only the target
   agent through brainscope.

6. **Keep a boring baseline.** Our violation checker reproduced the deployed
   validator's fail rate before any steering. If your eval's baseline doesn't
   match reality, fix the eval before believing anything else it says.

## Recipe

```bash
# 1. build pairs.jsonl - templated, one behaviour, same prompt both sides
# 2. extract + read the layer table
python -m brainscope.hidden_directions --model qwen3-4b \
    --pairs pairs.jsonl --name my-direction --out dirs.json
# 3. serve with the vector loaded
brainscope --model qwen3-4b --directions dirs.json
# 4. sweep: narrow layer window around the best layer, several strengths,
#    check BOTH your target metric and coherence on held-out prompts
curl -X POST localhost:8010/steer \
    -d '{"name": "my-direction", "strength": 6, "layer_from": 20, "layer_to": 22}'
```
