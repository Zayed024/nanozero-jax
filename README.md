# nanozero-jax

**Tinker-free GRPO in JAX, reproducing [rLLM](https://github.com/rllm-org/rllm)'s Countdown RL recipe on a single 16 GB card.**

The contribution is the *reproduction path*, not beating a baseline: a single-file, dependency-light JAX/Flax implementation of GRPO that RL-tunes a small model on the [Countdown](https://huggingface.co/datasets/Jiayi-Pan/Countdown-Tasks-3to4) task using rLLM's exact reward, on hardware anyone has (one Colab T4).

- **Headline:** Qwen2.5-0.5B-Instruct, LoRA GRPO, fits 16 GB via a single frozen base + adapter-off reference pass.
- **Ablation:** a from-scratch 10M TinyStories transformer ([Zayed024/tinystories-10m-jax](https://github.com/Zayed024/tinystories-10m-jax)) — runs the identical loop, establishing the capability floor.
- **Style:** one file, pure-JAX functional (params are a plain pytree), Karpathy aesthetic.

## Status

- [x] **Day 1** — Qwen2 forward + HF weight loader + logit-match gate (`< 1e-2` vs HF)
- [x] **Day 2** — fixed-buffer batched sampler + per-token logprobs (sample/re-score consistency = exact)
- [x] **Day 3** — Countdown data pipeline + vendored rLLM `compute_score` (15 reward tests green)
- [x] **Day 4** — GRPO loss + LoRA (zero-init delta=0, mean-0 group advantages, grad flows to LoRA only)
- [x] **Day 5** — rollout/train loop (`train.py`; CPU test: rollout → grpo_update → LoRA moves, base frozen)
- [x] **Day 6** — eval: greedy pass@1, LoRA-off (baseline) vs LoRA-on (trained) — before/after wired into `train.py`

## Results

**Qwen2.5-0.5B-Instruct · LoRA (rank 16, attention projections) · GRPO · one Colab T4:**

| | pass@1 (128 held-out Countdown problems, greedy) |
|---|---|
| baseline (LoRA off) | **0.00%** |
| after 200 GRPO steps (LoRA on) | **15.62%** (+15.62) |

Training dynamics matched the TinyZero recipe: the model first learns the *format*
(mean reward climbs 0.01 → 0.10 as `<answer>` tags appear), then begins actually
solving (`solved%` climbs from 0). Reward is rLLM's exact `compute_score`
(1.0 correct / 0.1 valid-format / 0.0 no answer). ~40 s/step at batch 64
(8 prompts × group 8, 256 new tokens) after the KV-cache + remat work below.

**What it took to fit one 16 GB card** (the reproduction path — this is the contribution):
- **LoRA-off reference**: π_ref = the same frozen base with adapters disabled — one
  copy of weights serves both policy and reference; zero-init B ⇒ KL starts at exactly 0.
- **Never materialize `[B, T, vocab]` logits** (at vocab 152k that single tensor is
  ~12 GB): decode computes logits only at the current position; scoring applies the LM
  head in checkpointed chunks with `B·chunk ≤ 1024`.
- **Per-layer gradient checkpointing** on the differentiated pass (otherwise 24 layers
  of attention residuals ≈ 16 GB).
- **KV-cache decoding**, bit-exact vs the no-cache reference (tested), ~20× faster steps.
- **Module-level jits** (a jit closure rebuilt per call recompiles every step and pins
  stale LoRA copies — the leak that OOM'd multi-hour runs).
- Degenerate GRPO groups (identical rewards ⇒ zero advantage) skip the backward.

## Train (Colab, one 16 GB GPU)

```bash
pip install -r requirements.txt optax
python nanozero.py        # first: confirm the HF logit-match gate passes (< 1e-2)
python train.py           # GRPO-LoRA on Countdown; prints baseline pass@1 → trained pass@1
```

`train.py` prints per-step `loss / reward / solved%` and a final `baseline → trained (Δ)` pass@1.
Sanity-check first by overfitting a handful of prompts (small `n_prompts`, more `steps`) — `solved%`
should climb. Headline = Qwen2.5-0.5B; run the 10M TinyStories model as the capability-floor ablation.

## Run the Day-1 gate

```bash
pip install -r requirements.txt
python nanozero.py        # asserts max|logits_jax - logits_hf| < 1e-2
```

Runs on CPU — no GPU needed for the gate. If JAX GPU isn't set up on your machine
(native Windows has no JAX GPU), run on a Colab CPU runtime or under WSL2.

## Reward attribution

The Countdown reward is vendored verbatim from rLLM's `rllm/rewards/countdown_reward.py`
(Apache-2.0) so the reproduction is byte-faithful and this repo stays free of rLLM's
heavy dependency tree. See the attribution comment at the vendor site.
