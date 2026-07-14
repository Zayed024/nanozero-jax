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

**Status: full pipeline built and CPU-verified (18 tests + structural smoke green).** The remaining step is the real training run on Colab (Qwen2.5-0.5B, T4) to produce the headline before/after numbers + the HF model.

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
