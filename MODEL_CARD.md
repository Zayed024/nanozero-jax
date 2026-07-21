---
license: apache-2.0
base_model: Qwen/Qwen2.5-0.5B-Instruct
tags:
- reinforcement-learning
- grpo
- lora
- jax
- countdown
- reasoning
language:
- en
pipeline_tag: text-generation
---

# NanoZero Countdown LoRA — GRPO from scratch in JAX, on one free T4

A LoRA adapter for [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct), RL-trained with **GRPO implemented from scratch in pure JAX** on the [Countdown task](https://huggingface.co/datasets/Jiayi-Pan/Countdown-Tasks-3to4) — a tinker-free reproduction of [rLLM](https://github.com/rllm-org/rllm)'s Countdown RL recipe that fits on a single free Colab T4 (16 GB).

**Code:** [github.com/Zayed024/nanozero-jax](https://github.com/Zayed024/nanozero-jax) — single-file model + GRPO, no training framework.

## Results

| | pass@1 (128 held-out Countdown problems, greedy) |
|---|---|
| Qwen2.5-0.5B-Instruct (baseline) | **0.00%** |
| + this adapter (200 GRPO steps) | **15.62%** |

The task: given numbers like `[3, 7, 11]` and a target like `28`, produce an arithmetic
equation using each number exactly once, in `<answer>...</answer>` tags. Reward is rLLM's
exact `compute_score` (1.0 correct / 0.1 valid-format / 0.0 no answer), vendored verbatim.
Training showed the TinyZero-style two-phase dynamic: format acquisition first (mean reward
0.01 → 0.10), then actual solving (solved% climbing from 0).

## Training setup

- **Algorithm:** GRPO — group-relative advantages (8 rollouts/prompt, z-scored), PPO-style
  clipped policy gradient, k3 KL penalty to the frozen reference (β=0.001).
- **Adapter:** LoRA rank 16 on attention projections (q/k/v/o). The **reference model is
  the same frozen base with adapters off** — one weight copy serves policy and reference;
  B is zero-init so the policy starts exactly at the reference.
- **Budget:** 200 steps × (8 prompts × group 8) × 256 new tokens, temperature 1.0,
  AdamW lr 1e-4, global-norm clip 1.0. ~40 s/step on a T4 after KV-caching.
- **Fitting 16 GB** (the point of the exercise): never materialize the full `[B, T, vocab]`
  logits tensor (chunked LM head, ≈12 GB avoided), per-layer gradient checkpointing on the
  differentiated pass, bit-exact KV-cache decoding, degenerate-group skip.
- **Forward-pass fidelity:** the from-scratch JAX Qwen2 matches HF logits to max
  |diff| = 2.6e-4 (argmax agreement 1.000) on the same weights.

## Usage

The adapter is a plain `.npz` of LoRA A/B matrices keyed `"{layer}__{proj}__{A|B}"`
(projections `wq/wk/wv/wo`, applied as `W·x + (x·A)·B`). With the NanoZero code:

```python
import nanozero as nz
from huggingface_hub import hf_hub_download

params, cfg, path = nz.load_params("Qwen/Qwen2.5-0.5B-Instruct")
lora = nz.load_lora(hf_hub_download("Zayed024/nanozero-countdown-lora",
                                    "nanozero_countdown_lora.npz"))
# generate with the adapter:
ids, mask, resp, lp = nz.generate(params, prompt_ids, prompt_mask, cfg,
                                  max_new=256, key=key, eos_id=eos, pad_id=pad,
                                  temperature=0.0, lora=lora)
```

(It is **not** a PEFT-format adapter; it pairs with the NanoZero codebase. Conversion to
PEFT is straightforward from the key layout above if you need it.)

## Intended use & limitations

Research/educational artifact demonstrating a minimal, verified GRPO pipeline. Trained
only on Countdown arithmetic — it improves equation-writing under the `<answer>` format
and nothing else; expect no gains (and possible format quirks) outside that task. Base
model license and usage terms apply.

## Acknowledgements

- Reward and recipe: [rLLM](https://github.com/rllm-org/rllm) (Berkeley Sky Computing Lab), `countdown_reward.py` vendored under Apache-2.0.
- Task data: [Jiayi-Pan/Countdown-Tasks-3to4](https://huggingface.co/datasets/Jiayi-Pan/Countdown-Tasks-3to4) (TinyZero).
- Base model: Qwen2.5-0.5B-Instruct.
