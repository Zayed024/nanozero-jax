"""NanoZero training: GRPO on Countdown, LoRA on a small model, single 16 GB card.

`grpo_update` is the trainable core (pure JAX + optax, CPU-testable on a tiny model).
`train` is the Colab orchestration that wires the real model + tokenizer + Countdown data:

    run each prompt G times -> reward each completion -> group-relative advantage ->
    GRPO loss vs the LoRA-off reference -> optax step on LoRA.

Usage (Colab, one 16 GB GPU):
    python train.py            # trains Qwen2.5-0.5B LoRA on Countdown
"""

from __future__ import annotations

import time

import optax

import nanozero as nz
from countdown import build_user_prompt, load_countdown, reward


# --------------------------------------------------------------------------------------
# trainable core (pure JAX + optax; no tokenizer/model download needed to test)
# --------------------------------------------------------------------------------------
def grpo_update(params, lora, opt_state, optimizer, batch, cfg, *, group_size, clip_eps=0.2, kl_beta=0.001, lora_scale=1.0):
    """One GRPO optimizer step on the LoRA params. `batch` is the rollout output:
    (full_ids, full_mask, resp_mask, old_lp, rewards). Returns (lora, opt_state, metrics)."""
    import jax

    full_ids, full_mask, resp_mask, old_lp, rewards = batch
    adv = nz.group_advantages(rewards, group_size)
    ref_lp = jax.lax.stop_gradient(nz.sequence_logprobs(params, full_ids, cfg, full_mask))  # frozen base

    def loss_fn(lora):
        # remat=True: this is the differentiated pass — without per-layer checkpointing
        # the backward residuals (24 x [B,nh,T,T]) OOM a 16 GB card.
        pol_lp = nz.sequence_logprobs(params, full_ids, cfg, full_mask, lora=lora, lora_scale=lora_scale, remat=True)
        return nz.grpo_loss(pol_lp, ref_lp, old_lp, adv, resp_mask, clip_eps=clip_eps, kl_beta=kl_beta)

    loss, grads = jax.value_and_grad(loss_fn)(lora)
    updates, opt_state = optimizer.update(grads, opt_state, lora)
    lora = optax.apply_updates(lora, updates)
    metrics = {
        "loss": float(loss),
        "reward_mean": float(rewards.mean()),
        "reward_solved": float((rewards >= 1.0).mean()),
        "adv_std": float(adv.std()),
    }
    return lora, opt_state, metrics


# --------------------------------------------------------------------------------------
# Colab orchestration (needs the real model + tokenizer)
# --------------------------------------------------------------------------------------
def _pass_at_1(completions, instances):
    """Fraction of completions that solve their countdown instance (reward == 1.0). Pure."""
    from countdown import compute_score

    solved = [compute_score(c, {"target": x["target"], "numbers": x["numbers"]}) >= 1.0 for c, x in zip(completions, instances)]
    return sum(solved) / max(len(solved), 1)


def evaluate_countdown(params, cfg, tok, instances, *, pad_id, eos_id, max_new=256, lora=None, eval_batch=16):
    """Greedy pass@1 on held-out countdown instances. lora=None -> baseline; lora set -> trained.
    Processed in mini-batches of `eval_batch` to keep T4 memory in check."""
    import jax

    completions = []
    for s in range(0, len(instances), eval_batch):
        chunk = instances[s : s + eval_batch]
        prompt_ids, prompt_mask = _encode_prompts(tok, [build_user_prompt(x["numbers"], x["target"]) for x in chunk], pad_id)
        full_ids, _, _, _ = nz.generate(params, prompt_ids, prompt_mask, cfg, max_new=max_new, key=jax.random.PRNGKey(0), eos_id=eos_id, pad_id=pad_id, temperature=0.0, lora=lora)
        completions += _decode(tok, full_ids[:, prompt_ids.shape[1]:])
    return _pass_at_1(completions, instances)


def _decode(tok, ids_2d):
    """batch_decode a JAX/numpy [B, T] id array: the fast (Rust) tokenizer only accepts
    plain Python ints, not jax arrays."""
    import numpy as np

    return tok.batch_decode(np.asarray(ids_2d).tolist(), skip_special_tokens=True)


def _chat_ids(tok, user_prompt):
    """Chat-template one user prompt -> list[int], across transformers versions
    (which variously return list[int], dict/BatchEncoding, or a batched list)."""
    out = tok.apply_chat_template([{"role": "user", "content": user_prompt}], add_generation_prompt=True, tokenize=True)
    if hasattr(out, "keys"):
        out = out["input_ids"]
    if out and isinstance(out[0], (list, tuple)):
        out = out[0]
    return list(out)


def _encode_prompts(tok, user_prompts, pad_id):
    """Chat-template each user prompt and LEFT-pad the batch. -> (ids [B,L], mask [B,L])."""
    import jax.numpy as jnp

    seqs = [_chat_ids(tok, p) for p in user_prompts]
    L = max(len(s) for s in seqs)
    ids = [[pad_id] * (L - len(s)) + list(s) for s in seqs]
    mask = [[0] * (L - len(s)) + [1] * len(s) for s in seqs]
    return jnp.asarray(ids, jnp.int32), jnp.asarray(mask, jnp.int32)


def train(*, steps=100, n_prompts=8, group_size=8, max_new=256, rank=16, lr=1e-4, kl_beta=0.001, temperature=1.0, seed=0, pool_size=2000, eval_n=32):
    import jax
    import jax.numpy as jnp
    from transformers import AutoTokenizer

    params, cfg, path = nz.load_params()
    tok = AutoTokenizer.from_pretrained(path)
    eos_id, pad_id = tok.eos_token_id, (tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id)

    lora = nz.init_lora(params, cfg, rank=rank, seed=seed)
    optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(lr))
    opt_state = optimizer.init(lora)

    pool = load_countdown(pool_size)
    held_out = load_countdown(pool_size + eval_n, split="train")[-eval_n:]  # held out AFTER the training pool
    key = jax.random.PRNGKey(seed)

    base_acc = evaluate_countdown(params, cfg, tok, held_out, pad_id=pad_id, eos_id=eos_id, max_new=max_new, lora=None)
    print(f"[eval] baseline (LoRA-off) pass@1 = {base_acc:.2%}")

    for step in range(steps):
        t0 = time.time()
        key, pk, gk = jax.random.split(key, 3)
        idx = jax.random.randint(pk, (n_prompts,), 0, len(pool))
        chosen = [pool[int(i)] for i in idx]
        rep = [c for c in chosen for _ in range(group_size)]  # G rollouts per prompt

        prompt_ids, prompt_mask = _encode_prompts(tok, [build_user_prompt(c["numbers"], c["target"]) for c in rep], pad_id)
        full_ids, full_mask, resp_mask, old_lp = nz.generate(
            params, prompt_ids, prompt_mask, cfg, max_new=max_new, key=gk, eos_id=eos_id, pad_id=pad_id, temperature=temperature, lora=lora
        )
        t_gen = time.time() - t0

        L = prompt_ids.shape[1]
        completions = _decode(tok, full_ids[:, L:])
        rewards = jnp.asarray([reward(c, r["numbers"], r["target"]) for c, r in zip(completions, rep)], jnp.float32)

        # All groups degenerate (identical reward within every group) -> every advantage
        # is 0 -> the update is a guaranteed no-op. Skip the expensive backward.
        adv = nz.group_advantages(rewards, group_size)
        if not bool(jnp.any(jnp.abs(adv) > 1e-8)):
            print(f"step {step:3d} | SKIP (all groups degenerate) | reward {float(rewards.mean()):.3f} | gen {t_gen:.0f}s")
            continue

        batch = (full_ids, full_mask, resp_mask, old_lp, rewards)
        lora, opt_state, m = grpo_update(params, lora, opt_state, optimizer, batch, cfg, group_size=group_size, kl_beta=kl_beta)
        t_all = time.time() - t0
        print(
            f"step {step:3d} | loss {m['loss']:+.4f} | reward {m['reward_mean']:.3f} | solved {m['reward_solved']:.2%} "
            f"| adv_std {m['adv_std']:.3f} | gen {t_gen:.0f}s upd {t_all - t_gen:.0f}s"
        )

    final_acc = evaluate_countdown(params, cfg, tok, held_out, pad_id=pad_id, eos_id=eos_id, max_new=max_new, lora=lora)
    print(f"[eval] baseline {base_acc:.2%} -> trained (LoRA-on) {final_acc:.2%}  (Δ {final_acc - base_acc:+.2%})")
    nz.save_lora(lora, "nanozero_countdown_lora.npz")
    print("[save] adapter -> nanozero_countdown_lora.npz")
    return lora


if __name__ == "__main__":
    train()
