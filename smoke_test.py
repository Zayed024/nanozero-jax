"""No-download structural check: build a tiny random-weight model with the SAME code
paths (GQA, RoPE, tied head, transposed linears) and assert shapes + finiteness.
Catches transpose/GQA/RoPE bugs without the 1GB Qwen download. Run: python smoke_test.py
"""
import jax
import jax.numpy as jnp
import numpy as np

from nanozero import Config, forward, generate, grpo_loss, group_advantages, init_lora, sequence_logprobs

# tiny config that still exercises every code path: GQA (4 q / 2 kv), even head_dim for rope
cfg = Config(
    vocab_size=128, hidden=32, n_layers=2, n_heads=4, n_kv_heads=2, head_dim=8,
    intermediate=64, rope_theta=1e6, eps=1e-6, tie_embeddings=True,
)

rng = np.random.default_rng(0)


def rn(*shape):
    return jnp.asarray(rng.standard_normal(shape).astype(np.float32) * 0.02)


def layer():
    nh, nkv, hd, H, I = cfg.n_heads, cfg.n_kv_heads, cfg.head_dim, cfg.hidden, cfg.intermediate
    return {
        "ln1": rn(H), "ln2": rn(H),
        "wq": rn(H, nh * hd), "bq": rn(nh * hd),
        "wk": rn(H, nkv * hd), "bk": rn(nkv * hd),
        "wv": rn(H, nkv * hd), "bv": rn(nkv * hd),
        "wo": rn(nh * hd, H),
        "w_gate": rn(H, I), "w_up": rn(H, I), "w_down": rn(I, H),
    }


params = {"embed": rn(cfg.vocab_size, cfg.hidden), "norm": rn(cfg.hidden), "layers": [layer() for _ in range(cfg.n_layers)]}

B, T = 2, 7
ids = jnp.asarray(rng.integers(0, cfg.vocab_size, size=(B, T)), dtype=jnp.int32)

logits = forward(params, ids, cfg)
assert logits.shape == (B, T, cfg.vocab_size), f"bad shape {logits.shape}"
assert jnp.all(jnp.isfinite(logits)), "non-finite logits"

# causality: logits at position t must not change when future tokens are altered
ids2 = ids.at[:, -1].set((ids[:, -1] + 1) % cfg.vocab_size)
l1 = forward(params, ids, cfg)
l2 = forward(params, ids2, cfg)
assert jnp.allclose(l1[:, :-1], l2[:, :-1], atol=1e-5), "causality violated: past logits changed"

# jit must compile + grad must flow (sanity for Day-4 GRPO)
jforward = jax.jit(lambda p, x: forward(p, x, cfg))
_ = jforward(params, ids).block_until_ready()
g = jax.grad(lambda p: forward(p, ids, cfg).sum())(params)
assert jnp.all(jnp.isfinite(g["embed"])), "non-finite grad"

print("shapes OK:", logits.shape)
print("causality OK (past logits invariant to future tokens)")
print("jit OK, grad OK")

# ---- Day 2: generation + per-token logprobs --------------------------------------------
PAD, EOS = 0, -1  # EOS=-1 never fires -> always generate full max_new
Lp, NEW = 5, 6
prompt = jnp.asarray(rng.integers(1, cfg.vocab_size, size=(B, Lp)), jnp.int32)  # avoid PAD id
pmask = jnp.ones((B, Lp), jnp.int32)
prompt = prompt.at[0, 0].set(PAD)   # left-pad one token in row 0 to exercise padding/offset
pmask = pmask.at[0, 0].set(0)

ids_g, full_mask, resp_mask, gen_logp = generate(
    params, prompt, pmask, cfg, max_new=NEW, key=jax.random.PRNGKey(0), eos_id=EOS, pad_id=PAD, temperature=1.0
)
assert ids_g.shape == (B, Lp + NEW), f"bad gen shape {ids_g.shape}"
assert int(resp_mask.sum()) == B * NEW, "response mask count wrong"

# THE Day-2 gate: re-score the sampled sequence under the same params; response-token
# logprobs must equal what generation recorded. Mismatch => mask/shift misalignment.
recomp = sequence_logprobs(params, ids_g, cfg, full_mask)
diff = float(jnp.max(jnp.abs((recomp - gen_logp) * resp_mask)))
assert diff < 1e-4, f"logprob consistency FAILED: max|diff|={diff:.2e}"
print(f"gen shape {ids_g.shape}, logprob consistency max|diff| = {diff:.2e}")

# greedy must be deterministic regardless of key
ga = generate(params, prompt, pmask, cfg, max_new=4, key=jax.random.PRNGKey(0), eos_id=EOS, pad_id=PAD, temperature=0.0)[0]
gb = generate(params, prompt, pmask, cfg, max_new=4, key=jax.random.PRNGKey(7), eos_id=EOS, pad_id=PAD, temperature=0.0)[0]
assert jnp.array_equal(ga, gb), "greedy decode not deterministic"
print("greedy deterministic across seeds: OK")

# ---- Day 4: LoRA + GRPO ------------------------------------------------------------------
# zero-init LoRA (B=0) must leave logits identical -> policy == reference at step 0
lora0 = init_lora(params, cfg, rank=4, seed=1)
logits_base = forward(params, ids, cfg)
logits_lora0 = forward(params, ids, cfg, lora=lora0)
assert jnp.max(jnp.abs(logits_base - logits_lora0)) < 1e-6, "zero-init LoRA changed logits"

# group advantages: z-score within each group -> mean 0 per group
r = jnp.asarray([0.0, 1.0, 0.0, 1.0])  # 2 groups of 2
adv = group_advantages(r, group_size=2)
assert adv.shape == (4,)
assert jnp.allclose(adv.reshape(2, 2).mean(axis=1), 0.0, atol=1e-5)

# grpo_loss is finite, and gradient flows ONLY into LoRA (B receives it; base is frozen)
attn_all = jnp.ones_like(ids)
resp = jnp.ones_like(ids, dtype=jnp.float32).at[:, :3].set(0.0)  # response = tokens after the prompt
old_lp = sequence_logprobs(params, ids, cfg, attn_all, lora=lora0)
ref_lp = sequence_logprobs(params, ids, cfg, attn_all)  # base, lora-off reference
adv_b = jnp.asarray([1.0, -1.0])


def _loss(lora):
    pol_lp = sequence_logprobs(params, ids, cfg, attn_all, lora=lora)
    return grpo_loss(pol_lp, ref_lp, old_lp, adv_b, resp)


loss_val, grads = jax.value_and_grad(_loss)(lora0)
assert jnp.isfinite(loss_val)
gB = grads["layers"][0]["wq"]["B"]
gA = grads["layers"][0]["wq"]["A"]
assert jnp.all(jnp.isfinite(gB)) and jnp.all(jnp.isfinite(gA))
assert jnp.any(gB != 0.0), "LoRA B should receive gradient"
print(f"LoRA/GRPO OK: zero-init delta=0, loss={float(loss_val):.4f}, grad flows to LoRA")

print("\n  [PASS] structural smoke test - forward + sampler + logprobs + GRPO/LoRA are sound.")
print("  (Run `python nanozero.py` on Colab for the real HF logit-match gate.)")
