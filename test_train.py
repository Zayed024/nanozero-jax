"""CPU test of the GRPO training core on a tiny random model (no download).

Verifies the full wiring: on-policy rollout (generate with LoRA) -> grpo_update ->
LoRA params actually move, loss is finite, and the reference pass is LoRA-off.
"""

import jax
import jax.numpy as jnp
import numpy as np
import optax

import nanozero as nz
from train import _pass_at_1, grpo_update


def _tiny():
    cfg = nz.Config(vocab_size=64, hidden=32, n_layers=2, n_heads=4, n_kv_heads=2, head_dim=8, intermediate=64, rope_theta=1e6, eps=1e-6, tie_embeddings=True)
    rng = np.random.default_rng(0)

    def rn(*s):
        return jnp.asarray(rng.standard_normal(s).astype(np.float32) * 0.02)

    def layer():
        nh, nkv, hd, H, I = cfg.n_heads, cfg.n_kv_heads, cfg.head_dim, cfg.hidden, cfg.intermediate
        return {"ln1": rn(H), "ln2": rn(H), "wq": rn(H, nh * hd), "bq": rn(nh * hd), "wk": rn(H, nkv * hd), "bk": rn(nkv * hd),
                "wv": rn(H, nkv * hd), "bv": rn(nkv * hd), "wo": rn(nh * hd, H), "w_gate": rn(H, I), "w_up": rn(H, I), "w_down": rn(I, H)}

    params = {"embed": rn(cfg.vocab_size, cfg.hidden), "norm": rn(cfg.hidden), "layers": [layer() for _ in range(cfg.n_layers)]}
    return params, cfg


def test_grpo_update_moves_lora():
    params, cfg = _tiny()
    lora = nz.init_lora(params, cfg, rank=4, seed=1)

    B, Lp, G = 4, 3, 2
    rng = np.random.default_rng(1)
    prompt = jnp.asarray(rng.integers(1, cfg.vocab_size, size=(B, Lp)), jnp.int32)
    pmask = jnp.ones((B, Lp), jnp.int32)

    # on-policy rollout with LoRA
    full_ids, full_mask, resp_mask, old_lp = nz.generate(
        params, prompt, pmask, cfg, max_new=4, key=jax.random.PRNGKey(0), eos_id=-1, pad_id=0, temperature=1.0, lora=lora
    )
    rewards = jnp.asarray([1.0, 0.0, 0.0, 1.0], jnp.float32)  # 2 groups of 2, one solved each

    optimizer = optax.adamw(1e-2)
    opt_state = optimizer.init(lora)
    batch = (full_ids, full_mask, resp_mask, old_lp, rewards)
    new_lora, _, m = grpo_update(params, lora, opt_state, optimizer, batch, cfg, group_size=G)

    assert jnp.isfinite(m["loss"])
    moved = float(jnp.max(jnp.abs(new_lora["layers"][0]["wq"]["B"] - lora["layers"][0]["wq"]["B"])))
    assert moved > 0.0, "LoRA did not update"


def test_pass_at_1():
    instances = [
        {"numbers": [4, 6], "target": 24},  # solved by "4 * 6"
        {"numbers": [4, 6], "target": 24},  # wrong answer below
        {"numbers": [1, 2, 3], "target": 6},  # no answer tag below
    ]
    completions = ["<answer>4 * 6</answer>", "<answer>4 + 6</answer>", "i give up"]
    assert _pass_at_1(completions, instances) == 1 / 3


def test_reference_is_lora_off():
    params, cfg = _tiny()
    lora0 = nz.init_lora(params, cfg, rank=4, seed=2)
    ids = jnp.asarray([[1, 2, 3, 4, 5]], jnp.int32)
    mask = jnp.ones_like(ids)
    lp_base = nz.sequence_logprobs(params, ids, cfg, mask)
    lp_lora0 = nz.sequence_logprobs(params, ids, cfg, mask, lora=lora0)
    assert float(jnp.max(jnp.abs(lp_base - lp_lora0))) < 1e-6


def test_lora_save_load_roundtrip(tmp_path):
    params, cfg = _tiny()
    # give the adapter non-zero B so the roundtrip is meaningful
    lora = nz.init_lora(params, cfg, rank=4, seed=3)
    rng = np.random.default_rng(9)
    for layer in lora["layers"]:
        for ab in layer.values():
            ab["B"] = jnp.asarray(rng.standard_normal(ab["B"].shape).astype(np.float32))
    path = str(tmp_path / "adapter.npz")
    nz.save_lora(lora, path)
    loaded = nz.load_lora(path)

    ids = jnp.asarray([[1, 2, 3, 4, 5]], jnp.int32)
    mask = jnp.ones_like(ids)
    lp_before = nz.sequence_logprobs(params, ids, cfg, mask, lora=lora)
    lp_after = nz.sequence_logprobs(params, ids, cfg, mask, lora=loaded)
    assert float(jnp.max(jnp.abs(lp_before - lp_after))) < 1e-6
