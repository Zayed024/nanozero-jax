"""
nanozero — tinker-free GRPO in JAX, reproducing rLLM's Countdown RL recipe on a single 16GB card.

Single file. Pure-JAX functional style (params are a plain pytree, forward is a pure
function) — no framework ceremony, so LoRA and the LoRA-off reference pass are a few
lines later, and autodiff/optax "just work".

Build order (this file grows over the week):
  [Day 1] model: Qwen2 forward + HF weight loader + logit-match gate   <-- THIS FILE
  [Day 2] sampler: fixed-length batched generation + per-token logprobs
  [Day 3] data + reward: Countdown dataset + vendored rLLM compute_score
  [Day 4] grpo: group advantages + clipped PG + k3 KL, on LoRA params
  [Day 5] train loop
  [Day 6] eval (pass@1, LoRA-off vs LoRA-on)

Day-1 acceptance gate (run `python nanozero.py`):
    max |logits_jax - logits_hf| < 1e-2   on Qwen2.5-0.5B-Instruct.
Runs on CPU (no GPU needed for the gate). Use a Colab CPU runtime or WSL2 if JAX
GPU isn't set up on your box.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from glob import glob

import jax
import jax.numpy as jnp
import numpy as np

REPO = "Qwen/Qwen2.5-0.5B-Instruct"


# --------------------------------------------------------------------------------------
# config
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Config:
    vocab_size: int
    hidden: int
    n_layers: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    intermediate: int
    rope_theta: float
    eps: float
    tie_embeddings: bool

    @staticmethod
    def from_hf(cfg: dict) -> "Config":
        hidden = cfg["hidden_size"]
        n_heads = cfg["num_attention_heads"]
        return Config(
            vocab_size=cfg["vocab_size"],
            hidden=hidden,
            n_layers=cfg["num_hidden_layers"],
            n_heads=n_heads,
            n_kv_heads=cfg["num_key_value_heads"],
            head_dim=cfg.get("head_dim", hidden // n_heads),
            intermediate=cfg["intermediate_size"],
            rope_theta=float(cfg.get("rope_theta", 1e6)),
            eps=float(cfg.get("rms_norm_eps", 1e-6)),
            tie_embeddings=bool(cfg.get("tie_word_embeddings", True)),
        )


# --------------------------------------------------------------------------------------
# rope (NeoX / "rotate_half" convention, exactly as HF Qwen2 does it)
# --------------------------------------------------------------------------------------
def rope_cos_sin(position_ids, head_dim: int, theta: float):
    """position_ids: [B, T] (per-token, supports padding/offsets) -> cos, sin each [B, T, head_dim]."""
    inv_freq = 1.0 / (theta ** (jnp.arange(0, head_dim, 2, dtype=jnp.float32) / head_dim))  # [hd/2]
    freqs = position_ids.astype(jnp.float32)[..., None] * inv_freq  # [B, T, hd/2]
    emb = jnp.concatenate([freqs, freqs], axis=-1)  # [B, T, hd]
    return jnp.cos(emb), jnp.sin(emb)


def rotate_half(x):
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return jnp.concatenate([-x2, x1], axis=-1)


def apply_rope(x, cos, sin):
    # x: [B, T, n_heads, head_dim];  cos/sin: [B, T, head_dim]
    cos = cos[:, :, None, :]
    sin = sin[:, :, None, :]
    return x * cos + rotate_half(x) * sin


# --------------------------------------------------------------------------------------
# layers (pure functions over the param pytree)
# --------------------------------------------------------------------------------------
def rmsnorm(x, weight, eps: float):
    x32 = x.astype(jnp.float32)
    var = jnp.mean(x32 * x32, axis=-1, keepdims=True)
    normed = x32 * jax.lax.rsqrt(var + eps)
    return (normed * weight.astype(jnp.float32)).astype(x.dtype)


def _lora_delta(x, lora_layer, key, scale):
    """LoRA low-rank delta for one projection: (x @ A) @ B * scale. 0 if not adapted."""
    ab = lora_layer.get(key) if lora_layer else None
    if ab is None:
        return 0.0
    return ((x @ ab["A"]) @ ab["B"]) * scale


def _attn_qkv(x, p, cfg: Config, lora_layer, lora_scale):
    """Project x -> (q [B,T,nh,hd], k [B,T,nkv,hd], v [B,T,nkv,hd]), pre-RoPE."""
    B, T, _ = x.shape
    nh, nkv, hd = cfg.n_heads, cfg.n_kv_heads, cfg.head_dim
    q = (x @ p["wq"] + p["bq"] + _lora_delta(x, lora_layer, "wq", lora_scale)).reshape(B, T, nh, hd)
    k = (x @ p["wk"] + p["bk"] + _lora_delta(x, lora_layer, "wk", lora_scale)).reshape(B, T, nkv, hd)
    v = (x @ p["wv"] + p["bv"] + _lora_delta(x, lora_layer, "wv", lora_scale)).reshape(B, T, nkv, hd)
    return q, k, v


def attention(x, p, cos, sin, cfg: Config, attn_bias, lora_layer=None, lora_scale=1.0):
    # attn_bias: [B, 1, T, T] additive (0 where allowed, -1e30 where masked)
    B, T, _ = x.shape
    nh, nkv, hd = cfg.n_heads, cfg.n_kv_heads, cfg.head_dim

    q, k, v = _attn_qkv(x, p, cfg, lora_layer, lora_scale)

    q = apply_rope(q, cos, sin)
    k = apply_rope(k, cos, sin)

    # GQA: expand kv heads to match query heads
    k = jnp.repeat(k, nh // nkv, axis=2)
    v = jnp.repeat(v, nh // nkv, axis=2)

    q = q.transpose(0, 2, 1, 3)  # [B, nh, T, hd]
    k = k.transpose(0, 2, 1, 3)
    v = v.transpose(0, 2, 1, 3)

    scores = (q @ k.transpose(0, 1, 3, 2)) / math.sqrt(hd)  # [B, nh, T, T]
    scores = scores.astype(jnp.float32) + attn_bias
    attn = jax.nn.softmax(scores, axis=-1).astype(x.dtype)

    out = attn @ v  # [B, nh, T, hd]
    out = out.transpose(0, 2, 1, 3).reshape(B, T, nh * hd)
    return out @ p["wo"] + _lora_delta(out, lora_layer, "wo", lora_scale)


def mlp(x, p):
    return (jax.nn.silu(x @ p["w_gate"]) * (x @ p["w_up"])) @ p["w_down"]


def backbone(params, ids, cfg: Config, attn_mask=None, position_ids=None, lora=None, lora_scale=1.0, remat=False):
    """ids: [B, T] int32 -> final hidden states [B, T, hidden] (pre-LM-head).

    attn_mask:    [B, T] 1 for real tokens, 0 for padding (default: all real).
    position_ids: [B, T] (default: arange, i.e. no left-padding).
    lora:         optional LoRA pytree (see `init_lora`). None -> frozen base pass
                  (this IS the GRPO reference pass — no separate model copy needed).
    remat:        gradient-checkpoint each layer. REQUIRED when differentiating at
                  training batch sizes: without it, backward stores every layer's
                  attention residuals ([B,nh,T,T] x 24 layers ~ 16 GB) and OOMs a T4.
                  Forward-only callers (generate/eval) leave it off.
    """
    B, T = ids.shape
    if position_ids is None:
        position_ids = jnp.broadcast_to(jnp.arange(T), (B, T))
    cos, sin = rope_cos_sin(position_ids, cfg.head_dim, cfg.rope_theta)

    # additive attention bias: causal AND (key is not padding)
    causal = jnp.tril(jnp.ones((T, T), dtype=bool))  # [T, T]
    bias = jnp.where(causal, 0.0, -1e30)[None, None]  # [1, 1, T, T]
    if attn_mask is not None:
        key_ok = attn_mask[:, None, None, :].astype(bool)  # [B, 1, 1, T]
        bias = jnp.where(key_ok, bias, -1e30)

    def _layer(x, p, ll):
        x = x + attention(rmsnorm(x, p["ln1"], cfg.eps), p, cos, sin, cfg, bias, ll, lora_scale)
        return x + mlp(rmsnorm(x, p["ln2"], cfg.eps), p)

    layer_fn = jax.checkpoint(_layer) if remat else _layer

    lora_layers = lora["layers"] if lora else None
    x = params["embed"][ids]  # [B, T, hidden]
    for i, p in enumerate(params["layers"]):
        x = layer_fn(x, p, lora_layers[i] if lora_layers else None)
    return rmsnorm(x, params["norm"], cfg.eps)


def _lm_head(params, cfg: Config):
    return params["embed"].T if cfg.tie_embeddings else params["lm_head"]


def forward(params, ids, cfg: Config, attn_mask=None, position_ids=None, lora=None, lora_scale=1.0):
    """Full logits [B, T, vocab]. NB: materializes B*T*vocab floats — fine for short
    sequences/small batches (the logit gate), but use `generate`/`sequence_logprobs`
    for training-sized batches (they avoid the full logits tensor)."""
    return backbone(params, ids, cfg, attn_mask, position_ids, lora, lora_scale) @ _lm_head(params, cfg)


# --------------------------------------------------------------------------------------
# weight loading (HF safetensors -> our pytree; linear weights transposed to [in, out])
# --------------------------------------------------------------------------------------
def load_params(repo: str = REPO):
    from huggingface_hub import snapshot_download
    from safetensors import safe_open

    path = snapshot_download(repo, allow_patterns=["*.json", "*.safetensors", "tokenizer*", "vocab*", "merges*"])
    cfg = Config.from_hf(json.load(open(f"{path}/config.json")))

    raw = {}
    for f in sorted(glob(f"{path}/*.safetensors")):
        with safe_open(f, framework="flax") as sf:  # returns jnp arrays, preserves bf16
            for key in sf.keys():
                raw[key] = sf.get_tensor(key).astype(jnp.float32)

    def g(name):
        return raw[name]

    params = {
        "embed": g("model.embed_tokens.weight"),  # [vocab, hidden]; lm_head is its transpose
        "norm": g("model.norm.weight"),
        "layers": [],
    }
    if not cfg.tie_embeddings:
        params["lm_head"] = g("lm_head.weight").T

    for i in range(cfg.n_layers):
        pre = f"model.layers.{i}."
        params["layers"].append({
            "ln1": g(pre + "input_layernorm.weight"),
            "wq": g(pre + "self_attn.q_proj.weight").T, "bq": g(pre + "self_attn.q_proj.bias"),
            "wk": g(pre + "self_attn.k_proj.weight").T, "bk": g(pre + "self_attn.k_proj.bias"),
            "wv": g(pre + "self_attn.v_proj.weight").T, "bv": g(pre + "self_attn.v_proj.bias"),
            "wo": g(pre + "self_attn.o_proj.weight").T,  # o_proj has no bias in Qwen2
            "ln2": g(pre + "post_attention_layernorm.weight"),
            "w_gate": g(pre + "mlp.gate_proj.weight").T,
            "w_up": g(pre + "mlp.up_proj.weight").T,
            "w_down": g(pre + "mlp.down_proj.weight").T,
        })
    return params, cfg, path


# --------------------------------------------------------------------------------------
# [Day 2] sampling + per-token logprobs
#
# Indexing convention (the thing that causes silent GRPO bugs): a per-token logprob is
# attributed to the POSITION OF THE TOKEN. out[:, p] = log p(token at p | tokens < p),
# with out[:, 0] = 0. `generate` records logprobs the same way, so a sequence sampled by
# the policy and then re-scored by `sequence_logprobs` under the same params must match
# exactly on the response tokens. That equality is the Day-2 correctness gate.
# --------------------------------------------------------------------------------------
def _positions(attn_mask):
    """position_ids from a (possibly left-padded) mask: 0 at the first real token, cumulative after."""
    pos = jnp.cumsum(attn_mask, axis=-1) - 1
    return jnp.clip(pos, 0, None).astype(jnp.int32)


def sequence_logprobs(params, ids, cfg: Config, attn_mask, lora=None, lora_scale=1.0, chunk: int = 64, remat=False):
    """Per-token logprobs under `params` (+ optional LoRA), position-aligned.
    This is the GRPO policy pass (lora set) and reference pass (lora=None).
    ids/attn_mask: [B, T] -> [B, T].

    The LM head is applied in `chunk`-sized slices over T (with rematerialization),
    so the full [B, T, vocab] logits tensor never exists — at vocab 152k that tensor
    is what OOMs a 16 GB card, not the model itself.
    """
    h = backbone(params, ids, cfg, attn_mask, _positions(attn_mask), lora, lora_scale, remat=remat)  # [B, T, H]
    head = _lm_head(params, cfg)
    B, T, _ = h.shape
    h_shift = h[:, :-1]  # [B, T-1, H]: hidden at p predicts the token at p+1
    tok = ids[:, 1:]  # [B, T-1]

    @jax.checkpoint
    def _chunk_lp(h_slice, tok_slice):
        logits = (h_slice @ head).astype(jnp.float32)  # [B, c, V] — only one chunk alive at a time
        logp = jax.nn.log_softmax(logits, axis=-1)
        return jnp.take_along_axis(logp, tok_slice[..., None], axis=-1)[..., 0]  # [B, c]

    parts = [_chunk_lp(h_shift[:, s : s + chunk], tok[:, s : s + chunk]) for s in range(0, T - 1, chunk)]
    lp = jnp.concatenate(parts, axis=1)  # [B, T-1]
    return jnp.pad(lp, ((0, 0), (1, 0)))  # [B, T]; shift so index == token position


# --------------------------------------------------------------------------------------
# KV-cache decoding: prefill the prompt once, then each new token attends against the
# cached K/V instead of recomputing the whole prefix — O(T) decode instead of O(T^2).
# --------------------------------------------------------------------------------------
def prefill(params, ids, mask, cfg: Config, *, cache_len: int, lora=None, lora_scale=1.0):
    """Run the prompt through the backbone, caching post-RoPE K/V per layer in
    zero-padded buffers of length `cache_len`. Returns (last_hidden [B,H], cache)."""
    B, L = ids.shape
    position_ids = _positions(mask)
    cos, sin = rope_cos_sin(position_ids, cfg.head_dim, cfg.rope_theta)
    causal = jnp.tril(jnp.ones((L, L), dtype=bool))
    bias = jnp.where(causal, 0.0, -1e30)[None, None]
    bias = jnp.where(mask[:, None, None, :].astype(bool), bias, -1e30)

    nh, nkv, hd = cfg.n_heads, cfg.n_kv_heads, cfg.head_dim
    lora_layers = lora["layers"] if lora else None
    x = params["embed"][ids]
    cache = []
    for i, p in enumerate(params["layers"]):
        ll = lora_layers[i] if lora_layers else None
        h = rmsnorm(x, p["ln1"], cfg.eps)
        q, k, v = _attn_qkv(h, p, cfg, ll, lora_scale)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        # cache the roped K and V, padded out to the full generation buffer
        k_buf = jnp.zeros((B, cache_len, nkv, hd), k.dtype).at[:, :L].set(k)
        v_buf = jnp.zeros((B, cache_len, nkv, hd), v.dtype).at[:, :L].set(v)
        cache.append({"k": k_buf, "v": v_buf})

        kx = jnp.repeat(k, nh // nkv, axis=2).transpose(0, 2, 1, 3)
        vx = jnp.repeat(v, nh // nkv, axis=2).transpose(0, 2, 1, 3)
        qx = q.transpose(0, 2, 1, 3)
        scores = (qx @ kx.transpose(0, 1, 3, 2)) / math.sqrt(hd)
        attn = jax.nn.softmax(scores.astype(jnp.float32) + bias, axis=-1).astype(x.dtype)
        out = (attn @ vx).transpose(0, 2, 1, 3).reshape(B, L, nh * hd)
        x = x + (out @ p["wo"] + _lora_delta(out, ll, "wo", lora_scale))
        x = x + mlp(rmsnorm(x, p["ln2"], cfg.eps), p)
    x = rmsnorm(x, params["norm"], cfg.eps)
    return x[:, -1], cache  # last position's hidden state -> first next-token logits


def _decode_step(params, cache, tok, cur, pos_row, key_mask, cfg: Config, lora, lora_scale):
    """One incremental decode step. tok [B] (the token just placed at buffer column
    `cur`), pos_row [B] (its RoPE position per row), key_mask [B, cache_len] (1 for
    every valid key including `cur`). Returns (next-token logits [B,V], updated cache)."""
    nh, nkv, hd = cfg.n_heads, cfg.n_kv_heads, cfg.head_dim
    B = tok.shape[0]
    cos, sin = rope_cos_sin(pos_row[:, None], cfg.head_dim, cfg.rope_theta)  # [B,1,hd]
    lora_layers = lora["layers"] if lora else None

    x = params["embed"][tok][:, None, :]  # [B,1,H]
    new_cache = []
    for i, p in enumerate(params["layers"]):
        ll = lora_layers[i] if lora_layers else None
        h = rmsnorm(x, p["ln1"], cfg.eps)
        q, k, v = _attn_qkv(h, p, cfg, ll, lora_scale)  # [B,1,...]
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        k_buf = jax.lax.dynamic_update_slice(cache[i]["k"], k, (0, cur, 0, 0))
        v_buf = jax.lax.dynamic_update_slice(cache[i]["v"], v, (0, cur, 0, 0))
        new_cache.append({"k": k_buf, "v": v_buf})

        kx = jnp.repeat(k_buf, nh // nkv, axis=2)  # [B,C,nh,hd]
        vx = jnp.repeat(v_buf, nh // nkv, axis=2)
        # q [B,1,nh,hd] vs all cached keys: scores [B,nh,C]
        scores = jnp.einsum("bxhd,bchd->bhc", q, kx) / math.sqrt(hd)
        scores = jnp.where(key_mask[:, None, :].astype(bool), scores.astype(jnp.float32), -1e30)
        attn = jax.nn.softmax(scores, axis=-1).astype(x.dtype)
        out = jnp.einsum("bhc,bchd->bhd", attn, vx).reshape(B, 1, nh * hd)
        x = x + (out @ p["wo"] + _lora_delta(out, ll, "wo", lora_scale))
        x = x + mlp(rmsnorm(x, p["ln2"], cfg.eps), p)
    x = rmsnorm(x, params["norm"], cfg.eps)[:, 0]  # [B,H]
    return x @ _lm_head(params, cfg), new_cache


def generate(params, prompt_ids, prompt_mask, cfg: Config, *, max_new, key, eos_id, pad_id, temperature=1.0, lora=None, lora_scale=1.0):
    """KV-cached batched sampling for left-padded prompts: prefill once, then O(1)
    work per generated token. Returns (all [B, L+max_new]):
        ids        full sequence (prompt + generation; pad after EOS)
        full_mask  1 on every real token (prompt + active generation)
        resp_mask  1 only on generated (response) token positions
        gen_logp   policy logprob of each generated token, position-aligned
    """
    B, L = prompt_ids.shape
    Lmax = L + max_new
    ids = jnp.concatenate([prompt_ids, jnp.full((B, max_new), pad_id, prompt_ids.dtype)], axis=1)
    mask = jnp.concatenate([prompt_mask, jnp.zeros((B, max_new), prompt_mask.dtype)], axis=1)
    gen_logp = jnp.zeros((B, Lmax), jnp.float32)
    finished = jnp.zeros(B, dtype=bool)
    keys = jax.random.split(key, max_new)
    arangeB = jnp.arange(B)
    plen = prompt_mask.sum(axis=1).astype(jnp.int32)  # real prompt length per row

    pre = jax.jit(lambda pr, i, m: prefill(pr, i, m, cfg, cache_len=Lmax, lora=lora, lora_scale=lora_scale))
    dec = jax.jit(
        lambda pr, c, t, cu, po, km: _decode_step(pr, c, t, cu, po, km, cfg, lora, lora_scale),
        donate_argnums=(1,),  # reuse the cache buffers in place
    )

    h_last, cache = pre(params, prompt_ids, prompt_mask)
    logits = (h_last @ _lm_head(params, cfg)).astype(jnp.float32)  # predicts the token at column L

    for s in range(max_new):
        cur = L + s
        raw = logits
        if temperature > 0:
            scaled = raw / temperature
            nxt = jax.random.categorical(keys[s], scaled, axis=-1)
        else:
            scaled = raw
            nxt = jnp.argmax(raw, axis=-1)
        lp = jax.nn.log_softmax(scaled, axis=-1)[arangeB, nxt]
        active = ~finished
        nxt = jnp.where(finished, pad_id, nxt)
        ids = ids.at[:, cur].set(nxt)
        mask = mask.at[:, cur].set(active.astype(mask.dtype))
        gen_logp = gen_logp.at[:, cur].set(lp * active)
        finished = finished | (nxt == eos_id)
        if bool(jnp.all(finished)) or s == max_new - 1:
            break
        # feed the just-placed token through one incremental step -> logits for cur+1
        logits, cache = dec(params, cache, nxt, jnp.int32(cur), plen + s, mask)

    resp_mask = mask.at[:, :L].set(0)
    return ids, mask, resp_mask, gen_logp


def generate_nocache(params, prompt_ids, prompt_mask, cfg: Config, *, max_new, key, eos_id, pad_id, temperature=1.0, lora=None, lora_scale=1.0):
    """Reference implementation without a KV cache (recomputes the prefix each step).
    Kept for the cached-vs-uncached equivalence test; use `generate` for real runs.
    """
    B, L = prompt_ids.shape
    Lmax = L + max_new
    ids = jnp.concatenate([prompt_ids, jnp.full((B, max_new), pad_id, prompt_ids.dtype)], axis=1)
    mask = jnp.concatenate([prompt_mask, jnp.zeros((B, max_new), prompt_mask.dtype)], axis=1)
    gen_logp = jnp.zeros((B, Lmax), jnp.float32)
    finished = jnp.zeros(B, dtype=bool)
    keys = jax.random.split(key, max_new)
    arangeB = jnp.arange(B)

    # Per-step logits ONLY at the current position: [B, V] instead of the full
    # [B, T, V] tensor (which at vocab 152k is gigabytes and OOMs the T4 autotuner).
    # `cur` is passed as a traced index so this compiles exactly once.
    def _step(pr, i, m, c):
        h = backbone(pr, i, cfg, m, _positions(m), lora, lora_scale)  # [B, T, H]
        h_cur = jax.lax.dynamic_index_in_dim(h, c, axis=1, keepdims=False)  # [B, H]
        return h_cur @ _lm_head(pr, cfg)  # [B, V]

    step_fn = jax.jit(_step)
    for s in range(max_new):
        cur = L + s
        raw = step_fn(params, ids, mask, jnp.int32(cur - 1)).astype(jnp.float32)  # [B, V]
        if temperature > 0:
            logits = raw / temperature
            nxt = jax.random.categorical(keys[s], logits, axis=-1)
        else:
            logits = raw
            nxt = jnp.argmax(raw, axis=-1)
        lp = jax.nn.log_softmax(logits, axis=-1)[arangeB, nxt]
        active = ~finished
        nxt = jnp.where(finished, pad_id, nxt)
        ids = ids.at[:, cur].set(nxt)
        mask = mask.at[:, cur].set(active.astype(mask.dtype))
        gen_logp = gen_logp.at[:, cur].set(lp * active)
        finished = finished | (nxt == eos_id)
        if bool(jnp.all(finished)):
            break

    resp_mask = mask.at[:, :L].set(0)
    return ids, mask, resp_mask, gen_logp


# --------------------------------------------------------------------------------------
# [Day 4] GRPO: LoRA params + group advantages + clipped PG loss with k3 KL
#
# Memory design: the reference model is the SAME frozen base run with lora=None, so
# there is only one copy of weights on the card. Only LoRA A/B tensors are trainable,
# so Adam optimizer state is tiny. B is zero-initialised -> at step 0 the policy equals
# the reference (delta = 0), which keeps the first KL term at 0.
# --------------------------------------------------------------------------------------
LORA_TARGETS = ("wq", "wk", "wv", "wo")


def init_lora(params, cfg: Config, *, rank: int = 16, seed: int = 0, targets=LORA_TARGETS):
    """LoRA adapters on attention projections. A ~ small normal, B = 0 (delta starts at 0)."""
    key = jax.random.PRNGKey(seed)
    layers = []
    for i in range(cfg.n_layers):
        layer = {}
        for t in targets:
            in_dim, out_dim = params["layers"][i][t].shape  # W stored as [in, out]
            key, sub = jax.random.split(key)
            a = jax.random.normal(sub, (in_dim, rank), jnp.float32) / jnp.sqrt(in_dim)
            b = jnp.zeros((rank, out_dim), jnp.float32)
            layer[t] = {"A": a, "B": b}
        layers.append(layer)
    return {"layers": layers}


def group_advantages(rewards, group_size: int, eps: float = 1e-6):
    """GRPO advantage: z-score rewards within each group of `group_size` rollouts. [N] -> [N]."""
    r = rewards.reshape(-1, group_size)
    adv = (r - r.mean(axis=1, keepdims=True)) / (r.std(axis=1, keepdims=True) + eps)
    return adv.reshape(-1)


def grpo_loss(pol_lp, ref_lp, old_lp, adv, resp_mask, *, clip_eps: float = 0.2, kl_beta: float = 0.001):
    """Token-level GRPO loss. All logprob args [B, T] position-aligned; adv [B]; mask [B, T].

    - clipped policy-gradient with importance ratio pi_theta/pi_old
    - DeepSeek k3 KL estimator to the reference (exp(r) - r - 1 >= 0)
    """
    ratio = jnp.exp(pol_lp - old_lp)  # [B, T]; == 1 on the first on-policy epoch
    a = adv[:, None]  # broadcast advantage over tokens
    pg = -jnp.minimum(ratio * a, jnp.clip(ratio, 1 - clip_eps, 1 + clip_eps) * a)
    logr = ref_lp - pol_lp
    kl = jnp.exp(logr) - logr - 1.0
    per_tok = pg + kl_beta * kl
    return (per_tok * resp_mask).sum() / jnp.clip(resp_mask.sum(), 1.0, None)


def save_lora(lora, path: str):
    """Serialize the LoRA adapter to a portable .npz (the only thing worth shipping)."""
    flat = {}
    for i, layer in enumerate(lora["layers"]):
        for proj, ab in layer.items():
            flat[f"{i}__{proj}__A"] = np.asarray(ab["A"])
            flat[f"{i}__{proj}__B"] = np.asarray(ab["B"])
    np.savez(path, **flat)


def load_lora(path: str):
    """Load a LoRA adapter saved by `save_lora` back into the pytree shape `forward` expects."""
    data = np.load(path if path.endswith(".npz") else path + ".npz")
    layers: dict[int, dict] = {}
    for k in data.files:
        i, proj, m = k.split("__")
        layers.setdefault(int(i), {}).setdefault(proj, {})[m] = jnp.asarray(data[k])
    return {"layers": [layers[i] for i in range(max(layers) + 1)]}


# --------------------------------------------------------------------------------------
# Day-1 gate: match HF logits
# --------------------------------------------------------------------------------------
def _logit_match_gate(repo: str = REPO, tol: float = 1e-2):
    from transformers import AutoTokenizer

    print(f"loading {repo} ...")
    params, cfg, path = load_params(repo)
    print(f"  config: {cfg}")

    tok = AutoTokenizer.from_pretrained(path)
    msgs = [{"role": "user", "content": "Using the numbers [3, 7, 11], make 28."}]
    # apply_chat_template's return type varies across transformers versions:
    # list[int], dict/BatchEncoding {'input_ids': ...}, or batched [[...]]. Normalize all.
    out = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True)
    if hasattr(out, "keys"):
        out = out["input_ids"]
    if out and isinstance(out[0], (list, tuple)):
        out = out[0]
    ids = np.asarray([list(out)], dtype=np.int32)
    print(f"  prompt tokens: {ids.shape}")

    logits = np.asarray(forward(params, jnp.asarray(ids), cfg))  # [1, T, vocab]

    # HF reference (float32, upcast from the same bf16 checkpoint -> identical weights)
    import torch
    from transformers import AutoModelForCausalLM

    hf = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float32)
    with torch.no_grad():
        ref = hf(torch.tensor(ids)).logits.float().numpy()

    diff = np.abs(logits - ref).max()
    argmax_match = (logits.argmax(-1) == ref.argmax(-1)).mean()
    print(f"\n  max |logit diff| = {diff:.3e}   (tol {tol:.0e})")
    print(f"  next-token argmax agreement = {argmax_match:.3f}")
    assert diff < tol, f"GATE FAILED: {diff:.3e} >= {tol:.0e} - check RoPE theta / GQA mapping / tied embeddings"
    print("\n  [PASS] Day-1 gate - forward is correct, proceed to the sampler.")


if __name__ == "__main__":
    _logit_match_gate()
