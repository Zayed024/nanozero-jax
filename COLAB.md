# Running NanoZero on Colab (T4)

The whole pipeline is CPU-verified (19 tests). This is the GPU run that produces the
real numbers. Use a **GPU runtime** (Runtime → Change runtime type → T4 GPU).

## 0. One-time: push the project to GitHub (run locally, in your shell)

```bash
cd D:/agentica/nanozero-jax
git init
git add nanozero.py countdown.py train.py smoke_test.py test_*.py requirements.txt README.md COLAB.md .gitignore
git commit -m "NanoZero: tinker-free GRPO in JAX (Countdown, LoRA)"
git branch -M main
git remote add origin https://github.com/Zayed024/nanozero-jax.git   # create this repo first on github
git push -u origin main
```

## 1. Setup (Colab cell)

```python
!git clone https://github.com/Zayed024/nanozero-jax.git
%cd nanozero-jax
!pip install -q transformers safetensors huggingface_hub datasets optax
# Colab's preinstalled jax already has CUDA; verify it sees the GPU:
import jax; print("jax devices:", jax.devices())
```

## 2. Day-1 gate on REAL weights (the one thing never run yet)

```python
!python nanozero.py          # loads Qwen2.5-0.5B, asserts max|logits_jax - logits_hf| < 1e-2
```
If this fails, the fix is almost always: RoPE theta (must be 1e6), GQA head mapping, tied
embeddings, or a transpose. If it passes, the forward pass is correct — proceed.

## 3. Overfit sanity (cheap, proves the loop learns)

Train on a *tiny* fixed pool so the model can memorize it — `solved%` should climb within
tens of steps. If it doesn't move, the bug is in advantage/logprob/mask wiring, not hyperparams.

```python
import train
train.train(steps=50, n_prompts=4, group_size=8, max_new=256, pool_size=8, lr=1e-4)
```

## 4. Real training run (the headline)

```python
import train
lora = train.train(steps=200, n_prompts=8, group_size=8, max_new=256, rank=16, lr=1e-4)
# prints per-step loss/reward/solved% and a final: baseline -> trained (Δ) pass@1
# saves the adapter to nanozero_countdown_lora.npz
```
Watch the T4 memory: if OOM, lower `n_prompts`, `group_size`, or `max_new` (512→256→192).
Each step recomputes the full prefix (no KV cache yet), so keep `max_new` modest.

## 5. The ablation (capability floor)

Run the **identical** loop on your 10M TinyStories model instead of Qwen2.5-0.5B. Expected:
it cannot follow the format, so `solved%` stays ~0 — that's the honest result, it locates the
capability threshold near ~0.5B. Frame it as an ablation, not a failure.

## 6. Ship

```python
from huggingface_hub import HfApi
HfApi().upload_file(path_or_fileobj="nanozero_countdown_lora.npz",
                    path_in_repo="nanozero_countdown_lora.npz",
                    repo_id="Zayed024/nanozero-countdown-lora", repo_type="model")
```
Then put the `baseline → trained` pass@1 numbers + reward curve in the README.
