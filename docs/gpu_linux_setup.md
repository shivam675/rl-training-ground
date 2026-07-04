# GPU setup on Linux (Ubuntu)

This is the exact, reproducible setup used to get both training lanes on the
NVIDIA GPU. Verified on Ubuntu 26.04 + GTX 1660 Ti (6 GB, driver 595 / CUDA 13).

## Why a separate venv

Ubuntu 26.04 ships **Python 3.14**, but `torch`, `jax`, `brax`, and `mujoco`
have no 3.14 wheels. We use [`uv`](https://docs.astral.sh/uv/) to install a
standalone **Python 3.12** in user space (no sudo, no system Python changes).

## Steps

```bash
# 1. uv (user-space, no sudo)
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

# 2. Python 3.12 venv at .venv (start_backend.sh prefers .venv/bin/python)
cd /home/sentinal/code/rl-training-ground
uv venv --python 3.12 .venv

# 3. Base deps (this pulls CPU jax; we override next)
export VIRTUAL_ENV="$PWD/.venv"
uv pip install -r backend/requirements.txt

# 4. CUDA jax plugin (matches the installed jax version). torch from PyPI is
#    already CUDA-enabled (torch 2.12+cu130) -- no extra step for torch.
uv pip install "jax[cuda13]==0.10.2"
```

## Verify

```bash
.venv/bin/python -c "import jax; print(jax.default_backend(), jax.devices())"
# -> gpu [CudaDevice(id=0)]
.venv/bin/python -c "import torch; print(torch.cuda.is_available())"
# -> True
```

## ROS / Gazebo PYTHONPATH gotcha (important)

If you have ROS 2 / Gazebo sourced in your shell, it injects its **Python 3.14**
site-packages onto `PYTHONPATH`, which leaks into this 3.12 venv and breaks
imports (and pytest plugin discovery). `scripts/start_backend.sh` now `unset
PYTHONPATH` before launching, so the app is safe. For ad-hoc commands:

```bash
env -u PYTHONPATH .venv/bin/python -m pytest backend/tests/ -q
```

## Which lane uses the GPU?

- **MJX (brax PPO)** — the Isaac-Sim-style lane: `jax.vmap` over hundreds–
  thousands of envs on the GPU, all training one shared policy. This is where
  "spawn 1000+ robots so training finishes fast" actually happens. Set a large
  `num_envs` (UI default 1024).
- **SB3** — GPU only helps for CNN / large-MLP policies; small MLPs train
  *faster on CPU* (SB3 warns about this). The SB3 speedup comes from **parallel
  envs** (`SubprocVecEnv`, capped at CPU cores), not CUDA. Override the device
  with `EASYRTG_SB3_DEVICE=cuda|cpu` if needed.

## VRAM note (6 GB card)

The 1660 Ti has 6 GB. MJX env count is bounded by model size × num_envs. Simple
robots handle thousands of envs; large/complex MJCF models may need a smaller
`num_envs`. If you hit OOM, lower `num_envs` or set
`XLA_PYTHON_CLIENT_MEM_FRACTION=0.8`.
```
