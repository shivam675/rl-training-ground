# GPU setup on Linux (Ubuntu)

Reproducible setup for the SB3 lane's CUDA Torch on Linux. (The massively
parallel GPU lane is Isaac Sim, which ships its own Python and has its own
setup — see the Isaac Sim install docs.)

## Why a separate venv

Ubuntu 26.04 ships **Python 3.14**, but `torch` has no 3.14 wheels. We use
[`uv`](https://docs.astral.sh/uv/) to install a standalone **Python 3.12** in
user space (no sudo, no system Python changes).

## Steps

```bash
# 1. uv (user-space, no sudo)
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

# 2. Python 3.12 venv at .venv (start_backend.sh prefers .venv/bin/python)
cd /home/sentinal/code/rl-training-ground
uv venv --python 3.12 .venv

# 3. Base deps. torch from PyPI is already CUDA-enabled (torch 2.12+cu130).
export VIRTUAL_ENV="$PWD/.venv"
uv pip install -r backend/requirements.txt
```

## Verify

```bash
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

- **Isaac Sim** — the massively parallel GPU lane: thousands of envs vectorized
  on the GPU, all training one shared policy. Runs as a separate process under
  Isaac Sim's own Python; see the Isaac Sim setup docs.
- **SB3** — GPU only helps for CNN / large-MLP policies; small MLPs train
  *faster on CPU* (SB3 warns about this). The SB3 speedup comes from **parallel
  envs** (`SubprocVecEnv`, capped at CPU cores), not CUDA. Override the device
  with `EASYRTG_SB3_DEVICE=cuda|cpu` if needed.
