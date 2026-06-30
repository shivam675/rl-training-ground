"""Frozen-app entry point for the EasyRTG backend.

A PyInstaller bundle has no ``python.exe``, so ``python -m uvicorn
backend.main:app`` can't be used there. The standalone build
(scripts/build_app.ps1) freezes THIS module instead: it runs the same ASGI app
via uvicorn programmatically. Running it as a plain script works too:

    python backend/run_server.py

Host/port are overridable via EASYRTG_HOST / EASYRTG_PORT.
"""

from __future__ import annotations

import os
import sys

# When run as a plain script (python backend/run_server.py), sys.path[0] is the
# backend/ dir, so `import backend.*` fails. Put the repo root on the path. In a
# PyInstaller bundle the package is collected directly, so skip it there.
if not getattr(sys, "frozen", False):
    _repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _repo_root not in sys.path:
        sys.path.insert(0, _repo_root)


def main() -> None:
    import uvicorn

    from backend.main import app

    host = os.environ.get("EASYRTG_HOST", "127.0.0.1")
    port = int(os.environ.get("EASYRTG_PORT", "8000"))
    # timeout_graceful_shutdown matches scripts/start_backend.ps1: the live
    # viewport WebSocket never closes on its own, so cap the drain so a single
    # SIGINT/terminate exits promptly instead of hanging.
    uvicorn.run(app, host=host, port=port, timeout_graceful_shutdown=2)


def _selfcheck() -> int:
    """Import the deps that backend.main loads only lazily (at train/tune time),
    so a packaging gap fails the BUILD instead of the user. /health alone can't
    catch these because SB3/optuna are imported inside functions.
    """
    import importlib

    # SB3's logger imports matplotlib + pandas at module top; optuna drives tuning.
    # MJX training imports most of its heavy stack lazily from the worker.
    mods = [
        "matplotlib.figure",
        "pandas",
        "optuna",
        "stable_baselines3",
        "pybullet",
        "mujoco",
        "jax",
        "brax",
        "flax",
        "optax",
        "chex",
        "orbax.checkpoint",
    ]
    for name in mods:
        importlib.import_module(name)
    from stable_baselines3 import A2C, PPO, SAC, TD3  # noqa: F401
    import jax
    import torch

    cuda_available = bool(torch.version.cuda and torch.cuda.is_available())
    if os.environ.get("EASYRTG_REQUIRE_CUDA") == "1" and not cuda_available:
        raise RuntimeError("CUDA Torch is required but unavailable in the frozen backend.")

    jax_gpu_devices = []
    for kind in ("gpu", "cuda"):
        try:
            jax_gpu_devices.extend(jax.devices(kind))
        except Exception:
            pass
    if os.environ.get("EASYRTG_REQUIRE_JAX_GPU") == "1" and not jax_gpu_devices:
        raise RuntimeError("JAX GPU is required but unavailable in the frozen backend.")

    torch_device = torch.cuda.get_device_name(0) if cuda_available else "CPU"
    jax_devices = [str(device) for device in jax.devices()]
    print(
        "selfcheck OK:",
        ", ".join(mods),
        f"| torch {torch.__version__} | CUDA {torch.version.cuda} | {torch_device}",
        f"| jax {jax.__version__} | backend {jax.default_backend()} | devices {jax_devices}",
    )
    return 0


if __name__ == "__main__":
    # Required so PyInstaller-frozen children (e.g. torch/SB3 worker processes)
    # don't re-launch the whole app.
    import multiprocessing

    multiprocessing.freeze_support()
    if len(sys.argv) > 1 and sys.argv[1] == "--selfcheck":
        sys.exit(_selfcheck())
    main()
