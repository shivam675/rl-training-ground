# MuJoCo/MJX Migration First Slice

EasyRTG currently centers simulation on `backend/simulation/pybullet_manager.py`.
The live viewport, `RtgGymEnv`, reward builder, SB3 training worker, evaluation
worker, and tuning worker all assume PyBullet state.

This slice keeps that path intact and adds two side-by-side lanes:

- `MuJoCoBackend`: single-environment model loading, stepping, names, and preview.
- `MJXBackend`: JAX/MJX batched stepping for no-render training checks.
- `MJXTrainingWorker`: separate worker for `sim_backend="mjx"` and the tiny
  `point_reach` task.

## Install

CPU install:

```bash
pip install -r backend/requirements.txt
```

For NVIDIA CUDA, replace the plain JAX wheel with the official CUDA wheel:

```bash
pip install -U "jax[cuda12]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

## Smoke Tests

```bash
python scripts/mujoco_smoke_test.py
python scripts/mjx_smoke_test.py
python scripts/mjx_batched_test.py --num-envs 1024
python -m backend.rl.mjx_training_worker --task point_reach --num-envs 1024 --timesteps 10000
```

## API

`GET /simulation/backends` reports PyBullet, MuJoCo, MJX availability, missing
optional packages, and detected JAX devices.

`POST /training/start` is backward-compatible. Omit `sim_backend` for PyBullet.
Use `{"sim_backend":"mjx","mjx_task":"point_reach","num_envs":1024}` for the
first MJX lane.

## Asset Strategy

MJCF is the canonical MuJoCo/MJX format. URDF conversion, mesh resolution,
inertia repair, robot-arm reach, pick/place, and quadrupeds are deferred until
the falling-box and point-reach paths work reliably.
