from __future__ import annotations

import argparse

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx


XML = """
<mujoco>
  <option timestep="0.002"/>
  <worldbody>
    <geom type="plane" size="2 2 0.1"/>
    <body name="box" pos="0 0 1">
      <joint name="free" type="free"/>
      <geom type="box" size="0.05 0.05 0.05" mass="1"/>
    </body>
  </worldbody>
</mujoco>
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-envs", type=int, default=1024)
    args = parser.parse_args()

    mj_model = mujoco.MjModel.from_xml_string(XML)
    mx_model = mjx.put_model(mj_model)
    base = mjx.make_data(mx_model)

    def make_data(height):
        return base.replace(qpos=base.qpos.at[2].set(height))

    batch_data = jax.vmap(make_data)(jnp.linspace(0.5, 1.5, args.num_envs))

    @jax.jit
    def batched_step(data):
        return jax.vmap(lambda d: mjx.step(mx_model, d))(data)

    for _ in range(100):
        batch_data = batched_step(batch_data)

    print("devices:", jax.devices())
    print("batch qpos shape:", batch_data.qpos.shape)
    print("Batched MJX test passed")


if __name__ == "__main__":
    main()
