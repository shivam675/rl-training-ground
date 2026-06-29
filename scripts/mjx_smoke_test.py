from __future__ import annotations

import jax
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
    mj_model = mujoco.MjModel.from_xml_string(XML)
    mx_model = mjx.put_model(mj_model)
    mx_data = mjx.make_data(mx_model)

    @jax.jit
    def rollout(data):
        def body(carry, _):
            data = mjx.step(mx_model, carry)
            return data, data.qpos

        return jax.lax.scan(body, data, None, length=1000)

    final_data, _qpos_history = rollout(mx_data)
    print("devices:", jax.devices())
    print("final qpos:", final_data.qpos)
    print("MJX smoke test passed")


if __name__ == "__main__":
    main()
