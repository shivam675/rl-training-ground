from __future__ import annotations

import numpy as np

import mujoco


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
    model = mujoco.MjModel.from_xml_string(XML)
    data = mujoco.MjData(model)
    for _ in range(1000):
        mujoco.mj_step(model, data)
    print("qpos:", np.asarray(data.qpos))
    print("MuJoCo smoke test passed")


if __name__ == "__main__":
    main()
