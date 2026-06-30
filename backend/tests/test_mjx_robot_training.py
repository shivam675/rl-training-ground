import pytest

from backend.config_service import ConfigService
from backend.models import ActionSelection, EnvConfig, TrainingStartRequest


def test_action_patch_accepts_mjx_pd_fields(tmp_path):
    service = ConfigService(tmp_path)
    config = EnvConfig(
        urdf_path="robot.urdf",
        actions=[
            ActionSelection(
                joint_index=0,
                joint_name="hip",
                enabled=False,
                control_mode="position",
            )
        ],
    )

    updated = service.apply_patch(
        config,
        {
            "actions": [
                {
                    "joint_index": 0,
                    "enabled": True,
                    "kp": 45.0,
                    "kd": 1.5,
                    "torque_limit": 12.0,
                }
            ]
        },
    )

    action = updated.actions[0]
    assert action.enabled is True
    assert action.kp == 45.0
    assert action.kd == 1.5
    assert action.torque_limit == 12.0


def test_mjx_action_mapping_uses_enabled_joint_names():
    mujoco = pytest.importorskip("mujoco")
    from backend.rl.mjx_robot_env import build_action_specs

    xml = """
    <mujoco>
      <worldbody>
        <body name="base">
          <joint name="root" type="free"/>
          <geom type="box" size=".2 .1 .05" mass="1"/>
          <body name="leg">
            <joint name="hip" type="hinge" axis="0 1 0" range="-1 1"/>
            <geom type="capsule" fromto="0 0 0 0 0 -.2" size=".03" mass=".1"/>
          </body>
        </body>
      </worldbody>
    </mujoco>
    """
    model = mujoco.MjModel.from_xml_string(xml)
    config = EnvConfig(
        urdf_path="inline.xml",
        actions=[
            ActionSelection(
                joint_index=0,
                joint_name="hip",
                enabled=True,
                control_mode="position",
                scale_low=-0.5,
                scale_high=0.5,
                kp=10.0,
                kd=0.5,
            )
        ],
    )

    specs = build_action_specs(model, config)

    assert len(specs) == 1
    assert specs[0].joint_name == "hip"
    assert specs[0].control_mode == "position"
    assert specs[0].dof_index >= 0


def test_load_mj_model_returns_mjx_compatible_cylinder_box_model(tmp_path):
    pytest.importorskip("mujoco")
    from mujoco import mjx
    from backend.rl.mjx_robot_env import load_mj_model

    robot = tmp_path / "robot.xml"
    robot.write_text(
        """
        <mujoco>
          <worldbody>
            <body name="box" pos="0 0 .1">
              <freejoint/>
              <geom type="box" size=".1 .1 .1" mass="1"/>
            </body>
            <body name="cylinder" pos="0 0 .3">
              <freejoint/>
              <geom type="cylinder" size=".05 .1" mass="1"/>
            </body>
          </worldbody>
        </mujoco>
        """,
        encoding="utf-8",
    )

    model = load_mj_model(str(robot))

    assert mjx.put_model(model).ngeom == 2


def test_configurable_mjx_env_steps_and_rewards(tmp_path):
    pytest.importorskip("mujoco")
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    from backend.models import ObservationSelection, RewardComponent
    from backend.rl.mjx_robot_env import ConfigurableMJXRobotEnv

    robot = tmp_path / "robot.xml"
    robot.write_text(
        """
        <mujoco>
          <worldbody>
            <body name="base" pos="0 0 .3">
              <joint name="root" type="free"/>
              <geom type="box" size=".2 .1 .05" mass="1"/>
              <body name="leg">
                <joint name="hip" type="hinge" axis="0 1 0" range="-1 1"/>
                <geom type="capsule" fromto="0 0 0 0 0 -.2" size=".03" mass=".1"/>
              </body>
            </body>
          </worldbody>
        </mujoco>
        """,
        encoding="utf-8",
    )
    config = EnvConfig(
        urdf_path=str(robot),
        observations=[
            ObservationSelection(key="base_position", enabled=True),
            ObservationSelection(key="joint_positions", enabled=True),
        ],
        actions=[
            ActionSelection(
                joint_index=0,
                joint_name="hip",
                enabled=True,
                control_mode="position",
                scale_low=-0.5,
                scale_high=0.5,
            )
        ],
        rewards=[
            RewardComponent(key="stay_alive", enabled=True, weight=1.0),
            RewardComponent(key="energy", enabled=True, weight=-0.01),
        ],
        terminations={"max_steps": 5, "min_base_height": 0.05},
    )

    env = ConfigurableMJXRobotEnv(config)
    state = env.reset(jax.random.PRNGKey(0))
    state = state.replace(
        info={
            **state.info,
            "steps": jnp.array(0),
            "truncation": jnp.array(0.0),
        }
    )
    next_state = env.step(state, jnp.zeros((env.action_size,)))

    assert env.observation_size == 4
    assert env.action_size == 1
    assert next_state.obs.shape == (4,)
    assert bool(jnp.isfinite(next_state.reward))
    assert set(next_state.metrics) == set(state.metrics)
    assert set(next_state.info) == set(state.info)


def test_mjx_worker_robot_task_requires_config(tmp_path):
    from backend.rl.mjx_training_worker import MJXTrainingWorker

    worker = MJXTrainingWorker(tmp_path)
    req = TrainingStartRequest(sim_backend="mjx", mjx_task="robot")

    with pytest.raises(ValueError, match="config"):
        worker.start(req)


def test_rest_mjx_robot_training_uses_current_config(monkeypatch):
    from fastapi.testclient import TestClient

    from backend import main
    from backend.models import ObservationSelection, RewardComponent, TrainingStatus

    class FakeMjxWorker:
        def __init__(self):
            self.started = None
            self.status = TrainingStatus(active=False, backend="mjx")
            self.telemetry = []

        def start(self, req):
            self.started = req
            return {"ok": True, "run_dir": "fake-run", "backend": "mjx"}

        def is_alive(self):
            return True

        def drain_events(self):
            return []

    config = EnvConfig(
        urdf_path="robot.xml",
        observations=[ObservationSelection(key="base_position", enabled=True)],
        actions=[ActionSelection(joint_index=0, joint_name="hip", enabled=True)],
        rewards=[RewardComponent(key="stay_alive", enabled=True, weight=1.0)],
    )

    class FakeConfigService:
        def load(self):
            return None

        def current_or_default(self, sim):
            return config

        def validate(self, candidate, sim):
            assert candidate is config
            return []

    fake_worker = FakeMjxWorker()
    monkeypatch.setattr(main, "mjx_training_worker", fake_worker, raising=False)
    monkeypatch.setattr(main, "config_service", FakeConfigService(), raising=False)

    with TestClient(main.app) as client:
        res = client.post(
            "/training/start",
            json={"sim_backend": "mjx", "mjx_task": "robot", "num_envs": 8},
        )

    assert res.status_code == 200, res.text
    assert fake_worker.started is not None
    assert fake_worker.started.config is config
    assert fake_worker.started.mjx_task == "robot"


def test_agent_start_training_can_use_mjx_backend(tmp_path):
    from backend.agents.tools import AgentToolbox
    from backend.models import ObservationSelection, RewardComponent

    config = EnvConfig(
        urdf_path="robot.xml",
        observations=[ObservationSelection(key="base_position", enabled=True)],
        actions=[ActionSelection(joint_index=0, joint_name="hip", enabled=True)],
        rewards=[RewardComponent(key="stay_alive", enabled=True, weight=1.0)],
    )
    captured = {}

    class FakeConfigService:
        def current_or_default(self, sim):
            return config

        def validate(self, candidate, sim):
            assert candidate is config
            return []

    class FakeTrainingWorker:
        def start(self, req):
            raise AssertionError("training_starter should handle dispatch")

    def starter(req):
        captured["req"] = req
        return {"ok": True, "backend": req.sim_backend}

    toolbox = AgentToolbox(
        object(),
        FakeTrainingWorker(),
        tmp_path,
        config_service=FakeConfigService(),
        training_starter=starter,
    )

    result = toolbox._start_training(
        sim_backend="mjx",
        num_envs=16,
        mjx_task="robot",
    )

    assert result == {"ok": True, "backend": "mjx"}
    assert captured["req"].sim_backend == "mjx"
    assert captured["req"].num_envs == 16
    assert captured["req"].mjx_task == "robot"
    assert captured["req"].config is config


def test_agent_tools_expose_simulation_backends(tmp_path):
    from backend.agents.tools import READ_TOOLS, AgentToolbox

    class FakeTrainingWorker:
        def start(self, req):
            return {"ok": True}

    toolbox = AgentToolbox(object(), FakeTrainingWorker(), tmp_path)
    names = {tool["function"]["name"] for tool in toolbox.definitions()}

    assert "get_simulation_backends" in READ_TOOLS
    assert "get_simulation_backends" in names
    assert "pybullet" in toolbox._get_simulation_backends()["available"]
