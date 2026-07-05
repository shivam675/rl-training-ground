"""Phase 7: regressions for the never-lose-a-model save path, the mid-episode
evaluation stop, and project-scoped run listing."""

from __future__ import annotations

import json
from pathlib import Path

from backend.config_service import ConfigService
from backend.models import EnvConfig
from backend.rl import evaluation
from backend.rl.training_worker import (
    MAX_HISTORY_POINTS,
    TrainingWorker,
    _promote_latest_checkpoint,
    _save_model_atomic,
)
from backend.run_registry import RunRegistry


def test_telemetry_since_cursor_survives_history_trim(tmp_path: Path):
    """After the front-trim, `since` (a global index) must still return the
    correct tail — the /training/telemetry endpoint slices points[since-dropped:]."""
    worker = TrainingWorker(tmp_path)  # not started: _telemetry_path is None, no disk writes
    n = MAX_HISTORY_POINTS + 100
    for i in range(n):
        worker.record_telemetry({"i": i})
    assert worker._telemetry_dropped == 100
    assert len(worker.telemetry) == MAX_HISTORY_POINTS
    assert worker.telemetry[0]["i"] == 100  # oldest kept is the 101st point
    # A client caught up to global index `since` gets exactly the newer tail.
    since = n - 10
    tail = worker.telemetry[max(0, since - worker._telemetry_dropped):]
    assert [p["i"] for p in tail] == list(range(n - 10, n))


def test_resume_stats_fall_back_to_run_root_for_checkpoints(tmp_path: Path):
    """Resuming from checkpoints/step_X.zip must find run_dir/vecnormalize.pkl,
    not a nonexistent checkpoints/vecnormalize.pkl (which silently resets stats)."""
    from backend.rl.env_factory import _resolve_resume_stats

    (tmp_path / "vecnormalize.pkl").write_bytes(b"stats")
    (tmp_path / "model.zip").write_bytes(b"final")
    checkpoints = tmp_path / "checkpoints"
    checkpoints.mkdir()
    (checkpoints / "step_40000.zip").write_bytes(b"ckpt")

    # Resume from the run-root model: sidecar is right there.
    assert _resolve_resume_stats(str(tmp_path / "model.zip")) == tmp_path / "vecnormalize.pkl"
    # Resume from a checkpoint: falls back one level up to the run root.
    assert _resolve_resume_stats(str(checkpoints / "step_40000.zip")) == tmp_path / "vecnormalize.pkl"
    # No stats anywhere → None (fresh normalizer, unchanged behaviour).
    assert _resolve_resume_stats(str(tmp_path / "runs" / "other" / "model.zip")) is None
    assert _resolve_resume_stats(None) is None


# --------------------------------------------------------------- model salvage


class _FakeModel:
    def __init__(self, payload: bytes = b"model-bytes", fail: bool = False):
        self.payload = payload
        self.fail = fail

    def save(self, path: str) -> None:
        if self.fail:
            raise RuntimeError("boom")
        # SB3 appends .zip; our atomic helper passes the explicit tmp name.
        Path(path).write_bytes(self.payload)


def test_save_model_atomic_writes_and_cleans_tmp(tmp_path: Path):
    assert _save_model_atomic(_FakeModel(), tmp_path) is True
    assert (tmp_path / "model.zip").read_bytes() == b"model-bytes"
    assert not (tmp_path / "model.zip.tmp").exists()


def test_save_model_atomic_failure_leaves_no_partial(tmp_path: Path):
    assert _save_model_atomic(_FakeModel(fail=True), tmp_path) is False
    assert not (tmp_path / "model.zip").exists()
    assert not (tmp_path / "model.zip.tmp").exists()


def test_promote_latest_checkpoint_recovers_a_crashed_run(tmp_path: Path):
    checkpoints = tmp_path / "checkpoints"
    checkpoints.mkdir()
    (checkpoints / "step_20000.zip").write_bytes(b"older")
    (checkpoints / "step_40000.zip").write_bytes(b"newest")  # highest step wins
    assert _promote_latest_checkpoint(tmp_path) is True
    assert (tmp_path / "model.zip").read_bytes() == b"newest"


def test_promote_skips_when_model_already_present(tmp_path: Path):
    (tmp_path / "model.zip").write_bytes(b"final")
    checkpoints = tmp_path / "checkpoints"
    checkpoints.mkdir()
    (checkpoints / "step_100.zip").write_bytes(b"ckpt")
    assert _promote_latest_checkpoint(tmp_path) is False
    assert (tmp_path / "model.zip").read_bytes() == b"final"


# ------------------------------------------------------ evaluation stop signal


class _StopAfter:
    """should_stop() that stays False long enough to enter one episode, then
    flips True mid-step. Without the in-loop check the episode would run to the
    5000-step cap."""

    def __init__(self, flip_on_call: int):
        self.calls = 0
        self.flip_on_call = flip_on_call

    def __call__(self) -> bool:
        self.calls += 1
        return self.calls >= self.flip_on_call


class _EndlessEnv:
    """Never terminates or truncates — only an honoured stop ends evaluation."""

    def __init__(self, *_args, **_kwargs):
        self.manager = None

    def reset(self, *args, **kwargs):
        return [[0.0]]

    def step(self, _action):
        return [[0.0]], [0.0], [False], [{}]

    def close(self):
        pass


def test_evaluation_stops_mid_episode(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(evaluation, "_load_model", lambda _p: _FakeModelPredict())
    monkeypatch.setattr(evaluation, "make_vecnormalize_env", lambda *_a, **_k: _EndlessEnv())
    model_path = tmp_path / "model.zip"
    model_path.write_bytes(b"x")
    summary = evaluation.evaluate_model(
        model_path,
        EnvConfig(urdf_path="r2d2.urdf"),
        episodes=3,
        deterministic=True,
        should_stop=_StopAfter(flip_on_call=3),
        broadcast=None,
    )
    # The first episode is cut short far below the 5000-step cap.
    assert summary["episodes"], "the partial episode should be recorded"
    assert summary["episodes"][0]["length"] < 50


class _FakeModelPredict:
    def predict(self, _obs, deterministic=True):
        return [0.0], None


# ----------------------------------------------------- project-scoped listing


def _make_run(runs_dir: Path, name: str, project_id: str | None) -> None:
    run_dir = runs_dir / name
    run_dir.mkdir(parents=True)
    (run_dir / "model.zip").write_bytes(b"m")
    config = {"algorithm": "PPO", "total_timesteps": 1000, "config": {"urdf_path": "r2d2.urdf"}}
    if project_id is not None:
        config["config"]["project_id"] = project_id
        config["config"]["project_name"] = f"proj-{project_id}"
    (run_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")


def test_list_runs_filters_by_project(tmp_path: Path):
    runs = tmp_path / "runs"
    _make_run(runs, "20260101-000001", project_id="aaa")
    _make_run(runs, "20260101-000002", project_id="bbb")
    _make_run(runs, "20260101-000003", project_id=None)  # legacy/untagged
    registry = RunRegistry(runs)

    all_runs = registry.list_runs()
    assert len(all_runs) == 3
    assert {r.get("project_id") for r in all_runs} == {"aaa", "bbb", None}

    only_a = registry.list_runs(project_id="aaa")
    assert [r["name"] for r in only_a] == ["20260101-000001"]
    assert only_a[0]["project_name"] == "proj-aaa"

    # Untagged runs never leak into a project-scoped view.
    assert registry.list_runs(project_id="zzz") == []


# ------------------------------------------------------------ project identity


def test_ensure_identity_assigns_stable_id_and_name(tmp_path: Path):
    service = ConfigService(tmp_path)
    base = EnvConfig(urdf_path="r2d2.urdf")
    assert base.project_id is None

    stamped = service.ensure_identity(base, name="My Robot")
    assert stamped.project_id and len(stamped.project_id) == 32
    assert stamped.project_name == "My Robot"

    # A second pass keeps the same id (stable across patches/saves).
    again = service.ensure_identity(stamped)
    assert again.project_id == stamped.project_id
    assert again is stamped  # unchanged → same object
