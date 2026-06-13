"""Registry over training run artifacts: metadata, telemetry and evaluations.

Every run lives in ``runs/<timestamp>/`` with config.json, telemetry.jsonl,
monitor.csv, model.zip and (after evaluations) evaluations.json. The registry
reads those artifacts on demand — the filesystem stays the source of truth, so
runs survive backend restarts and can be copied between machines.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any

MAX_CURVE_POINTS = 200


class RunRegistry:
    def __init__(self, runs_dir: Path):
        self.runs_dir = runs_dir

    # ------------------------------------------------------------------ paths

    def run_dir(self, name: str) -> Path | None:
        path = (self.runs_dir / name).resolve()
        if not path.is_dir() or self.runs_dir.resolve() not in path.parents:
            return None
        return path

    # ------------------------------------------------------------------- list

    def list_runs(
        self, limit: int = 50, project_id: str | None = None
    ) -> list[dict[str, Any]]:
        """List run summaries, newest first.

        When ``project_id`` is given, only runs tagged with that project are
        returned. Legacy/untagged runs (no project_id) are kept out of a
        project-scoped list — they surface in the unfiltered "All runs" view.
        """
        runs: list[dict[str, Any]] = []
        if not self.runs_dir.exists():
            return runs
        for run_dir in sorted(self.runs_dir.iterdir(), reverse=True):
            if not run_dir.is_dir():
                continue
            summary = self._summary(run_dir)
            if project_id is not None and summary.get("project_id") != project_id:
                continue
            runs.append(summary)
            if len(runs) >= limit:
                break
        return runs

    def _summary(self, run_dir: Path) -> dict[str, Any]:
        entry: dict[str, Any] = {
            "name": run_dir.name,
            "model_saved": (run_dir / "model.zip").exists(),
        }
        config = self._read_json(run_dir / "config.json")
        if config:
            entry["algorithm"] = config.get("algorithm")
            entry["total_timesteps"] = config.get("total_timesteps")
            entry["learning_rate"] = config.get("learning_rate")
            env_config = config.get("config") or {}
            entry["urdf_path"] = env_config.get("urdf_path")
            entry["project_id"] = env_config.get("project_id")
            entry["project_name"] = env_config.get("project_name")
        rewards = self._telemetry_rewards(run_dir)
        if rewards:
            entry["reward_best"] = max(rewards)
            entry["reward_last"] = rewards[-1]
        evaluations = self._read_json(run_dir / "evaluations.json") or []
        if evaluations:
            entry["eval_count"] = len(evaluations)
            entry["eval_best_mean"] = max(e.get("mean_reward", 0) for e in evaluations)
        return entry

    # ---------------------------------------------------------------- details

    def run_details(self, name: str) -> dict[str, Any] | None:
        run_dir = self.run_dir(name)
        if run_dir is None:
            return None
        details = self._summary(run_dir)
        details["config"] = self._read_json(run_dir / "config.json")
        details["evaluations"] = self._read_json(run_dir / "evaluations.json") or []
        details["telemetry"] = self._telemetry_curve(run_dir)
        details["checkpoints"] = sorted(
            p.name for p in (run_dir / "checkpoints").glob("step_*.zip")
        ) if (run_dir / "checkpoints").exists() else []
        return details

    def record_evaluation(self, name: str, summary: dict[str, Any]) -> None:
        run_dir = self.run_dir(name)
        if run_dir is None:
            return
        path = run_dir / "evaluations.json"
        evaluations = self._read_json(path) or []
        evaluations.append(summary)
        path.write_text(json.dumps(evaluations, indent=2), encoding="utf-8")

    # ----------------------------------------------------------------- export

    def export_bundle(self, name: str) -> Path | None:
        """Zip the run's artifacts into a portable bundle."""
        run_dir = self.run_dir(name)
        if run_dir is None:
            return None
        bundle = run_dir / f"easyrtg-run-{name}.zip"
        with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as archive:
            for artifact in (
                "model.zip",
                "config.json",
                "telemetry.jsonl",
                "monitor.csv",
                "evaluations.json",
                "training_log.txt",
                "vecnormalize.pkl",
                "normalization.json",
            ):
                path = run_dir / artifact
                if path.exists():
                    archive.write(path, arcname=artifact)
        return bundle

    # ---------------------------------------------------------------- compare

    def compare(self, names: list[str]) -> dict[str, Any]:
        rows = []
        for name in names:
            run_dir = self.run_dir(name)
            if run_dir is None:
                rows.append({"name": name, "error": "unknown run"})
                continue
            rows.append(self._summary(run_dir))
        return {"runs": rows}

    # ----------------------------------------------------------------- helpers

    @staticmethod
    def _read_json(path: Path) -> Any:
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

    def _telemetry_points(self, run_dir: Path) -> list[dict[str, Any]]:
        path = run_dir / "telemetry.jsonl"
        if not path.exists():
            return []
        points = []
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    points.append(json.loads(line))
        except (OSError, json.JSONDecodeError):
            pass
        return points

    def _telemetry_rewards(self, run_dir: Path) -> list[float]:
        return [
            p["reward_mean"]
            for p in self._telemetry_points(run_dir)
            if p.get("reward_mean") is not None
        ]

    def _telemetry_curve(self, run_dir: Path) -> list[dict[str, Any]]:
        points = self._telemetry_points(run_dir)
        if len(points) <= MAX_CURVE_POINTS:
            return points
        step = len(points) / MAX_CURVE_POINTS
        return [points[int(i * step)] for i in range(MAX_CURVE_POINTS)]
