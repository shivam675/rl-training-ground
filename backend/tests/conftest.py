"""Hermetic isolation for the backend test suite.

The suite imports the live ``backend.main`` singletons, which are wired to real
on-disk dirs (``project_configs/current_env.json``, ``app_settings/``,
``runs/``). Without isolation, running the tests overwrites the user's current
env config + app settings and litters ``runs/`` with junk timestamped runs
(test_phase2/3 POST a real ``/training/start``). See docs/APP_RECON.md §6.

Isolation is **session-scoped**, not per-test, on purpose: the suite is an
ordered pipeline that shares mutable disk state across files (e.g. phase1
enables + saves a config that phase2/3 read back, phase2 trains a model that its
own later test resumes). Per-test tmp dirs would break that chain, and a
function-scoped ``monkeypatch`` also applies too late — module-scoped ``client``
fixtures (phase5) run setup writes before it. So patch once at session start,
before any fixture, and share one tmp state dir for the whole run. The user's
real dirs are never touched.
"""

from __future__ import annotations

import shutil

import pytest


@pytest.fixture(scope="session", autouse=True)
def isolate_on_disk_state(tmp_path_factory):
    from backend import main

    mp = pytest.MonkeyPatch()
    tmp = tmp_path_factory.mktemp("rtg_state")
    runs = tmp / "runs"
    cfg = tmp / "project_configs"
    settings = tmp / "app_settings"
    for d in (runs, cfg, settings, settings / "logs"):
        d.mkdir(parents=True, exist_ok=True)

    # Seed the session config from the real one so the ordered pipeline starts
    # from the same state it would with the real file (reads preserved, writes
    # redirected).
    real_cfg = main.config_service.path
    if real_cfg.exists():
        shutil.copy2(real_cfg, cfg / real_cfg.name)

    # Module-level path constants (resolved at call time inside endpoints).
    mp.setattr(main, "RUNS_DIR", runs)
    mp.setattr(main, "PROJECT_CONFIG_DIR", cfg)
    mp.setattr(main, "APP_SETTINGS_DIR", settings)
    mp.setattr(main, "OLLAMA_SETTINGS_PATH", settings / "ollama.json")
    mp.setattr(main, "AGENT_SETTINGS_PATH", settings / "agent_settings.json")
    mp.setattr(main, "APP_PREFERENCES_PATH", settings / "preferences.json")

    # Singletons captured their paths at construction — redirect the instances.
    mp.setattr(main.config_service, "config_dir", cfg)
    mp.setattr(main.config_service, "path", cfg / real_cfg.name)
    mp.setattr(main.config_service, "history_path", cfg / main.config_service.history_path.name)
    for worker in (main.registry, main.training_worker, main.isaac_training_worker, main.toolbox):
        mp.setattr(worker, "runs_dir", runs)

    yield
    mp.undo()
