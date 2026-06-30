#!/usr/bin/env pwsh
<#
Build a STANDALONE Windows app: Flutter frontend + PyInstaller-frozen backend.

Output (folder only — zip it yourself):
  dist\<AppName>\              the runnable folder
  dist\<AppName>\<AppName>.exe the ONLY exe the user launches

Layout (one launch exe; backend + Python runtime hidden in runtime\):
  <AppName>.exe                renamed Flutter runner (the one to double-click)
  flutter_windows.dll, data\   Flutter runtime (must sit beside the exe)
  runtime\backend.exe          frozen backend, auto-started by the app
  urdf_files\                  sample robots (only with -IncludeSamples)

Standalone = the end user needs NO Python install. The backend is frozen with
PyInstaller from the dev venv (.venv312). The build installs CUDA Torch when
needed and refuses to package a CPU-only backend. Stable-Baselines3 uses the
backend's CUDA device selection for training, tuning, and evaluation. MJX uses
GPU only when the build env has CUDA-enabled JAX; pass -RequireJaxGpu to fail
the build otherwise.

Usage:
  scripts\build_app.ps1                       full build (uses .venv312)
  scripts\build_app.ps1 -Python C:\env\python.exe   use a different env
  scripts\build_app.ps1 -TorchIndexUrl https://download.pytorch.org/whl/cu128
  scripts\build_app.ps1 -RequireJaxGpu     also require CUDA-enabled JAX for MJX
  scripts\build_app.ps1 -SkipBackend          rebuild frontend + repackage only
  scripts\build_app.ps1 -SkipFlutter          rebuild backend + repackage only
  scripts\build_app.ps1 -Clean                wipe build\ and dist\ first

Note: keep the app folder somewhere writable (Desktop, not Program Files) — the
frozen backend writes config/runs beside itself.
#>
param(
    [string]$Python = "",
    [string]$AppName = "EasyRTG",
    [string]$TorchIndexUrl = "https://download.pytorch.org/whl/cu128",
    [switch]$RequireJaxGpu,
    [switch]$IncludeSamples,   # bundle the whole urdf_files\ tree (~1.8 GB); off by default
    [switch]$SkipFlutter,
    [switch]$SkipBackend,
    [switch]$Clean
)

# Native tools here (python/pybullet/pip/PyInstaller/flutter) write progress and
# banners to stderr. Under Windows PowerShell 5.1, $ErrorActionPreference='Stop'
# would turn that stderr into a terminating NativeCommandError. So leave native
# commands alone (their exit codes are checked via $LASTEXITCODE) and force Stop
# only on the mutating cmdlets, where a failure must abort the build.
$ErrorActionPreference = 'Continue'
$PSDefaultParameterValues = @{
    'New-Item:ErrorAction'        = 'Stop'
    'Copy-Item:ErrorAction'       = 'Stop'
    'Rename-Item:ErrorAction'     = 'Stop'
    'Compress-Archive:ErrorAction' = 'Stop'
}
$scriptDir = $PSScriptRoot
$root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$buildDir = Join-Path $root 'build\standalone'
$distDir  = Join-Path $root 'dist'
$appDir   = Join-Path $distDir $AppName
$frozenBackend = Join-Path $buildDir 'pyi-dist\backend'
$flutterApp = Join-Path $root 'frontend\rtg-flutter-app'
$flutterRelease = Join-Path $flutterApp 'build\windows\x64\runner\Release'

function Section($m) { Write-Host "`n=== $m ===" -ForegroundColor Cyan }

# Pick a Python env that has pybullet (the hard miss last time) + torch.
function Resolve-BuildPython {
    if ($Python) { return (Resolve-Path $Python).Path }
    foreach ($v in @('.venv312', '.venv', '.venv-rtg')) {
        $cand = Join-Path $root "$v\Scripts\python.exe"
        if (Test-Path $cand) { return $cand }
    }
    throw "No build venv found (.venv312/.venv/.venv-rtg). Pass -Python <python.exe>."
}

Push-Location $root
try {
    if ($Clean) {
        Section "Clean build\ and dist\"
        Remove-Item -Recurse -Force $buildDir, $distDir -ErrorAction SilentlyContinue
    }
    New-Item -ItemType Directory -Force -Path $buildDir, $distDir | Out-Null

    # ---- Freeze backend -------------------------------------------------------
    if (-not $SkipBackend) {
        $buildPy = Resolve-BuildPython
        Write-Host "Build Python: $buildPy"

        Section "Ensure CUDA Torch"
        & $buildPy -c "import torch,sys; sys.exit(0 if torch.version.cuda and torch.cuda.is_available() else 1)"
        if ($LASTEXITCODE -ne 0) {
            Write-Host "Installing CUDA Torch from $TorchIndexUrl ..."
            & $buildPy -m pip install --upgrade --force-reinstall --no-deps torch --index-url $TorchIndexUrl
            if ($LASTEXITCODE -ne 0) { throw "CUDA Torch installation failed." }
        }

        Section "Verify GPU deps (pybullet + CUDA torch + MJX deps) and PyInstaller"
        & $buildPy -c "import pybullet,torch,sys; ok=bool(torch.version.cuda and torch.cuda.is_available()); print('pybullet OK; torch', torch.__version__, 'CUDA', torch.version.cuda, '-', torch.cuda.get_device_name(0) if ok else 'UNAVAILABLE'); sys.exit(0 if ok else 1)"
        if ($LASTEXITCODE -ne 0) {
            throw "Build requires CUDA Torch and a working NVIDIA GPU/driver. Use -TorchIndexUrl to select another official CUDA wheel channel."
        }
        $mjxProbe = @'
import os
import sys

import brax
import chex
import flax
import jax
import mujoco
import optax
import orbax.checkpoint

gpu = []
for kind in ("gpu", "cuda"):
    try:
        gpu.extend(jax.devices(kind))
    except Exception:
        pass

devices = [str(device) for device in jax.devices()]
print("MJX deps OK; jax", jax.__version__, "backend", jax.default_backend(), "devices", devices)
sys.exit(0 if (os.environ.get("EASYRTG_REQUIRE_JAX_GPU") != "1" or gpu) else 2)
'@
        $mjxProbePath = Join-Path $buildDir 'mjx_probe.py'
        Set-Content -LiteralPath $mjxProbePath -Value $mjxProbe -Encoding ASCII -ErrorAction Stop
        if ($RequireJaxGpu) { $env:EASYRTG_REQUIRE_JAX_GPU = '1' }
        try {
            & $buildPy $mjxProbePath
            $mjxCode = $LASTEXITCODE
        }
        finally {
            Remove-Item Env:\EASYRTG_REQUIRE_JAX_GPU -ErrorAction SilentlyContinue
            Remove-Item -LiteralPath $mjxProbePath -Force -ErrorAction SilentlyContinue
        }
        if ($mjxCode -eq 2) {
            throw "MJX deps are installed, but JAX GPU is unavailable. Install CUDA-enabled JAX or omit -RequireJaxGpu for a CPU-MJX build."
        }
        if ($mjxCode -ne 0) {
            throw "MJX dependencies are missing or broken. Install mujoco, jax, brax, flax, optax, chex, and orbax-checkpoint."
        }
        & $buildPy -c "import importlib.util,sys; sys.exit(0 if importlib.util.find_spec('PyInstaller') else 1)"
        if ($LASTEXITCODE -ne 0) {
            Write-Host "Installing PyInstaller..."
            & $buildPy -m pip install pyinstaller
            if ($LASTEXITCODE -ne 0) { throw "pip install pyinstaller failed." }
        }

        Section "Freeze backend (PyInstaller, onedir, CUDA)"
        # Excludes: big libs not used at runtime. Do NOT exclude matplotlib or
        # pandas: stable_baselines3.common.logger imports both at module top, so
        # excluding them breaks ALL training/tuning (SB3 is imported lazily, so
        # a /health check won't catch it — the tuning smoke-test below does).
        # Also keep sympy/networkx/jinja2/fsspec/filelock/mpmath (torch needs them).
        $pyiArgs = @(
            '-m', 'PyInstaller', '--noconfirm', '--clean',
            '--name', 'backend',
            '--distpath', (Join-Path $buildDir 'pyi-dist'),
            '--workpath', (Join-Path $buildDir 'pyi-work'),
            '--specpath', (Join-Path $buildDir 'pyi-spec'),
            '--paths', $root,
            '--collect-submodules', 'backend',
            '--hidden-import', 'pybullet',
            '--collect-all', 'pybullet_data',
            '--collect-all', 'stable_baselines3',
            '--collect-all', 'gymnasium',
            '--collect-all', 'uvicorn',
            '--collect-all', 'mujoco',
            '--collect-all', 'jax',
            '--collect-all', 'jaxlib',
            '--collect-all', 'brax',
            '--collect-all', 'flax',
            '--collect-all', 'optax',
            '--collect-all', 'chex',
            '--collect-all', 'orbax',
            '--collect-binaries', 'torch',
            # matplotlib is imported lazily by the Optuna/SB3 tuning path; without
            # it "Hyperparameter tuning failed: No module named 'matplotlib'".
            # collect-all grabs its mpl-data + dynamically-loaded backends.
            '--collect-all', 'matplotlib',
            '--copy-metadata', 'torch',
            '--copy-metadata', 'gymnasium',
            '--copy-metadata', 'stable_baselines3',
            '--copy-metadata', 'numpy',
            '--copy-metadata', 'mujoco',
            '--copy-metadata', 'jax',
            '--copy-metadata', 'jaxlib',
            '--copy-metadata', 'brax',
            '--copy-metadata', 'flax',
            '--copy-metadata', 'optax',
            '--copy-metadata', 'chex',
            '--copy-metadata', 'orbax-checkpoint',
            '--exclude-module', 'pytest',
            '--exclude-module', 'tkinter',
            '--exclude-module', 'IPython',
            '--exclude-module', 'notebook',
            '--exclude-module', 'PyQt5',
            '--exclude-module', 'PyQt6',
            '--exclude-module', 'PySide2',
            '--exclude-module', 'PySide6',
            '--exclude-module', 'torchvision',
            '--exclude-module', 'torchaudio',
            (Join-Path $root 'backend\run_server.py')
        )
        foreach ($pkg in @('jax_cuda12_plugin', 'jax_cuda12_pjrt', 'jax_cuda13_plugin', 'jax_cuda13_pjrt')) {
            & $buildPy -c "import importlib.util,sys; sys.exit(0 if importlib.util.find_spec('$pkg') else 1)"
            if ($LASTEXITCODE -eq 0) { $pyiArgs += @('--collect-all', $pkg) }
        }
        & $buildPy @pyiArgs
        if ($LASTEXITCODE -ne 0) { throw "PyInstaller failed." }
        if (-not (Test-Path (Join-Path $frozenBackend 'backend.exe'))) {
            throw "PyInstaller did not produce backend.exe."
        }

        Section "Smoke-test frozen backend"
        # 1) --selfcheck: imports the lazily-loaded heavy deps (SB3 -> matplotlib
        #    + pandas, optuna, pybullet). Catches over-aggressive excludes that a
        #    /health check would miss (SB3 is imported lazily at train/tune time).
        $beExe = Join-Path $frozenBackend 'backend.exe'
        $env:EASYRTG_REQUIRE_CUDA = '1'
        if ($RequireJaxGpu) { $env:EASYRTG_REQUIRE_JAX_GPU = '1' }
        try {
            $scOut = & $beExe --selfcheck 2>&1 | Out-String
            $scCode = $LASTEXITCODE
        }
        finally {
            Remove-Item Env:\EASYRTG_REQUIRE_CUDA -ErrorAction SilentlyContinue
            Remove-Item Env:\EASYRTG_REQUIRE_JAX_GPU -ErrorAction SilentlyContinue
        }
        if ($scCode -ne 0) {
            Write-Host $scOut
            throw "Frozen backend --selfcheck failed (CUDA or another needed module is unavailable)."
        }
        Write-Host "  selfcheck: $($scOut.Trim())" -ForegroundColor Green

        # 2) /health: the server actually boots and serves.
        $env:EASYRTG_PYBULLET_GUI = '0'
        $env:EASYRTG_PORT = '8079'
        $bp = Start-Process -FilePath $beExe -PassThru -NoNewWindow
        $beOk = $false
        foreach ($i in 1..60) {
            Start-Sleep -Milliseconds 500
            if ($bp.HasExited) { break }
            try { if ((Invoke-RestMethod 'http://127.0.0.1:8079/health' -TimeoutSec 2).ok) { $beOk = $true; break } } catch {}
        }
        if (-not $bp.HasExited) { Stop-Process -Id $bp.Id -Force }
        Remove-Item Env:\EASYRTG_PORT, Env:\EASYRTG_PYBULLET_GUI -ErrorAction SilentlyContinue
        if (-not $beOk) { throw "Frozen backend failed /health." }
        Write-Host "Frozen backend healthy (lazy train/tune imports verified)." -ForegroundColor Green
    }

    # ---- Build Flutter --------------------------------------------------------
    if (-not $SkipFlutter) {
        Section "Build Flutter (windows release)"
        Push-Location $flutterApp
        try {
            & flutter build windows --release
            if ($LASTEXITCODE -ne 0) { throw "flutter build failed." }
        }
        finally { Pop-Location }
    }

    # ---- Assemble -------------------------------------------------------------
    Section "Assemble $AppName"
    if (-not (Test-Path $flutterRelease)) {
        throw "Flutter release not found: $flutterRelease (run without -SkipFlutter)."
    }
    if (-not (Test-Path (Join-Path $frozenBackend 'backend.exe'))) {
        throw "Frozen backend not found (run without -SkipBackend)."
    }
    Remove-Item -Recurse -Force $appDir -ErrorAction SilentlyContinue
    New-Item -ItemType Directory -Force -Path $appDir | Out-Null

    # Flutter runner (.exe + DLLs + data), then rename the runner to <AppName>.exe
    # so it's the single, obvious exe to launch (Flutter locates data\/DLLs by
    # the exe's directory, not its name, so renaming is safe).
    Copy-Item (Join-Path $flutterRelease '*') $appDir -Recurse -Force
    $runner = Get-ChildItem $appDir -Filter *.exe | Select-Object -First 1
    if (-not $runner) { throw "No Flutter runner exe in $flutterRelease." }
    $launch = Join-Path $appDir "$AppName.exe"
    if ($runner.FullName -ne $launch) { Rename-Item $runner.FullName "$AppName.exe" }

    # Frozen backend hidden in runtime\ (backend_launcher finds runtime\backend.exe).
    $appRuntime = Join-Path $appDir 'runtime'
    New-Item -ItemType Directory -Force -Path $appRuntime | Out-Null
    Copy-Item (Join-Path $frozenBackend '*') $appRuntime -Recurse -Force
    # The urdf_files\ tree is ~1.8 GB of sample robots. The app loads URDFs by
    # path at runtime and ships pybullet_data's samples (r2d2, plane) inside the
    # backend, so don't bundle it unless asked. This is the big size win.
    if ($IncludeSamples -and (Test-Path (Join-Path $root 'urdf_files'))) {
        Copy-Item (Join-Path $root 'urdf_files') (Join-Path $appDir 'urdf_files') -Recurse -Force
    }

    # No zip — folder only (zip it yourself).
    $sizeGB = [math]::Round((Get-ChildItem $appDir -Recurse -File | Measure-Object Length -Sum).Sum / 1GB, 2)
    Write-Host "`nDone." -ForegroundColor Green
    Write-Host "  App folder: $appDir ($sizeGB GB)" -ForegroundColor Green
    Write-Host "  Run:        $AppName.exe" -ForegroundColor Green
}
finally {
    Pop-Location
    Set-Location $scriptDir
}
