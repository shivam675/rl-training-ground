#!/usr/bin/env pwsh
# Dev convenience (Windows). Mirrors scripts/dev_run.sh: clear leftovers from a
# previous session, start the backend, wait for it, run the desktop app, then
# stop the backend (and its uvicorn child) when the app exits.
$ErrorActionPreference = 'Stop'
$scripts = $PSScriptRoot

# Defines Stop-Residue: a backend still holding port 8000, and any orphaned
# Isaac/Kit tree. See that script for why each one silently breaks the next run.
. (Join-Path $scripts 'stop_residue.ps1')
Stop-Residue | Out-Null

# Direct mode so the spawned process tree is a clean parent we can kill.
$env:EASYRTG_SUPERVISE = '0'
$backend = Start-Process -FilePath 'powershell' -PassThru -ArgumentList @(
    '-NoProfile', '-ExecutionPolicy', 'Bypass',
    '-File', (Join-Path $scripts 'start_backend.ps1')
)
try {
    Start-Sleep -Seconds 2
    & (Join-Path $scripts 'start_frontend.ps1')
}
finally {
    if ($backend -and -not $backend.HasExited) {
        # /T kills the uvicorn child too, /F forces it.
        taskkill /PID $backend.Id /T /F | Out-Null
    }
    # Killing the backend orphans any Isaac subprocess it was tailing: the
    # worker's own taskkill never runs. Reap here so the GPU and the WebRTC
    # ports are free the moment the app closes, not on the next launch.
    Stop-Residue | Out-Null
}
