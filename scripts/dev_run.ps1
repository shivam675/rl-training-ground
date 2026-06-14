#!/usr/bin/env pwsh
# Dev convenience (Windows). Mirrors scripts/dev_run.sh: start the backend,
# wait for it, run the desktop app, then stop the backend (and its uvicorn
# child) when the app exits.
$ErrorActionPreference = 'Stop'
$scripts = $PSScriptRoot

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
}
