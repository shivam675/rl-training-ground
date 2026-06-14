#!/usr/bin/env pwsh
# Supervised backend launch (Windows). Mirrors scripts/start_backend.sh:
# auto-restarts on crash with a short backoff, and prefers the repo venv.
Set-Location (Join-Path $PSScriptRoot '..')

function Resolve-Python {
    if ($env:PYTHON) { return $env:PYTHON }
    $candidates = @(
        (Join-Path $PWD '.venv\Scripts\python.exe'),
        (Join-Path $PWD '.venv-rtg\Scripts\python.exe')
    )
    foreach ($c in $candidates) { if (Test-Path $c) { return $c } }
    return 'python'
}

$python = Resolve-Python

# EASYRTG_SUPERVISE=0 -> run uvicorn directly (no restart loop). Used by the
# desktop app's launcher so killing this process kills the server too.
if ($env:EASYRTG_SUPERVISE -eq '0') {
    & $python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
    exit $LASTEXITCODE
}

$RestartDelay = 2
while ($true) {
    & $python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
    $code = $LASTEXITCODE
    # Clean exit / Ctrl-C: stop supervising.
    if ($code -eq 0 -or $code -eq $null) { exit $code }
    Write-Host "[supervisor] backend exited with code $code - restarting in ${RestartDelay}s" -ForegroundColor Yellow
    Start-Sleep -Seconds $RestartDelay
}
