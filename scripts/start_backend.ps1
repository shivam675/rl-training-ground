#!/usr/bin/env pwsh
# Supervised backend launch (Windows). Mirrors scripts/start_backend.sh:
# auto-restarts on crash with a short backoff, and prefers the repo venv.
# try/finally returns to the scripts dir on any exit (incl. Ctrl+C) so the
# script can be launched again without cd-ing back.

function Resolve-Python {
    if ($env:PYTHON) { return $env:PYTHON }
    $candidates = @(
        (Join-Path $PWD '.venv312\Scripts\python.exe'),
        (Join-Path $PWD '.venv\Scripts\python.exe'),
        (Join-Path $PWD '.venv-rtg\Scripts\python.exe')
    )
    foreach ($c in $candidates) { if (Test-Path $c) { return $c } }
    return 'python'
}

$scriptDir = $PSScriptRoot
$code = 0
try {
    Set-Location (Join-Path $PSScriptRoot '..')
    $python = Resolve-Python

    # EASYRTG_SUPERVISE=0 -> run uvicorn directly (no restart loop). Used by the
    # desktop app's launcher so killing this process kills the server too.
    if ($env:EASYRTG_SUPERVISE -eq '0') {
        # --timeout-graceful-shutdown caps the shutdown wait: the live-viewport
        # WebSocket never closes on its own, so without it the first Ctrl+C
        # hangs forever waiting for connections (needing a second Ctrl+C). With
        # it, one Ctrl+C drains for 2s then force-cancels and exits cleanly.
        & $python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --timeout-graceful-shutdown 2
        $code = $LASTEXITCODE
    }
    else {
        $RestartDelay = 2
        while ($true) {
            & $python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --timeout-graceful-shutdown 2
            $code = $LASTEXITCODE
            # Clean exit / Ctrl-C: stop supervising.
            if ($code -eq 0 -or $null -eq $code) { break }
            Write-Host "[supervisor] backend exited with code $code - restarting in ${RestartDelay}s" -ForegroundColor Yellow
            Start-Sleep -Seconds $RestartDelay
        }
    }
}
finally {
    Set-Location $scriptDir
}
exit $code
