#!/usr/bin/env pwsh
<#
.SYNOPSIS
Kill leftovers from a previous EasyRTG session. Run it standalone, or let
dev_run.ps1 dot-source it (it calls Stop-Residue before launching and again on
exit). Pass -WhatIfOnly to list what would be killed without killing anything.

Two kinds of leftover survive a crash or a Ctrl+C, and both bite silently:

* A backend still listening on 8000. BackendLauncher attaches to whatever answers
  /health and never kills it, so a stale server keeps serving OLD CODE against a
  new checkout - you edit, relaunch, and see no change. (The trap CLAUDE.md warns
  about under backend_launcher.dart.)
* An orphaned Isaac Sim / Kit tree. IsaacTrainingWorker only reaps it while the
  backend is alive to notice the stop; kill the backend first and Kit lives on
  holding the GPU and the WebRTC ports (49100/47998), so the next run starts
  degraded or not at all.

ASCII only, deliberately: Windows PowerShell 5.1 reads a BOM-less UTF-8 .ps1 as
ANSI, so a UTF-8 em dash arrives as a smart right quote - which the tokenizer
treats as a string delimiter and the file stops parsing.
#>
param([switch]$WhatIfOnly)

function Stop-Residue {
    param([switch]$WhatIfOnly)

    $targets = @()

    Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue |
        Select-Object -ExpandProperty OwningProcess -Unique |
        ForEach-Object {
            $targets += [pscustomobject]@{ Target = $_; What = 'backend on port 8000' }
        }

    # Matched by executable path under the Isaac Lab install, not by process name:
    # Kit runs as python.exe out of _isaac_sim\kit\python, and killing python.exe
    # by name would take the machine's other interpreters with it. The second arm
    # catches the "cmd /c isaaclab.bat -p" wrapper, whose own exe is system
    # cmd.exe. It matches the bat path rather than the install directory, so a dev
    # shell that merely mentions Isaac is not treated as an Isaac process.
    $bat = if ($env:EASYRTG_ISAACLAB_BAT) { $env:EASYRTG_ISAACLAB_BAT } else { 'C:\IsaacLab\isaaclab.bat' }
    $prefix = (Split-Path $bat -Parent).TrimEnd('\') + '\'
    Get-CimInstance Win32_Process -ErrorAction SilentlyContinue |
        Where-Object {
            ($_.ExecutablePath -and $_.ExecutablePath.StartsWith($prefix, 'OrdinalIgnoreCase')) -or
            ($_.CommandLine -and $_.CommandLine -like "*$bat*")
        } |
        ForEach-Object {
            $targets += [pscustomobject]@{ Target = $_.ProcessId; What = "Isaac $($_.Name)" }
        }

    if (-not $targets) {
        Write-Host 'stop_residue: nothing to clean.'
        return 0
    }
    foreach ($t in $targets) {
        if ($WhatIfOnly) {
            Write-Host ("stop_residue: would kill PID {0} - {1}" -f $t.Target, $t.What)
            continue
        }
        Write-Host ("stop_residue: killing PID {0} - {1}" -f $t.Target, $t.What)
        # /T takes the child tree (uvicorn under powershell, Kit under cmd).
        taskkill /PID $t.Target /T /F 2>&1 | Out-Null
    }
    return $targets.Count
}

# Only act when invoked directly; dot-sourcing just defines the function.
if ($MyInvocation.InvocationName -ne '.') {
    Stop-Residue -WhatIfOnly:$WhatIfOnly | Out-Null
}
