#!/usr/bin/env pwsh
# Run the Flutter desktop app on Windows. Mirrors scripts/start_frontend.sh
# (no CC/CXX overrides needed: Flutter uses MSVC on Windows).
# try/finally returns to the scripts dir on any exit (incl. Ctrl+C) so the
# script can be launched again without cd-ing back.
$scriptDir = $PSScriptRoot
try {
    Set-Location (Join-Path $PSScriptRoot '..\frontend\rtg-flutter-app')
    flutter run -d windows
}
finally {
    Set-Location $scriptDir
}
