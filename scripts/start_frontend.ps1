#!/usr/bin/env pwsh
# Run the Flutter desktop app on Windows. Mirrors scripts/start_frontend.sh
# (no CC/CXX overrides needed: Flutter uses MSVC on Windows).
Set-Location (Join-Path $PSScriptRoot '..\frontend\rtg-flutter-app')
flutter run -d windows
