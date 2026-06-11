#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../frontend/rtg-flutter-app"
export CC="${CC:-gcc}"
export CXX="${CXX:-g++}"
exec flutter run -d linux
