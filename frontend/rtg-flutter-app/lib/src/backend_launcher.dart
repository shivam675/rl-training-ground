import 'dart:async';
import 'dart:convert';
import 'dart:io';

import 'package:flutter/foundation.dart';
import 'package:http/http.dart' as http;

/// Spawns and owns the backend when it isn't already running, so the user
/// never has to start uvicorn manually. If the backend was started by someone
/// else we leave it alone (and never kill it on exit).
class BackendLauncher {
  Process? _process;

  bool get spawnedByUs => _process != null;

  Future<bool> _healthy() async {
    try {
      final res = await http
          .get(Uri.parse('http://127.0.0.1:8000/health'))
          .timeout(const Duration(seconds: 2));
      return res.statusCode == 200 &&
          (jsonDecode(res.body) as Map)['ok'] == true;
    } catch (_) {
      return false;
    }
  }

  /// Repo root: walk up from the working directory until we find both the
  /// backend package and the scripts dir. Covers `flutter run` from the app
  /// dir and a built bundle run in-place.
  String? _findRepoRoot() {
    var dir = Directory.current;
    for (var depth = 0; depth < 8; depth++) {
      if (Directory('${dir.path}/backend').existsSync() &&
          Directory('${dir.path}/scripts').existsSync()) {
        return dir.path;
      }
      final parent = dir.parent;
      if (parent.path == dir.path) break;
      dir = parent;
    }
    return null;
  }

  String _windowsPython(String root) {
    for (final rel in [r'.venv\Scripts\python.exe', r'.venv-rtg\Scripts\python.exe']) {
      final candidate = File('$root\\$rel');
      if (candidate.existsSync()) return candidate.path;
    }
    return 'python';
  }

  /// Start the backend for the current platform. Returns null when it can't be
  /// located — the caller then logs and gives up; the app still runs and will
  /// connect to a manually-started backend. EASYRTG_SUPERVISE=0 makes the
  /// child exec uvicorn directly, so killing it actually stops the server.
  Future<Process?> _spawnBackend() async {
    final env = {'EASYRTG_SUPERVISE': '0'};

    // Explicit override always wins: a .ps1, a .sh, or any executable.
    final override = Platform.environment['EASYRTG_BACKEND'];
    if (override != null && File(override).existsSync()) {
      final lower = override.toLowerCase();
      if (Platform.isWindows && lower.endsWith('.ps1')) {
        return Process.start(
          'powershell',
          ['-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', override],
          environment: env,
        );
      }
      if (lower.endsWith('.sh')) {
        return Process.start('bash', [override], environment: env);
      }
      return Process.start(override, const [], environment: env);
    }

    final root = _findRepoRoot();
    if (root == null) return null;

    if (Platform.isWindows) {
      // Run uvicorn directly: no bash, no PowerShell execution-policy hurdles.
      return Process.start(
        _windowsPython(root),
        ['-m', 'uvicorn', 'backend.main:app', '--host', '127.0.0.1', '--port', '8000'],
        workingDirectory: root,
        environment: env,
      );
    }

    // POSIX: the supervised script in direct mode.
    final script = '$root/scripts/start_backend.sh';
    if (!File(script).existsSync()) return null;
    return Process.start('bash', [script], environment: env);
  }

  /// Returns true once the backend is reachable (whether or not we spawned it).
  Future<bool> ensureRunning() async {
    if (await _healthy()) return true;
    try {
      _process = await _spawnBackend();
    } catch (e) {
      debugPrint('BackendLauncher: failed to spawn backend: $e');
      return false;
    }
    if (_process == null) {
      debugPrint(
        'BackendLauncher: backend offline and could not be located '
        '(set EASYRTG_BACKEND to a start script, or start it manually).',
      );
      return false;
    }
    _process!.stdout.drain<void>();
    _process!.stderr.drain<void>();
    for (var i = 0; i < 60; i++) {
      await Future<void>.delayed(const Duration(milliseconds: 500));
      if (await _healthy()) {
        debugPrint('BackendLauncher: backend up (spawned).');
        return true;
      }
    }
    debugPrint('BackendLauncher: backend did not become healthy in 30s.');
    return false;
  }

  /// Kill the backend only if this app started it.
  void shutdown() {
    final process = _process;
    if (process == null) return;
    debugPrint('BackendLauncher: stopping spawned backend.');
    process.kill(ProcessSignal.sigterm);
    _process = null;
  }
}
