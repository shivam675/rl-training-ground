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

  String? _findScript() {
    final override = Platform.environment['EASYRTG_BACKEND'];
    if (override != null && File(override).existsSync()) return override;
    // Walk up from the working directory looking for the repo's script —
    // covers `flutter run` from the app dir and the built bundle in-place.
    var dir = Directory.current;
    for (var depth = 0; depth < 6; depth++) {
      final candidate = File('${dir.path}/scripts/start_backend.sh');
      if (candidate.existsSync()) return candidate.path;
      final parent = dir.parent;
      if (parent.path == dir.path) break;
      dir = parent;
    }
    return null;
  }

  /// Returns true once the backend is reachable (whether or not we spawned it).
  Future<bool> ensureRunning() async {
    if (await _healthy()) return true;
    final script = _findScript();
    if (script == null) {
      debugPrint(
        'BackendLauncher: backend offline and start_backend.sh not found '
        '(set EASYRTG_BACKEND to override).',
      );
      return false;
    }
    try {
      // EASYRTG_SUPERVISE=0 → the script execs uvicorn directly, so killing
      // this process actually stops the server (no orphaned child).
      _process = await Process.start(
        'bash',
        [script],
        environment: {'EASYRTG_SUPERVISE': '0'},
      );
      _process!.stdout.drain<void>();
      _process!.stderr.drain<void>();
    } catch (e) {
      debugPrint('BackendLauncher: failed to spawn backend: $e');
      return false;
    }
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
