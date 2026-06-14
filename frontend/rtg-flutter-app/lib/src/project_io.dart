import 'dart:convert';
import 'dart:io';

import 'package:file_picker/file_picker.dart';
import 'package:flutter/material.dart';

import 'app_state.dart';

/// Desktop project file handling for the `.rtg` format. The dialogs live here
/// (they need a [BuildContext]); the actual load/save round-trips go through
/// [AppState] so the backend stays the single source of truth for the config.
class ProjectIO {
  static const String extension = 'rtg';

  /// Returns true when a fresh project was actually started (not cancelled).
  static Future<bool> newProject(BuildContext context, AppState state) async {
    final confirmed = await _confirm(
      context,
      title: 'Start a new project?',
      body:
          'This unloads the current robot and clears the observation, action '
          'and reward setup. Save first if you want to keep it.',
      confirmLabel: 'New project',
    );
    if (!confirmed) return false;
    await state.newProject();
    return true;
  }

  /// Returns true when a project file was picked and applied.
  static Future<bool> open(BuildContext context, AppState state) async {
    final result = await FilePicker.platform.pickFiles(
      type: FileType.custom,
      allowedExtensions: [extension],
      dialogTitle: 'Open EasyRTG project',
    );
    final path = result?.files.single.path;
    if (path == null) return false;
    Map<String, dynamic> doc;
    try {
      final text = await File(path).readAsString();
      final decoded = jsonDecode(text);
      if (decoded is! Map) throw const FormatException('not a JSON object');
      doc = decoded.cast<String, dynamic>();
    } catch (e) {
      if (context.mounted) {
        _toast(context, 'Could not read project file: $e');
      }
      return false;
    }
    final rawConfig = doc['config'] ?? doc;
    if (rawConfig is! Map) {
      if (context.mounted) _toast(context, 'Project file has no config.');
      return false;
    }
    await state.openProjectConfig(rawConfig.cast<String, dynamic>(), path);
    return true;
  }

  static Future<void> save(BuildContext context, AppState state) async {
    final existing = state.currentProjectPath;
    if (existing == null) {
      await saveAs(context, state);
      return;
    }
    await state.saveProjectToPath(existing);
  }

  static Future<void> saveAs(BuildContext context, AppState state) async {
    final path = await FilePicker.platform.saveFile(
      dialogTitle: 'Save EasyRTG project',
      fileName: state.currentProjectName ?? 'project.$extension',
      type: FileType.custom,
      allowedExtensions: [extension],
    );
    if (path == null) return;
    final full = path.endsWith('.$extension') ? path : '$path.$extension';
    await state.saveProjectToPath(full);
  }

  static void _toast(BuildContext context, String message) {
    ScaffoldMessenger.of(context)
      ..clearSnackBars()
      ..showSnackBar(SnackBar(content: Text(message)));
  }

  static Future<bool> _confirm(
    BuildContext context, {
    required String title,
    required String body,
    required String confirmLabel,
  }) async {
    final result = await showDialog<bool>(
      context: context,
      builder: (context) => AlertDialog(
        title: Text(title),
        content: Text(body),
        actions: [
          TextButton(
            onPressed: () => Navigator.of(context).pop(false),
            child: const Text('Cancel'),
          ),
          FilledButton(
            onPressed: () => Navigator.of(context).pop(true),
            child: Text(confirmLabel),
          ),
        ],
      ),
    );
    return result ?? false;
  }
}
