import 'package:flutter/material.dart';
import 'package:flutter_riverpod/legacy.dart';
import 'package:shared_preferences/shared_preferences.dart';

final themeControllerProvider = ChangeNotifierProvider<ThemeController>((ref) {
  return ThemeController()..load();
});

/// Brightness only. The accent-preset picker was deleted in the 2026-07-30 UI
/// refactor: the theme is ink-only by design, so a hue chooser had nothing left
/// to tint. The stale `theme_accent` pref of past installs is simply ignored.
class ThemeController extends ChangeNotifier {
  static const _modeKey = 'theme_mode';

  // Light-first: the app opens in light mode unless the user has chosen
  // otherwise (dark remains supported and persisted, if unpolished).
  ThemeMode mode = ThemeMode.light;

  Future<void> load() async {
    final prefs = await SharedPreferences.getInstance();
    mode = switch (prefs.getString(_modeKey)) {
      'dark' => ThemeMode.dark,
      'system' => ThemeMode.system,
      _ => ThemeMode.light,
    };
    notifyListeners();
  }

  Future<void> setMode(ThemeMode value) async {
    if (mode == value) return;
    mode = value;
    notifyListeners();
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString(_modeKey, switch (value) {
      ThemeMode.light => 'light',
      ThemeMode.system => 'system',
      ThemeMode.dark => 'dark',
    });
  }
}
