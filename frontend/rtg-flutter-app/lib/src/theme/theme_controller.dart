import 'package:flutter/material.dart';
import 'package:flutter_riverpod/legacy.dart';
import 'package:shared_preferences/shared_preferences.dart';

import 'app_theme.dart';

final themeControllerProvider = ChangeNotifierProvider<ThemeController>((ref) {
  return ThemeController()..load();
});

class ThemeController extends ChangeNotifier {
  static const _modeKey = 'theme_mode';
  static const _accentKey = 'theme_accent';

  // Light-first: the app opens in light mode unless the user has chosen
  // otherwise (dark remains fully supported and persisted).
  ThemeMode mode = ThemeMode.light;
  int accentIndex = 0;

  AccentPreset get accent =>
      accentPresets[accentIndex.clamp(0, accentPresets.length - 1)];

  Future<void> load() async {
    final prefs = await SharedPreferences.getInstance();
    final storedMode = prefs.getString(_modeKey);
    mode = switch (storedMode) {
      'dark' => ThemeMode.dark,
      'system' => ThemeMode.system,
      _ => ThemeMode.light,
    };
    accentIndex = (prefs.getInt(_accentKey) ?? 0).clamp(
      0,
      accentPresets.length - 1,
    );
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

  Future<void> setAccent(int index) async {
    if (accentIndex == index) return;
    accentIndex = index.clamp(0, accentPresets.length - 1);
    notifyListeners();
    final prefs = await SharedPreferences.getInstance();
    await prefs.setInt(_accentKey, accentIndex);
  }
}
