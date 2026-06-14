import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';

import 'easy_colors.dart';

/// An accent color preset selectable from Settings.
class AccentPreset {
  const AccentPreset(this.name, this.seed);

  final String name;
  final Color seed;
}

const accentPresets = <AccentPreset>[
  AccentPreset('Teal', Color(0xff4fb7a8)),
  AccentPreset('Ocean', Color(0xff4f9cff)),
  AccentPreset('Violet', Color(0xff9a7bff)),
  AccentPreset('Magenta', Color(0xffe060b8)),
  AccentPreset('Crimson', Color(0xffef5466)),
  AccentPreset('Amber', Color(0xffe8a33d)),
  AccentPreset('Lime', Color(0xff8fc742)),
  AccentPreset('Slate', Color(0xff8a9bb0)),
];

ThemeData buildAppTheme({required Brightness brightness, required Color seed}) {
  final dark = brightness == Brightness.dark;
  final scheme = ColorScheme.fromSeed(seedColor: seed, brightness: brightness);

  final surface = dark ? const Color(0xff15191f) : const Color(0xfff4f6f9);
  final card = dark ? const Color(0xff1d232b) : Colors.white;
  final outline = dark ? const Color(0xff2c343f) : const Color(0xffdde3ea);

  final textTheme = GoogleFonts.interTextTheme(
    dark ? ThemeData.dark().textTheme : ThemeData.light().textTheme,
  );

  return ThemeData(
    brightness: brightness,
    colorScheme: scheme.copyWith(
      surface: surface,
      surfaceContainerLow: card,
      outlineVariant: outline,
    ),
    scaffoldBackgroundColor: surface,
    textTheme: textTheme,
    splashFactory: InkSparkle.splashFactory,
    visualDensity: VisualDensity.comfortable,
    cardTheme: CardThemeData(
      color: card,
      elevation: 0,
      margin: EdgeInsets.zero,
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(12),
        side: BorderSide(color: outline),
      ),
    ),
    dividerTheme: DividerThemeData(color: outline, space: 1, thickness: 1),
    inputDecorationTheme: InputDecorationTheme(
      isDense: true,
      filled: true,
      fillColor: dark ? const Color(0xff161b22) : const Color(0xffeef1f5),
      border: OutlineInputBorder(
        borderRadius: BorderRadius.circular(8),
        borderSide: BorderSide(color: outline),
      ),
      enabledBorder: OutlineInputBorder(
        borderRadius: BorderRadius.circular(8),
        borderSide: BorderSide(color: outline),
      ),
      focusedBorder: OutlineInputBorder(
        borderRadius: BorderRadius.circular(8),
        borderSide: BorderSide(color: scheme.primary, width: 1.6),
      ),
    ),
    filledButtonTheme: FilledButtonThemeData(
      style: FilledButton.styleFrom(
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(8)),
      ),
    ),
    outlinedButtonTheme: OutlinedButtonThemeData(
      style: OutlinedButton.styleFrom(
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(8)),
        side: BorderSide(color: outline),
      ),
    ),
    chipTheme: ChipThemeData(
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(8),
        side: BorderSide(color: outline),
      ),
    ),
    navigationRailTheme: NavigationRailThemeData(
      backgroundColor: dark ? const Color(0xff10141a) : Colors.white,
      indicatorColor: scheme.primary.withValues(alpha: 0.18),
      selectedIconTheme: IconThemeData(color: scheme.primary),
      selectedLabelTextStyle: TextStyle(
        color: scheme.primary,
        fontSize: 11.5,
        fontWeight: FontWeight.w600,
      ),
      unselectedLabelTextStyle: TextStyle(
        color: dark ? const Color(0xff8b97a5) : const Color(0xff5d6976),
        fontSize: 11.5,
      ),
    ),
    tooltipTheme: TooltipThemeData(
      waitDuration: const Duration(milliseconds: 400),
      decoration: BoxDecoration(
        color: dark ? const Color(0xff2c343f) : const Color(0xff3a4350),
        borderRadius: BorderRadius.circular(6),
      ),
      textStyle: const TextStyle(color: Colors.white, fontSize: 12),
    ),
    snackBarTheme: SnackBarThemeData(
      behavior: SnackBarBehavior.floating,
      width: 380,
      backgroundColor: dark ? const Color(0xff2c343f) : const Color(0xff323b46),
      contentTextStyle: const TextStyle(color: Colors.white),
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(10)),
    ),
    dataTableTheme: DataTableThemeData(
      headingRowHeight: 36,
      dataRowMinHeight: 32,
      dataRowMaxHeight: 40,
      headingTextStyle: textTheme.labelMedium?.copyWith(
        fontWeight: FontWeight.w700,
        color: dark ? const Color(0xff9aa6b2) : const Color(0xff5d6976),
      ),
      dividerThickness: 0.6,
    ),
    listTileTheme: const ListTileThemeData(
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.all(Radius.circular(8)),
      ),
    ),
    sliderTheme: const SliderThemeData(
      showValueIndicator: ShowValueIndicator.onDrag,
    ),
    extensions: <ThemeExtension<dynamic>>[EasyColors.of(brightness)],
  );
}

/// Monospace style for code, paths, JSON and numeric data.
TextStyle monoStyle(BuildContext context, {double? fontSize, Color? color}) {
  return GoogleFonts.jetBrainsMono(
    fontSize: fontSize ?? 12.5,
    color: color,
    height: 1.45,
  );
}
