import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';

import 'easy_colors.dart';
import 'tokens.dart';

/// The app theme. Design decisions and the token table live in
/// `docs/superpowers/plans/2026-07-30-ui-refactor.md`.
///
/// Two properties of this theme are load-bearing rather than cosmetic:
///
/// **Ink-only accent.** [ColorScheme.primary] is a near-black, not a hue. That
/// makes the four [EasyColors] semantics — success / warning / danger / info —
/// the only saturated colour anywhere in the app, so a reward-audit verdict is
/// the thing your eye lands on. Do not reintroduce a tinted accent for
/// decoration; if something needs emphasis, use weight or ink.
///
/// **Shadow-only surfaces.** Cards carry no outline. Depth separates a surface
/// from the canvas. See [Shadows].

/// Neutral ramp. Kept private: everything outside this file should read these
/// through [ColorScheme] so light and dark resolve automatically.
class _Neutrals {
  const _Neutrals._();

  static const canvas = Color(0xffF4F5F7);
  static const surface = Color(0xffFFFFFF);
  static const sunken = Color(0xffF1F2F4);
  static const line = Color(0xffE7E9EC);
  static const line2 = Color(0xffDCE0E4);
  static const ink = Color(0xff15181C);
  static const txt = Color(0xff2B3138);
  static const txt2 = Color(0xff69717B);
  static const txt3 = Color(0xff9AA1AA);

  // Dark counterparts. Deliberately unpolished this pass (see D8 in the plan) —
  // shadow-only surfaces barely read on dark, so a proper dark theme needs
  // borders back and is its own design job. These keep dark *usable*, not
  // premium: the accent inverts to near-white so filled buttons stay visible.
  static const dCanvas = Color(0xff15191f);
  static const dSurface = Color(0xff1d232b);
  static const dSunken = Color(0xff161b22);
  static const dLine = Color(0xff2c343f);
  static const dInk = Color(0xffF2F4F6);
  static const dTxt = Color(0xffE4E8EC);
  static const dTxt2 = Color(0xff9AA6B2);
  static const dTxt3 = Color(0xff6F7883);
}

ThemeData buildAppTheme({required Brightness brightness}) {
  final dark = brightness == Brightness.dark;

  final accent = dark ? _Neutrals.dInk : _Neutrals.ink;
  final onAccent = dark ? _Neutrals.ink : Colors.white;
  final canvas = dark ? _Neutrals.dCanvas : _Neutrals.canvas;
  final surface = dark ? _Neutrals.dSurface : _Neutrals.surface;
  final sunken = dark ? _Neutrals.dSunken : _Neutrals.sunken;
  final line = dark ? _Neutrals.dLine : _Neutrals.line;
  final onSurface = dark ? _Neutrals.dTxt : _Neutrals.txt;
  final muted = dark ? _Neutrals.dTxt2 : _Neutrals.txt2;

  // fromSeed fills every M3 role we don't care about; the copyWith pins the
  // ones we do. Seeding from ink yields a neutral palette, so the roles left
  // unpinned stay grey rather than drifting to some hue.
  final scheme =
      ColorScheme.fromSeed(
        seedColor: _Neutrals.ink,
        brightness: brightness,
      ).copyWith(
        primary: accent,
        onPrimary: onAccent,
        surface: canvas,
        onSurface: onSurface,
        surfaceContainerLow: surface,
        surfaceContainerHighest: sunken,
        outlineVariant: line,
        outline: dark ? _Neutrals.dTxt3 : _Neutrals.line2,
      );

  final textTheme = GoogleFonts.geistTextTheme(
    dark ? ThemeData.dark().textTheme : ThemeData.light().textTheme,
  ).apply(bodyColor: onSurface, displayColor: onSurface);

  return ThemeData(
    brightness: brightness,
    colorScheme: scheme,
    scaffoldBackgroundColor: canvas,
    textTheme: textTheme,
    splashFactory: InkSparkle.splashFactory,
    visualDensity: VisualDensity.comfortable,
    cardTheme: CardThemeData(
      color: surface,
      // Shadow-only: the border is gone, so the elevation is what makes a card
      // visible against the canvas. Dropping one without the other leaves an
      // invisible white-on-white surface.
      elevation: 1,
      shadowColor: Shadows.materialCast,
      surfaceTintColor: Colors.transparent,
      margin: EdgeInsets.zero,
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(Radii.lg),
      ),
    ),
    dividerTheme: DividerThemeData(color: line, space: 1, thickness: 1),
    inputDecorationTheme: InputDecorationTheme(
      isDense: true,
      filled: true,
      fillColor: sunken,
      border: _inputBorder(Colors.transparent),
      enabledBorder: _inputBorder(Colors.transparent),
      focusedBorder: _inputBorder(accent, width: 1.6),
      hintStyle: TextStyle(color: dark ? _Neutrals.dTxt3 : _Neutrals.txt3),
    ),
    filledButtonTheme: FilledButtonThemeData(
      style: FilledButton.styleFrom(
        backgroundColor: accent,
        foregroundColor: onAccent,
        shape: const StadiumBorder(),
        padding: const EdgeInsets.symmetric(horizontal: Insets.lg),
      ),
    ),
    outlinedButtonTheme: OutlinedButtonThemeData(
      style: OutlinedButton.styleFrom(
        foregroundColor: onSurface,
        backgroundColor: surface,
        shape: const StadiumBorder(),
        side: BorderSide(color: dark ? _Neutrals.dLine : _Neutrals.line2),
        padding: const EdgeInsets.symmetric(horizontal: Insets.lg),
      ),
    ),
    textButtonTheme: TextButtonThemeData(
      style: TextButton.styleFrom(foregroundColor: accent),
    ),
    chipTheme: ChipThemeData(
      backgroundColor: surface,
      side: BorderSide(color: dark ? _Neutrals.dLine : _Neutrals.line2),
      shape: const StadiumBorder(),
    ),
    tooltipTheme: TooltipThemeData(
      waitDuration: const Duration(milliseconds: 400),
      decoration: BoxDecoration(
        color: dark ? _Neutrals.dLine : _Neutrals.ink,
        borderRadius: BorderRadius.circular(Radii.sm),
      ),
      textStyle: TextStyle(color: dark ? _Neutrals.dTxt : Colors.white, fontSize: 12),
    ),
    snackBarTheme: SnackBarThemeData(
      behavior: SnackBarBehavior.floating,
      width: 380,
      backgroundColor: dark ? _Neutrals.dLine : _Neutrals.ink,
      contentTextStyle: TextStyle(color: dark ? _Neutrals.dTxt : Colors.white),
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(Radii.md),
      ),
    ),
    dataTableTheme: DataTableThemeData(
      headingRowHeight: 36,
      dataRowMinHeight: 32,
      dataRowMaxHeight: 40,
      headingTextStyle: textTheme.labelMedium?.copyWith(
        fontWeight: FontWeight.w700,
        color: muted,
      ),
      dividerThickness: 0.6,
    ),
    listTileTheme: const ListTileThemeData(
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.all(Radius.circular(Radii.sm)),
      ),
    ),
    sliderTheme: const SliderThemeData(
      showValueIndicator: ShowValueIndicator.onDrag,
    ),
    extensions: <ThemeExtension<dynamic>>[EasyColors.of(brightness)],
  );
}

OutlineInputBorder _inputBorder(Color color, {double width = 1}) {
  return OutlineInputBorder(
    borderRadius: BorderRadius.circular(Radii.sm),
    borderSide: BorderSide(color: color, width: width),
  );
}

/// Monospace style for code, paths, JSON and numeric data.
TextStyle monoStyle(BuildContext context, {double? fontSize, Color? color}) {
  return GoogleFonts.geistMono(
    fontSize: fontSize ?? 12.5,
    color: color,
    height: 1.45,
  );
}
