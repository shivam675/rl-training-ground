import 'package:flutter/material.dart';

/// Semantic status colors layered on top of the Material [ColorScheme].
///
/// These four hues (success / warning / danger / info) recur all over the app —
/// status dots, banners, validation results, charts. Centralizing them here
/// (instead of the hardcoded hex that was scattered across panels) means one
/// source of truth that adapts to brightness: the dark variants are bright and
/// luminous, the light variants are darker and more saturated so colored text
/// stays legible on light surfaces.
///
/// Access via the [BuildContext] extension: `context.colors.success`.
@immutable
class EasyColors extends ThemeExtension<EasyColors> {
  const EasyColors({
    required this.success,
    required this.warning,
    required this.danger,
    required this.info,
  });

  /// Positive / done / online.
  final Color success;

  /// Caution / non-fatal problem / locked.
  final Color warning;

  /// Failure / error / offline.
  final Color danger;

  /// Neutral accent for informational charts and chips.
  final Color info;

  /// Tuned for dark surfaces (luminous).
  static const dark = EasyColors(
    success: Color(0xff5fe089),
    warning: Color(0xffffc857),
    danger: Color(0xffff6f64),
    info: Color(0xff4f9cff),
  );

  /// Tuned for light surfaces (darker / more saturated for text contrast).
  static const light = EasyColors(
    success: Color(0xff1f9d57),
    warning: Color(0xffb7791f),
    danger: Color(0xffd23b30),
    info: Color(0xff1f6feb),
  );

  static EasyColors of(Brightness brightness) =>
      brightness == Brightness.dark ? dark : light;

  @override
  EasyColors copyWith({
    Color? success,
    Color? warning,
    Color? danger,
    Color? info,
  }) {
    return EasyColors(
      success: success ?? this.success,
      warning: warning ?? this.warning,
      danger: danger ?? this.danger,
      info: info ?? this.info,
    );
  }

  @override
  EasyColors lerp(ThemeExtension<EasyColors>? other, double t) {
    if (other is! EasyColors) return this;
    return EasyColors(
      success: Color.lerp(success, other.success, t)!,
      warning: Color.lerp(warning, other.warning, t)!,
      danger: Color.lerp(danger, other.danger, t)!,
      info: Color.lerp(info, other.info, t)!,
    );
  }
}

/// Ergonomic access to the semantic palette: `context.colors.warning`.
/// Falls back to the dark palette if the extension is somehow absent.
extension EasyColorsContext on BuildContext {
  EasyColors get colors =>
      Theme.of(this).extension<EasyColors>() ?? EasyColors.dark;
}
