/// Design tokens — a single spacing and radius scale so density and rounding
/// stay consistent as new surfaces are built. Prefer these over ad-hoc numeric
/// literals in new code.
library;

/// Spacing scale (logical pixels).
class Insets {
  const Insets._();

  static const double xs = 4;
  static const double sm = 8;
  static const double md = 12;
  static const double lg = 16;
  static const double xl = 24;
  static const double xxl = 32;
}

/// Corner-radius scale.
class Radii {
  const Radii._();

  static const double sm = 8;
  static const double md = 10;
  static const double lg = 12;
  static const double pill = 999;
}
