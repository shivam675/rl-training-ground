/// Design tokens — a single spacing, radius and elevation scale so density,
/// rounding and depth stay consistent as new surfaces are built. Prefer these
/// over ad-hoc numeric literals in new code.
///
/// Values come from `docs/superpowers/plans/2026-07-30-ui-refactor.md`. The
/// refactor is shadow-only: surfaces separate from the canvas by depth, not by
/// an outline, so [Shadows] is what a panel reaches for where it used to reach
/// for a border.
library;

import 'dart:ui' show PointMode;

import 'package:flutter/material.dart';

/// Spacing scale (logical pixels).
class Insets {
  const Insets._();

  static const double xs = 4;
  static const double sm = 8;
  static const double md = 12;
  static const double lg = 16;
  static const double xl = 24;
  static const double xxl = 32;

  /// Gutter every scrollable page body uses. Pinned to a name because five
  /// panels had drifted to a bare `EdgeInsets.all(14)` while the sixth used
  /// [lg] — half a step apart, which reads as sloppy rather than dense.
  static const double page = lg;
}

/// The 16px dot grid painted on the page canvas behind the panels. Only the
/// gutters show it: [Panel] paints an opaque surface over the rest.
class DotGrid extends StatelessWidget {
  const DotGrid({super.key, required this.child});

  final Widget child;

  static const double spacing = 16;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return CustomPaint(
      painter: _DotGridPainter(
        // `outline`, not `outlineVariant`. The hairline neutral (#E7E9EC) is
        // three shades off the canvas (#F4F5F7) — at 2px that is invisible, so
        // the grid may as well not be painted at all. This is the control-outline
        // grey, damped on dark where the same value reads as halftone.
        color: theme.brightness == Brightness.dark
            ? theme.colorScheme.outline.withValues(alpha: 0.35)
            : theme.colorScheme.outline,
      ),
      child: child,
    );
  }
}

class _DotGridPainter extends CustomPainter {
  const _DotGridPainter({required this.color});

  final Color color;

  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()..color = color;
    // One drawPoints call, not a dot-per-drawCircle loop: a 1600x900 canvas is
    // ~5.6k dots and this repaints on every page transition.
    final points = <Offset>[];
    for (var y = DotGrid.spacing / 2; y < size.height; y += DotGrid.spacing) {
      for (var x = DotGrid.spacing / 2; x < size.width; x += DotGrid.spacing) {
        points.add(Offset(x, y));
      }
    }
    canvas.drawPoints(
      PointMode.points,
      points,
      paint
        ..strokeCap = StrokeCap.round
        ..strokeWidth = 2,
    );
  }

  @override
  bool shouldRepaint(_DotGridPainter old) => old.color != color;
}

/// Corner-radius scale: inputs and chips, tiles, cards, pills.
class Radii {
  const Radii._();

  static const double sm = 8;
  static const double md = 12;
  static const double lg = 14;
  static const double pill = 999;
}

/// Elevation scale. Three steps only — a resting surface, a floating one, and
/// the main canvas card. All cast from the same near-black so stacked surfaces
/// read as one light source rather than a pile of greys.
class Shadows {
  const Shadows._();

  /// The shade every shadow is cast from — the ink neutral at low alpha.
  static const Color _cast = Color(0xff10141a);

  /// Resting: chips, stat tiles, the app bar. Barely there, just enough to lift
  /// off the canvas.
  static const List<BoxShadow> sm = [
    BoxShadow(color: Color(0x0d10141a), blurRadius: 2, offset: Offset(0, 1)),
  ];

  /// Floating: the composer, menus, anything that sits above content.
  static const List<BoxShadow> md = [
    BoxShadow(color: Color(0x0a10141a), blurRadius: 2, offset: Offset(0, 1)),
    BoxShadow(
      color: Color(0x2410141a),
      blurRadius: 24,
      spreadRadius: -10,
      offset: Offset(0, 8),
    ),
  ];

  /// The canvas card — the single most prominent surface on a page.
  static const List<BoxShadow> lg = [
    BoxShadow(color: Color(0x0a10141a), blurRadius: 2, offset: Offset(0, 1)),
    BoxShadow(
      color: Color(0x2e10141a),
      blurRadius: 44,
      spreadRadius: -16,
      offset: Offset(0, 18),
    ),
  ];

  /// Colour for [Material.shadowColor] / [CardThemeData.shadowColor], where a
  /// widget casts its own shadow from an `elevation` and will not take a
  /// [BoxShadow] list.
  static const Color materialCast = _cast;
}
