import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

import '../theme/app_theme.dart';
import '../theme/easy_colors.dart';
import '../theme/tokens.dart';

/// Copies [text] and confirms with a floating snackbar.
Future<void> copyToClipboard(
  BuildContext context,
  String text, {
  String? label,
}) async {
  await Clipboard.setData(ClipboardData(text: text));
  if (!context.mounted) return;
  ScaffoldMessenger.of(context)
    ..clearSnackBars()
    ..showSnackBar(
      SnackBar(
        duration: const Duration(milliseconds: 1400),
        content: Row(
          children: [
            Icon(Icons.check_circle, color: context.colors.success, size: 18),
            const SizedBox(width: 10),
            Expanded(
              child: Text(
                label ?? 'Copied to clipboard',
                maxLines: 1,
                overflow: TextOverflow.ellipsis,
              ),
            ),
          ],
        ),
      ),
    );
}

/// Small icon button that copies [text] and flips to a checkmark briefly.
class CopyIconButton extends StatefulWidget {
  const CopyIconButton({
    super.key,
    required this.text,
    this.tooltip = 'Copy',
    this.label,
    this.size = 16,
  });

  final String text;
  final String tooltip;
  final String? label;
  final double size;

  @override
  State<CopyIconButton> createState() => _CopyIconButtonState();
}

class _CopyIconButtonState extends State<CopyIconButton> {
  bool copied = false;

  @override
  Widget build(BuildContext context) {
    return IconButton(
      tooltip: widget.tooltip,
      visualDensity: VisualDensity.compact,
      iconSize: widget.size,
      onPressed: () async {
        await copyToClipboard(context, widget.text, label: widget.label);
        if (!mounted) return;
        setState(() => copied = true);
        Future.delayed(const Duration(milliseconds: 1400), () {
          if (mounted) setState(() => copied = false);
        });
      },
      icon: AnimatedSwitcher(
        duration: const Duration(milliseconds: 180),
        transitionBuilder: (child, animation) =>
            ScaleTransition(scale: animation, child: child),
        child: copied
            ? Icon(
                Icons.check,
                key: const ValueKey('check'),
                size: widget.size,
                color: context.colors.success,
              )
            : Icon(
                Icons.copy_rounded,
                key: const ValueKey('copy'),
                size: widget.size,
              ),
      ),
    );
  }
}

/// The main content surface: a floating card with a title bar.
///
/// Not a [Card] — [Shadows.lg] is the designed elevation and Material's
/// `elevation:` only approximates it. The card has no outline at all; depth is
/// the only thing separating it from the canvas. The hairline *under the
/// header* stays, though: it separates two regions of the same surface, which
/// is a different job from outlining the surface itself.
class Panel extends StatelessWidget {
  const Panel({
    super.key,
    required this.title,
    required this.child,
    this.icon,
    this.actions = const [],
  });

  final String title;
  final IconData? icon;
  final Widget child;
  final List<Widget> actions;

  /// Height of the header bar.
  static const double headerHeight = 54;

  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;
    final radius = BorderRadius.circular(Radii.lg);
    return Padding(
      // Roomier than the old all(8): shadow-lg reaches 18px down with a 44px
      // blur, and a tight gutter clips it into a smudge.
      padding: const EdgeInsets.fromLTRB(
        Insets.lg,
        Insets.md,
        Insets.lg,
        Insets.lg,
      ),
      child: DecoratedBox(
        decoration: BoxDecoration(
          color: scheme.surfaceContainerLow,
          borderRadius: radius,
          boxShadow: Shadows.lg,
        ),
        // A Material ancestor, which [Card] used to provide for free. Without
        // it every InkWell inside a panel loses its ripple and ListTile throws
        // "background color or ink splashes may be invisible" — silently, in
        // debug only. transparency: the DecoratedBox above already paints.
        child: Material(
          type: MaterialType.transparency,
          child: ClipRRect(
            borderRadius: radius,
            child: LayoutBuilder(
              builder: (context, constraints) => Column(
                crossAxisAlignment: CrossAxisAlignment.stretch,
                // min + Flexible (not max + Expanded) so a Panel whose [child]
                // shrink-wraps — a `shrinkWrap: true` ListView — sizes to its
                // content instead of stretching to the height it was offered.
                // A child that fills (every other page body is a plain
                // ListView) still fills, so this is a no-op for them.
                mainAxisSize: MainAxisSize.min,
                // ...and no Flexible at all when the height is unbounded: a
                // non-zero flex under an unbounded main axis is a RenderFlex
                // assertion, which is what a Panel inside a scroll view gives
                // you. See the [child] entry below.
                children: [
                  Container(
                    // Clamp so a Panel squeezed below its own chrome clips the
                    // header instead of asserting. Cheap insurance: an animated
                    // or fractional height can land under 54 for a frame.
                    height: constraints.hasBoundedHeight
                        ? (constraints.maxHeight < headerHeight
                              ? constraints.maxHeight
                              : headerHeight)
                        : headerHeight,
                    padding: const EdgeInsets.symmetric(horizontal: Insets.xl),
                    decoration: BoxDecoration(
                      border: Border(
                        bottom: BorderSide(color: scheme.outlineVariant),
                      ),
                    ),
                    child: Row(
                      children: [
                        if (icon != null) ...[
                          // Neutral, not primary: an icon is a signpost, not a
                          // status, and ink-only reserves colour for meaning.
                          Icon(icon, size: 17, color: scheme.onSurfaceVariant),
                          const SizedBox(width: Insets.sm),
                        ],
                        Expanded(
                          child: Text(
                            title,
                            overflow: TextOverflow.ellipsis,
                            style: const TextStyle(
                              fontWeight: FontWeight.w600,
                              fontSize: 15,
                              letterSpacing: -0.15,
                            ),
                          ),
                        ),
                        ...actions,
                      ],
                    ),
                  ),
                  if (constraints.hasBoundedHeight)
                    Flexible(child: child)
                  else
                    child,
                ],
              ),
            ),
          ),
        ),
      ),
    );
  }
}

/// Section heading used inside panels.
class SectionHeader extends StatelessWidget {
  const SectionHeader(this.title, {super.key, this.trailing});

  final String title;
  final Widget? trailing;

  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;
    return Padding(
      padding: const EdgeInsets.only(top: Insets.md, bottom: Insets.sm),
      child: Row(
        children: [
          // Expanded, and NO Spacer beside it. `Flexible` + `Spacer` both took
          // flex 1, so the Spacer only ever got *half* the free space and the
          // loose Flexible's unclaimed half was dropped at the row's right
          // edge — which parked every trailing action (Test reward, the runs
          // refresh, the vector-size chip, the checklist progress) mid-row
          // instead of against the right margin. Expanded is tight: the label
          // takes the whole gap and ellipsizes, [trailing] sits flush right.
          Expanded(
            child: Text(
              title.toUpperCase(),
              maxLines: 1,
              overflow: TextOverflow.ellipsis,
              // Grey, not primary. A section label is the quietest thing on
              // the page; it was previously the loudest.
              style: TextStyle(
                fontWeight: FontWeight.w600,
                fontSize: 11,
                letterSpacing: 0.9,
                color: scheme.onSurfaceVariant,
              ),
            ),
          ),
          if (trailing != null) ...[
            const SizedBox(width: Insets.md),
            trailing!,
          ],
        ],
      ),
    );
  }
}

/// Compact stat tile: label on top, prominent value below, copyable.
class StatChip extends StatelessWidget {
  const StatChip({
    super.key,
    required this.label,
    required this.value,
    this.icon,
    this.color,
  });

  final String label;
  final String value;
  final IconData? icon;
  final Color? color;

  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;
    // [color] is for a *semantic* value (a success-green mean reward), not
    // decoration. Unset means ink — the tile carries no tint of its own.
    final valueColor = color ?? scheme.onSurface;
    return Tooltip(
      message: 'Click to copy',
      child: InkWell(
        borderRadius: BorderRadius.circular(Radii.md),
        onTap: () => copyToClipboard(context, value, label: 'Copied "$value"'),
        child: Container(
          // md, not lg: three of these share a 340px column, and 16px of gutter
          // on each side left too little for "MOVABLE" — the label ellipsised
          // to "MOVA…", which is worse than a tighter tile.
          padding: const EdgeInsets.symmetric(
            horizontal: Insets.md,
            vertical: Insets.md,
          ),
          decoration: BoxDecoration(
            borderRadius: BorderRadius.circular(Radii.md),
            color: scheme.surfaceContainerLow,
            boxShadow: Shadows.sm,
          ),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            mainAxisSize: MainAxisSize.min,
            children: [
              // Flexible + ellipsis on both lines: these tiles are laid out in
              // fixed-width Row slots (three across a 340px column), so a long
              // robot name must clip rather than overflow the tile.
              Row(
                mainAxisSize: MainAxisSize.min,
                children: [
                  if (icon != null) ...[
                    Icon(icon, size: 13, color: scheme.onSurfaceVariant),
                    const SizedBox(width: 5),
                  ],
                  Flexible(
                    child: Text(
                      label.toUpperCase(),
                      maxLines: 1,
                      overflow: TextOverflow.ellipsis,
                      style: TextStyle(
                        fontSize: 10.5,
                        fontWeight: FontWeight.w600,
                        letterSpacing: 0.7,
                        color: scheme.onSurfaceVariant,
                      ),
                    ),
                  ),
                ],
              ),
              const SizedBox(height: Insets.xs),
              Text(
                value,
                maxLines: 1,
                overflow: TextOverflow.ellipsis,
                style: TextStyle(
                  fontSize: 16,
                  fontWeight: FontWeight.w600,
                  letterSpacing: -0.3,
                  color: valueColor,
                  fontFeatures: const [FontFeature.tabularFigures()],
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

/// Friendly placeholder for empty or offline content.
class EmptyState extends StatelessWidget {
  const EmptyState({
    super.key,
    required this.icon,
    required this.title,
    this.subtitle,
    this.action,
  });

  final IconData icon;
  final String title;
  final String? subtitle;
  final Widget? action;

  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;
    return Center(
      child: SingleChildScrollView(
        padding: const EdgeInsets.all(Insets.xl),
        child: ConstrainedBox(
          constraints: const BoxConstraints(maxWidth: 520),
          child: DottedBorderBox(
            child: Column(
              mainAxisSize: MainAxisSize.min,
              children: [
                Container(
                  width: 52,
                  height: 52,
                  decoration: BoxDecoration(
                    // Rounded square, not a circle: it reads as a placeholder
                    // slot rather than an avatar.
                    borderRadius: BorderRadius.circular(Radii.lg),
                    color: scheme.surfaceContainerHighest,
                  ),
                  child: Icon(icon, size: 24, color: scheme.onSurfaceVariant),
                ),
                const SizedBox(height: Insets.lg),
                Text(
                  title,
                  textAlign: TextAlign.center,
                  style: const TextStyle(
                    fontWeight: FontWeight.w600,
                    fontSize: 17,
                    letterSpacing: -0.2,
                  ),
                ),
                if (subtitle != null) ...[
                  const SizedBox(height: Insets.sm),
                  Text(
                    subtitle!,
                    textAlign: TextAlign.center,
                    style: TextStyle(
                      fontSize: 13,
                      height: 1.5,
                      color: scheme.onSurfaceVariant,
                    ),
                  ),
                ],
                if (action != null) ...[
                  const SizedBox(height: Insets.xl),
                  action!,
                ],
              ],
            ),
          ),
        ),
      ),
    );
  }
}

/// A dashed rounded rectangle, for drop targets and empty slots. Flutter has no
/// dashed [BorderSide], so the dashes are painted; everything else in the app
/// uses a solid hairline or a shadow.
class DottedBorderBox extends StatelessWidget {
  const DottedBorderBox({super.key, required this.child});

  final Widget child;

  @override
  Widget build(BuildContext context) {
    return CustomPaint(
      painter: _DashedRectPainter(
        color: Theme.of(context).colorScheme.outline,
        radius: 16,
      ),
      child: Padding(
        padding: const EdgeInsets.symmetric(horizontal: 40, vertical: 44),
        child: child,
      ),
    );
  }
}

class _DashedRectPainter extends CustomPainter {
  const _DashedRectPainter({required this.color, required this.radius});

  final Color color;
  final double radius;

  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()
      ..color = color
      ..style = PaintingStyle.stroke
      ..strokeWidth = 1.5;
    final rrect = RRect.fromRectAndRadius(
      Offset.zero & size,
      Radius.circular(radius),
    );
    // Walk the rounded-rect path and stroke alternating 6px on / 5px off.
    for (final metric in (Path()..addRRect(rrect)).computeMetrics()) {
      var d = 0.0;
      while (d < metric.length) {
        final end = (d + 6).clamp(0.0, metric.length);
        canvas.drawPath(metric.extractPath(d, end), paint);
        d = end + 5;
      }
    }
  }

  @override
  bool shouldRepaint(_DashedRectPainter old) =>
      old.color != color || old.radius != radius;
}

/// Monospace value with a copy affordance, used for paths, JSON and IDs.
class CopyableValue extends StatelessWidget {
  const CopyableValue({
    super.key,
    required this.value,
    this.maxLines = 1,
    this.fontSize,
  });

  final String value;
  final int maxLines;
  final double? fontSize;

  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;
    return Container(
      padding: const EdgeInsets.only(left: 10),
      decoration: BoxDecoration(
        borderRadius: BorderRadius.circular(Radii.sm),
        // Sunken fill, no outline — a recessed field reads as inset without
        // adding another rectangle to the page.
        color: scheme.surfaceContainerHighest,
      ),
      child: Row(
        children: [
          Expanded(
            child: Text(
              value,
              maxLines: maxLines,
              overflow: TextOverflow.ellipsis,
              style: monoStyle(context, fontSize: fontSize),
            ),
          ),
          CopyIconButton(text: value),
        ],
      ),
    );
  }
}

/// Colored status dot + label, e.g. backend online/offline.
class StatusDot extends StatelessWidget {
  const StatusDot({super.key, required this.label, required this.online});

  final String label;
  final bool online;

  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;
    final color = online ? context.colors.success : context.colors.danger;
    return LayoutBuilder(
      builder: (context, constraints) {
        // When squeezed (narrow panels), degrade to just the dot.
        final compact =
            constraints.hasBoundedWidth && constraints.maxWidth < 80;
        return Tooltip(
          message: compact ? label : '',
          child: Container(
            padding: compact
                ? const EdgeInsets.all(5)
                : const EdgeInsets.symmetric(horizontal: 10, vertical: 5),
            // Neutral pill, coloured dot. The dot already carries the state;
            // tinting the whole chip (and its border, and its text) made a
            // routine "Connected" as loud as a real warning.
            decoration: BoxDecoration(
              borderRadius: BorderRadius.circular(Radii.pill),
              color: scheme.surfaceContainerHighest,
            ),
            child: compact
                ? _PulsingDot(color: color, animate: online)
                : Row(
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      _PulsingDot(color: color, animate: online),
                      const SizedBox(width: 7),
                      Flexible(
                        child: Text(
                          label,
                          maxLines: 1,
                          overflow: TextOverflow.ellipsis,
                          style: TextStyle(
                            fontSize: 12,
                            fontWeight: FontWeight.w500,
                            color: scheme.onSurfaceVariant,
                          ),
                        ),
                      ),
                    ],
                  ),
          ),
        );
      },
    );
  }
}

class _PulsingDot extends StatefulWidget {
  const _PulsingDot({required this.color, required this.animate});

  final Color color;
  final bool animate;

  @override
  State<_PulsingDot> createState() => _PulsingDotState();
}

class _PulsingDotState extends State<_PulsingDot>
    with SingleTickerProviderStateMixin {
  late final AnimationController controller = AnimationController(
    vsync: this,
    duration: const Duration(milliseconds: 1600),
  );

  @override
  void initState() {
    super.initState();
    if (widget.animate) controller.repeat();
  }

  @override
  void didUpdateWidget(covariant _PulsingDot oldWidget) {
    super.didUpdateWidget(oldWidget);
    if (widget.animate && !controller.isAnimating) {
      controller.repeat();
    } else if (!widget.animate && controller.isAnimating) {
      controller.stop();
      controller.value = 0;
    }
  }

  @override
  void dispose() {
    controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return AnimatedBuilder(
      animation: controller,
      builder: (context, _) {
        final t = controller.value;
        return SizedBox(
          width: 14,
          height: 14,
          child: Stack(
            alignment: Alignment.center,
            children: [
              if (widget.animate)
                Container(
                  width: 6 + 8 * t,
                  height: 6 + 8 * t,
                  decoration: BoxDecoration(
                    shape: BoxShape.circle,
                    color: widget.color.withValues(alpha: 0.4 * (1 - t)),
                  ),
                ),
              Container(
                width: 7,
                height: 7,
                decoration: BoxDecoration(
                  shape: BoxShape.circle,
                  color: widget.color,
                ),
              ),
            ],
          ),
        );
      },
    );
  }
}
