import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

import '../theme/app_theme.dart';
import '../theme/easy_colors.dart';

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

/// Card-style panel with an icon, title and optional trailing actions.
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

  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;
    return Padding(
      padding: const EdgeInsets.all(8),
      child: Card(
        clipBehavior: Clip.antiAlias,
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            Padding(
              padding: const EdgeInsets.fromLTRB(14, 8, 8, 8),
              child: Row(
                children: [
                  if (icon != null) ...[
                    Icon(icon, size: 17, color: scheme.primary),
                    const SizedBox(width: 8),
                  ],
                  Expanded(
                    child: Text(
                      title,
                      overflow: TextOverflow.ellipsis,
                      style: const TextStyle(
                        fontWeight: FontWeight.w700,
                        fontSize: 13.5,
                        letterSpacing: 0.2,
                      ),
                    ),
                  ),
                  ...actions,
                ],
              ),
            ),
            const Divider(),
            Expanded(child: child),
          ],
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
      padding: const EdgeInsets.only(top: 6, bottom: 8),
      child: Row(
        children: [
          Text(
            title.toUpperCase(),
            style: TextStyle(
              fontWeight: FontWeight.w700,
              fontSize: 11.5,
              letterSpacing: 1.1,
              color: scheme.primary,
            ),
          ),
          const SizedBox(width: 10),
          Expanded(child: Divider(color: scheme.outlineVariant)),
          if (trailing != null) ...[const SizedBox(width: 8), trailing!],
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
    final accent = color ?? scheme.primary;
    return Tooltip(
      message: 'Click to copy',
      child: InkWell(
        borderRadius: BorderRadius.circular(10),
        onTap: () => copyToClipboard(context, value, label: 'Copied "$value"'),
        child: Container(
          padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 10),
          decoration: BoxDecoration(
            borderRadius: BorderRadius.circular(10),
            border: Border.all(color: scheme.outlineVariant),
            color: accent.withValues(alpha: 0.06),
          ),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            mainAxisSize: MainAxisSize.min,
            children: [
              Row(
                mainAxisSize: MainAxisSize.min,
                children: [
                  if (icon != null) ...[
                    Icon(icon, size: 13, color: accent),
                    const SizedBox(width: 5),
                  ],
                  Text(
                    label,
                    style: TextStyle(
                      fontSize: 11,
                      color: scheme.onSurface.withValues(alpha: 0.65),
                    ),
                  ),
                ],
              ),
              const SizedBox(height: 3),
              Text(
                value,
                style: TextStyle(
                  fontSize: 15,
                  fontWeight: FontWeight.w700,
                  color: accent,
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
        padding: const EdgeInsets.all(24),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Container(
              padding: const EdgeInsets.all(18),
              decoration: BoxDecoration(
                shape: BoxShape.circle,
                color: scheme.primary.withValues(alpha: 0.08),
              ),
              child: Icon(
                icon,
                size: 34,
                color: scheme.primary.withValues(alpha: 0.8),
              ),
            ),
            const SizedBox(height: 14),
            Text(
              title,
              textAlign: TextAlign.center,
              style: const TextStyle(
                fontWeight: FontWeight.w700,
                fontSize: 14.5,
              ),
            ),
            if (subtitle != null) ...[
              const SizedBox(height: 6),
              ConstrainedBox(
                constraints: const BoxConstraints(maxWidth: 360),
                child: Text(
                  subtitle!,
                  textAlign: TextAlign.center,
                  style: TextStyle(
                    fontSize: 12.5,
                    height: 1.4,
                    color: scheme.onSurface.withValues(alpha: 0.6),
                  ),
                ),
              ),
            ],
            if (action != null) ...[const SizedBox(height: 16), action!],
          ],
        ),
      ),
    );
  }
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
        borderRadius: BorderRadius.circular(8),
        color: scheme.onSurface.withValues(alpha: 0.05),
        border: Border.all(color: scheme.outlineVariant),
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
            decoration: BoxDecoration(
              borderRadius: BorderRadius.circular(20),
              color: color.withValues(alpha: 0.1),
              border: Border.all(color: color.withValues(alpha: 0.35)),
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
                            fontWeight: FontWeight.w600,
                            color: color,
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
