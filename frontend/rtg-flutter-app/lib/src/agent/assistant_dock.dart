import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:flutter_riverpod/legacy.dart';
import 'package:shared_preferences/shared_preferences.dart';

import 'chat_view.dart';

const _dockPrefKey = 'easyrtg.dock_expanded.v1';
const _dockWidthPrefKey = 'easyrtg.dock_width.v1';
const double dockCollapsedWidth = 52;
const double dockMinWidth = 300;
const double dockMaxWidth = 720;
const double dockDefaultWidth = 380;

/// Whether the co-pilot dock is expanded. App-scoped so any surface (the
/// shell's auto-collapse, an "Ask the assistant" button) can open/close it.
final dockExpandedProvider = StateProvider<bool>((ref) => true);

/// The dock's expanded width (user-resizable via the drag handle).
final dockWidthProvider = StateProvider<double>((ref) => dockDefaultWidth);

/// Set the dock open/closed state and persist it. Use from anywhere.
Future<void> setDockExpanded(WidgetRef ref, bool value) async {
  if (ref.read(dockExpandedProvider) == value) return;
  ref.read(dockExpandedProvider.notifier).state = value;
  try {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setBool(_dockPrefKey, value);
  } catch (_) {}
}

/// The persistent assistant rail on the right of the shell. Expanded it shows
/// the conversation and can be dragged wider/narrower; collapsed it's a slim
/// strip that reopens it. The shell auto-collapses it on narrow windows.
class AssistantDock extends ConsumerStatefulWidget {
  const AssistantDock({super.key});

  @override
  ConsumerState<AssistantDock> createState() => _AssistantDockState();
}

class _AssistantDockState extends ConsumerState<AssistantDock> {
  bool _dragging = false;

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addPostFrameCallback((_) => _restore());
  }

  Future<void> _restore() async {
    try {
      final prefs = await SharedPreferences.getInstance();
      final savedExpanded = prefs.getBool(_dockPrefKey);
      final savedWidth = prefs.getDouble(_dockWidthPrefKey);
      if (!mounted) return;
      if (savedExpanded != null) {
        ref.read(dockExpandedProvider.notifier).state = savedExpanded;
      }
      if (savedWidth != null) {
        ref.read(dockWidthProvider.notifier).state = savedWidth.clamp(
          dockMinWidth,
          dockMaxWidth,
        );
      }
    } catch (_) {}
  }

  void _onDragUpdate(double dx) {
    // Dragging the handle left (negative dx) widens the dock.
    final next = (ref.read(dockWidthProvider) - dx).clamp(
      dockMinWidth,
      dockMaxWidth,
    );
    ref.read(dockWidthProvider.notifier).state = next;
  }

  Future<void> _onDragEnd() async {
    setState(() => _dragging = false);
    try {
      final prefs = await SharedPreferences.getInstance();
      await prefs.setDouble(_dockWidthPrefKey, ref.read(dockWidthProvider));
    } catch (_) {}
  }

  /// Lay [child] out at a fixed [width] regardless of the animating container
  /// width, clipping the overflow — so the chat never re-flows (or throws a
  /// RenderFlex overflow) mid-animation; it slides in from the right.
  Widget _fixedWidth(double width, Widget child) {
    return ClipRect(
      child: OverflowBox(
        minWidth: width,
        maxWidth: width,
        alignment: Alignment.centerLeft,
        child: SizedBox(width: width, child: child),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;
    final expanded = ref.watch(dockExpandedProvider);
    final width = ref
        .watch(dockWidthProvider)
        .clamp(dockMinWidth, dockMaxWidth);
    return AnimatedContainer(
      // No tween while dragging, so resize tracks the cursor exactly; otherwise
      // animate the open/close transition.
      duration: _dragging ? Duration.zero : const Duration(milliseconds: 220),
      curve: Curves.easeOutCubic,
      width: expanded ? width : dockCollapsedWidth,
      decoration: BoxDecoration(
        color: scheme.surface,
        border: Border(left: BorderSide(color: scheme.outlineVariant)),
      ),
      child: AnimatedSwitcher(
        duration: const Duration(milliseconds: 160),
        child: expanded
            ? KeyedSubtree(
                key: const ValueKey('dock-chat'),
                child: _fixedWidth(
                  width,
                  Row(
                    children: [
                      _ResizeHandle(
                        dragging: _dragging,
                        onStart: () => setState(() => _dragging = true),
                        onUpdate: _onDragUpdate,
                        onEnd: _onDragEnd,
                      ),
                      Expanded(
                        child: AssistantChat(
                          onCollapse: () => setDockExpanded(ref, false),
                        ),
                      ),
                    ],
                  ),
                ),
              )
            : KeyedSubtree(
                key: const ValueKey('dock-strip'),
                child: _fixedWidth(
                  dockCollapsedWidth,
                  _CollapsedStrip(onExpand: () => setDockExpanded(ref, true)),
                ),
              ),
      ),
    );
  }
}

/// Thin draggable divider on the dock's left edge to resize it.
class _ResizeHandle extends StatefulWidget {
  const _ResizeHandle({
    required this.dragging,
    required this.onStart,
    required this.onUpdate,
    required this.onEnd,
  });

  final bool dragging;
  final VoidCallback onStart;
  final void Function(double dx) onUpdate;
  final VoidCallback onEnd;

  @override
  State<_ResizeHandle> createState() => _ResizeHandleState();
}

class _ResizeHandleState extends State<_ResizeHandle> {
  bool _hover = false;

  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;
    final active = _hover || widget.dragging;
    return MouseRegion(
      cursor: SystemMouseCursors.resizeLeftRight,
      onEnter: (_) => setState(() => _hover = true),
      onExit: (_) => setState(() => _hover = false),
      child: GestureDetector(
        behavior: HitTestBehavior.translucent,
        onHorizontalDragStart: (_) => widget.onStart(),
        onHorizontalDragUpdate: (details) => widget.onUpdate(details.delta.dx),
        onHorizontalDragEnd: (_) => widget.onEnd(),
        child: SizedBox(
          width: 10,
          child: Center(
            child: SizedBox(
              width: active ? 3 : 1.5,
              height: double.infinity,
              child: ColoredBox(
                color: active ? scheme.primary : scheme.outlineVariant,
              ),
            ),
          ),
        ),
      ),
    );
  }
}

class _CollapsedStrip extends StatelessWidget {
  const _CollapsedStrip({required this.onExpand});

  final VoidCallback onExpand;

  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;
    return Tooltip(
      message: 'Open the assistant',
      child: InkWell(
        onTap: onExpand,
        child: Column(
          children: [
            const SizedBox(height: 10),
            IconButton(
              tooltip: 'Open the assistant',
              visualDensity: VisualDensity.compact,
              iconSize: 18,
              onPressed: onExpand,
              icon: const Icon(Icons.chevron_left),
            ),
            const SizedBox(height: 4),
            CircleAvatar(
              radius: 15,
              backgroundColor: scheme.primary.withValues(alpha: 0.15),
              child: Icon(
                Icons.smart_toy_outlined,
                size: 17,
                color: scheme.primary,
              ),
            ),
            const Spacer(),
            RotatedBox(
              quarterTurns: 3,
              child: Text(
                'ASSISTANT',
                style: TextStyle(
                  fontSize: 11,
                  fontWeight: FontWeight.w700,
                  letterSpacing: 2,
                  color: scheme.onSurface.withValues(alpha: 0.5),
                ),
              ),
            ),
            const Spacer(),
          ],
        ),
      ),
    );
  }
}
