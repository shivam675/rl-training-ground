import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../app_state.dart';
import '../nav.dart';
import '../theme/theme_controller.dart';

/// The left navigation rail. Custom (rather than [NavigationRail]) so the
/// destinations can be grouped under section labels — Home · BUILD · TRAIN ·
/// tools — which turns the rail into a map of the workflow, and so locked
/// destinations can show an inline lock badge. The brand logo doubles as the
/// application (File-style) menu; a theme toggle is pinned at the bottom.
class AppRail extends ConsumerWidget {
  const AppRail({
    super.key,
    required this.onSelect,
    required this.onNew,
    required this.onOpen,
    required this.onSave,
    required this.onSaveAs,
    required this.onSettings,
    required this.onDiagnostics,
    required this.onAbout,
  });

  /// Called when a destination is tapped (locked or not — the host decides
  /// whether to navigate or explain why it's locked).
  final void Function(AppPage page) onSelect;
  final VoidCallback onNew;
  final VoidCallback onOpen;
  final VoidCallback onSave;
  final VoidCallback onSaveAs;
  final VoidCallback onSettings;
  final VoidCallback onDiagnostics;
  final VoidCallback onAbout;

  static const _build = [AppPage.robot, AppPage.obsAction, AppPage.rewards];
  static const _train = [AppPage.training, AppPage.evaluation];
  static const _tools = [AppPage.settings, AppPage.logs];

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final scheme = Theme.of(context).colorScheme;
    final dark = Theme.of(context).brightness == Brightness.dark;
    final selected = ref.watch(navIndexProvider);
    final state = ref.watch(appStateProvider);
    final theme = ref.watch(themeControllerProvider);

    // Only Training is gated; Evaluation stays reachable so past runs can always
    // be inspected and compared.
    bool locked(AppPage page) =>
        page == AppPage.training && !state.canStartTraining;

    Widget item(AppPage page) => _RailItem(
      page: page,
      selected: selected == page,
      locked: locked(page),
      onTap: () => onSelect(page),
    );

    return Container(
      width: 92,
      color: dark ? const Color(0xff10141a) : Colors.white,
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          const SizedBox(height: 10),
          Center(
            child: _AppMenuButton(
              onNew: onNew,
              onOpen: onOpen,
              onSave: onSave,
              onSaveAs: onSaveAs,
              onSettings: onSettings,
              onDiagnostics: onDiagnostics,
              onAbout: onAbout,
            ),
          ),
          const SizedBox(height: 10),
          Expanded(
            child: SingleChildScrollView(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.stretch,
                children: [
                  item(AppPage.home),
                  const _RailGroupLabel('Build'),
                  for (final page in _build) item(page),
                  const _RailGroupLabel('Train'),
                  for (final page in _train) item(page),
                ],
              ),
            ),
          ),
          Divider(height: 1, color: scheme.outlineVariant),
          const SizedBox(height: 4),
          for (final page in _tools) item(page),
          const SizedBox(height: 4),
          IconButton(
            tooltip: dark ? 'Switch to light mode' : 'Switch to dark mode',
            onPressed: () =>
                theme.setMode(dark ? ThemeMode.light : ThemeMode.dark),
            icon: Icon(
              dark ? Icons.light_mode_outlined : Icons.dark_mode_outlined,
              size: 20,
            ),
          ),
          const SizedBox(height: 8),
        ],
      ),
    );
  }
}

class _RailItem extends StatelessWidget {
  const _RailItem({
    required this.page,
    required this.selected,
    required this.locked,
    required this.onTap,
  });

  final AppPage page;
  final bool selected;
  final bool locked;
  final VoidCallback onTap;

  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;
    final color = selected
        ? scheme.primary
        : scheme.onSurface.withValues(alpha: locked ? 0.38 : 0.7);
    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 3),
      child: InkWell(
        borderRadius: BorderRadius.circular(10),
        onTap: onTap,
        child: AnimatedContainer(
          duration: const Duration(milliseconds: 160),
          padding: const EdgeInsets.symmetric(vertical: 8),
          decoration: BoxDecoration(
            borderRadius: BorderRadius.circular(10),
            color: selected ? scheme.primary.withValues(alpha: 0.14) : null,
          ),
          child: Column(
            children: [
              if (locked)
                _LockedIcon(icon: page.icon, color: color)
              else
                Icon(page.icon, size: 22, color: color),
              const SizedBox(height: 5),
              Text(
                page.label,
                textAlign: TextAlign.center,
                maxLines: 2,
                style: TextStyle(
                  fontSize: 10.5,
                  height: 1.15,
                  fontWeight: selected ? FontWeight.w700 : FontWeight.w500,
                  color: color,
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

/// A nav icon dimmed with a small lock overlay, for gated destinations.
class _LockedIcon extends StatelessWidget {
  const _LockedIcon({required this.icon, required this.color});

  final IconData icon;
  final Color color;

  @override
  Widget build(BuildContext context) {
    return Stack(
      clipBehavior: Clip.none,
      children: [
        Icon(icon, size: 22, color: color),
        Positioned(
          right: -6,
          top: -4,
          child: Icon(
            Icons.lock,
            size: 11,
            color: Theme.of(
              context,
            ).colorScheme.onSurface.withValues(alpha: 0.6),
          ),
        ),
      ],
    );
  }
}

class _RailGroupLabel extends StatelessWidget {
  const _RailGroupLabel(this.text);

  final String text;

  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;
    return Padding(
      padding: const EdgeInsets.fromLTRB(12, 12, 12, 4),
      child: Row(
        children: [
          Text(
            text.toUpperCase(),
            style: TextStyle(
              fontSize: 9.5,
              fontWeight: FontWeight.w700,
              letterSpacing: 1.1,
              color: scheme.onSurface.withValues(alpha: 0.4),
            ),
          ),
          const SizedBox(width: 6),
          Expanded(child: Divider(height: 1, color: scheme.outlineVariant)),
        ],
      ),
    );
  }
}

/// The brand logo doubles as the application menu trigger (File-style actions).
class _AppMenuButton extends StatelessWidget {
  const _AppMenuButton({
    required this.onNew,
    required this.onOpen,
    required this.onSave,
    required this.onSaveAs,
    required this.onSettings,
    required this.onDiagnostics,
    required this.onAbout,
  });

  final VoidCallback onNew;
  final VoidCallback onOpen;
  final VoidCallback onSave;
  final VoidCallback onSaveAs;
  final VoidCallback onSettings;
  final VoidCallback onDiagnostics;
  final VoidCallback onAbout;

  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;
    return MenuAnchor(
      menuChildren: [
        MenuItemButton(
          leadingIcon: const Icon(Icons.note_add_outlined, size: 18),
          shortcut: const SingleActivator(
            LogicalKeyboardKey.keyN,
            control: true,
          ),
          onPressed: onNew,
          child: const Text('New Project'),
        ),
        MenuItemButton(
          leadingIcon: const Icon(Icons.folder_open_outlined, size: 18),
          shortcut: const SingleActivator(
            LogicalKeyboardKey.keyO,
            control: true,
          ),
          onPressed: onOpen,
          child: const Text('Open Project…'),
        ),
        MenuItemButton(
          leadingIcon: const Icon(Icons.save_outlined, size: 18),
          shortcut: const SingleActivator(
            LogicalKeyboardKey.keyS,
            control: true,
          ),
          onPressed: onSave,
          child: const Text('Save Project'),
        ),
        MenuItemButton(
          leadingIcon: const Icon(Icons.save_as_outlined, size: 18),
          shortcut: const SingleActivator(
            LogicalKeyboardKey.keyS,
            control: true,
            shift: true,
          ),
          onPressed: onSaveAs,
          child: const Text('Save Project As…'),
        ),
        const Divider(height: 8),
        MenuItemButton(
          leadingIcon: const Icon(Icons.settings_outlined, size: 18),
          onPressed: onSettings,
          child: const Text('Settings'),
        ),
        MenuItemButton(
          leadingIcon: const Icon(Icons.bug_report_outlined, size: 18),
          onPressed: onDiagnostics,
          child: const Text('Export Diagnostics…'),
        ),
        MenuItemButton(
          leadingIcon: const Icon(Icons.info_outline, size: 18),
          onPressed: onAbout,
          child: const Text('About EasyRTG'),
        ),
      ],
      builder: (context, controller, child) {
        return Tooltip(
          message: 'Application menu — New, Open, Save…',
          child: InkWell(
            borderRadius: BorderRadius.circular(11),
            onTap: () =>
                controller.isOpen ? controller.close() : controller.open(),
            child: Container(
              width: 38,
              height: 38,
              decoration: BoxDecoration(
                borderRadius: BorderRadius.circular(11),
                gradient: LinearGradient(
                  begin: Alignment.topLeft,
                  end: Alignment.bottomRight,
                  colors: [
                    scheme.primary,
                    scheme.primary.withValues(alpha: 0.55),
                  ],
                ),
              ),
              child: const Stack(
                alignment: Alignment.center,
                children: [
                  Icon(
                    Icons.precision_manufacturing,
                    size: 21,
                    color: Colors.white,
                  ),
                  Positioned(
                    right: 2,
                    bottom: 1,
                    child: Icon(
                      Icons.arrow_drop_down,
                      size: 13,
                      color: Colors.white,
                    ),
                  ),
                ],
              ),
            ),
          ),
        );
      },
    );
  }
}
