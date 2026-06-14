import 'dart:async';

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import 'agent/assistant_dock.dart';
import 'app_state.dart';
import 'nav.dart';
import 'notifications.dart';
import 'panels/evaluation_panel.dart';
import 'panels/home_panel.dart';
import 'panels/logs_panel.dart';
import 'panels/observation_action_panel.dart';
import 'panels/reward_panel.dart';
import 'panels/robot_panel.dart';
import 'panels/settings_panel.dart';
import 'panels/training_panel.dart';
import 'project_io.dart';
import 'theme/theme_controller.dart';
import 'widgets/app_rail.dart';
import 'widgets/command_palette.dart';
import 'widgets/common.dart';
import 'widgets/notification_panel.dart';

class EasyRtgHome extends ConsumerStatefulWidget {
  const EasyRtgHome({super.key});

  @override
  ConsumerState<EasyRtgHome> createState() => _EasyRtgHomeState();
}

class _EasyRtgHomeState extends ConsumerState<EasyRtgHome> {
  AppPage get selected => ref.watch(navIndexProvider);
  final scaffoldKey = GlobalKey<ScaffoldState>();
  StreamSubscription<AgentNotification>? toastSubscription;

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addPostFrameCallback((_) {
      toastSubscription = ref
          .read(notificationCenterProvider)
          .toasts
          .listen(showToast);
    });
  }

  @override
  void dispose() {
    toastSubscription?.cancel();
    super.dispose();
  }

  void showToast(AgentNotification notification) {
    if (!mounted) return;
    final color = severityColor(context, notification.severity);
    ScaffoldMessenger.of(context)
      ..clearSnackBars()
      ..showSnackBar(
        SnackBar(
          duration: const Duration(seconds: 4),
          content: Row(
            children: [
              Icon(severityIcon(notification.severity), color: color, size: 18),
              const SizedBox(width: 10),
              Expanded(
                child: Column(
                  mainAxisSize: MainAxisSize.min,
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      notification.title,
                      maxLines: 1,
                      overflow: TextOverflow.ellipsis,
                      style: const TextStyle(fontWeight: FontWeight.w700),
                    ),
                    if (notification.body.isNotEmpty)
                      Text(
                        notification.body,
                        maxLines: 2,
                        overflow: TextOverflow.ellipsis,
                        style: const TextStyle(fontSize: 12),
                      ),
                  ],
                ),
              ),
            ],
          ),
          action: SnackBarAction(
            label: 'View',
            onPressed: () {
              ref.read(notificationCenterProvider).markAllRead();
              scaffoldKey.currentState?.openEndDrawer();
            },
          ),
        ),
      );
  }

  // Training stays locked until the environment setup is valid. Evaluation is
  // intentionally NOT gated: it inspects past runs, which don't depend on the
  // current setup. Robot Setup / Obs-Action / Rewards stay open so the user —
  // with the assistant's help — can complete that setup.
  static const _gatedPages = {AppPage.training};

  bool _isLocked(AppPage page, AppState state) =>
      _gatedPages.contains(page) && !state.canStartTraining;

  // The co-pilot dock auto-collapses on narrow windows so the page stays
  // usable, and reopens when there's room again. Manual toggles still work.
  bool _autoCollapsedForNarrow = false;
  void _maybeAutoCollapseDock(double width) {
    final narrow = width < 1100;
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (!mounted) return;
      if (narrow && !_autoCollapsedForNarrow) {
        _autoCollapsedForNarrow = true;
        setDockExpanded(ref, false);
      } else if (!narrow && _autoCollapsedForNarrow) {
        _autoCollapsedForNarrow = false;
        setDockExpanded(ref, true);
      }
    });
  }

  void _onDestinationSelected(AppPage page, AppState state) {
    if (_isLocked(page, state)) {
      final blockers = state.trainingBlockers();
      ScaffoldMessenger.of(context)
        ..clearSnackBars()
        ..showSnackBar(
          SnackBar(
            duration: const Duration(seconds: 4),
            content: Row(
              children: [
                const Icon(Icons.lock_outline, size: 18),
                const SizedBox(width: 10),
                Expanded(
                  child: Text(
                    blockers.isEmpty
                        ? '${page.label} is locked.'
                        : '${page.label} unlocks once you: ${blockers.join(', ')}.',
                  ),
                ),
              ],
            ),
            action: SnackBarAction(
              label: 'Robot Setup',
              onPressed: () =>
                  ref.read(navIndexProvider.notifier).state = AppPage.robot,
            ),
          ),
        );
      return;
    }
    ref.read(navIndexProvider.notifier).state = page;
  }

  Future<void> _handleNew(AppState state) async {
    if (await ProjectIO.newProject(context, state) && mounted) {
      // Fresh workspace: point the user at where to begin.
      ref.read(navIndexProvider.notifier).state = AppPage.robot;
    }
  }

  Future<void> _handleOpen(AppState state) async {
    if (await ProjectIO.open(context, state) && mounted) {
      // Land on the dashboard so the setup status is visible at a glance.
      ref.read(navIndexProvider.notifier).state = AppPage.home;
    }
  }

  Map<ShortcutActivator, VoidCallback> _shortcutBindings(AppState state) {
    return {
      const SingleActivator(LogicalKeyboardKey.keyN, control: true): () =>
          _handleNew(state),
      const SingleActivator(LogicalKeyboardKey.keyO, control: true): () =>
          _handleOpen(state),
      const SingleActivator(LogicalKeyboardKey.keyS, control: true): () =>
          ProjectIO.save(context, state),
      const SingleActivator(
        LogicalKeyboardKey.keyS,
        control: true,
        shift: true,
      ): () =>
          ProjectIO.saveAs(context, state),
      const SingleActivator(LogicalKeyboardKey.keyK, control: true): () =>
          _showCommandPalette(state),
    };
  }

  void _showAbout() {
    showAboutDialog(
      context: context,
      applicationName: 'EasyRTG',
      applicationVersion: '0.1.0',
      applicationIcon: const Icon(Icons.precision_manufacturing, size: 32),
      children: const [
        Text(
          'Train reinforcement-learning policies for URDF robots — load a '
          'robot, let the assistant draft observations, actions and a reward, '
          'then train and evaluate.',
        ),
      ],
    );
  }

  void _showCommandPalette(AppState state) {
    final theme = ref.read(themeControllerProvider);
    final dark = Theme.of(context).brightness == Brightness.dark;
    final commands = <PaletteCommand>[
      for (final page in AppPage.values)
        PaletteCommand(
          label: page.label,
          hint: 'Go to',
          icon: page.icon,
          run: () => _onDestinationSelected(page, state),
        ),
      PaletteCommand(
        label: 'New Project',
        hint: 'Project',
        icon: Icons.note_add_outlined,
        run: () => _handleNew(state),
      ),
      PaletteCommand(
        label: 'Open Project…',
        hint: 'Project',
        icon: Icons.folder_open_outlined,
        run: () => _handleOpen(state),
      ),
      PaletteCommand(
        label: 'Save Project',
        hint: 'Project',
        icon: Icons.save_outlined,
        run: () => ProjectIO.save(context, state),
      ),
      PaletteCommand(
        label: 'Save Project As…',
        hint: 'Project',
        icon: Icons.save_as_outlined,
        run: () => ProjectIO.saveAs(context, state),
      ),
      PaletteCommand(
        label: 'Open the assistant',
        hint: 'Action',
        icon: Icons.smart_toy_outlined,
        run: () => setDockExpanded(ref, true),
      ),
      PaletteCommand(
        label: 'Load sample robot (R2D2)',
        hint: 'Action',
        icon: Icons.bolt,
        run: () => state.loadUrdf(
          path: 'r2d2.urdf',
          basePosition: const [0.0, 0.0, 0.5],
          fixedBase: false,
          addPlane: true,
        ),
      ),
      PaletteCommand(
        label: dark ? 'Switch to light mode' : 'Switch to dark mode',
        hint: 'Appearance',
        icon: dark ? Icons.light_mode_outlined : Icons.dark_mode_outlined,
        run: () => theme.setMode(dark ? ThemeMode.light : ThemeMode.dark),
      ),
      PaletteCommand(
        label: 'Export diagnostics…',
        hint: 'Action',
        icon: Icons.bug_report_outlined,
        run: () => _exportDiagnostics(state),
      ),
    ];
    showCommandPalette(context, commands);
  }

  @override
  Widget build(BuildContext context) {
    final state = ref.watch(appStateProvider);
    final scheme = Theme.of(context).colorScheme;
    final urdfPath = state.robotInfo?['path']?.toString();

    return CallbackShortcuts(
      bindings: _shortcutBindings(state),
      child: Focus(
        autofocus: false,
        child: Scaffold(
          key: scaffoldKey,
          endDrawer: const NotificationDrawer(),
          onEndDrawerChanged: (open) {
            if (open) ref.read(notificationCenterProvider).markAllRead();
          },
          body: LayoutBuilder(
            builder: (context, shellConstraints) {
              _maybeAutoCollapseDock(shellConstraints.maxWidth);
              return Row(
                children: [
                  AppRail(
                    onSelect: (page) => _onDestinationSelected(page, state),
                    onNew: () => _handleNew(state),
                    onOpen: () => _handleOpen(state),
                    onSave: () => ProjectIO.save(context, state),
                    onSaveAs: () => ProjectIO.saveAs(context, state),
                    onSettings: () =>
                        ref.read(navIndexProvider.notifier).state =
                            AppPage.settings,
                    onDiagnostics: () => _exportDiagnostics(state),
                    onAbout: _showAbout,
                  ),
                  VerticalDivider(width: 1, color: scheme.outlineVariant),
                  Expanded(
                    child: Column(
                      children: [
                        _Header(
                          selectedLabel: selected.label,
                          onOpenNotifications: () =>
                              scaffoldKey.currentState?.openEndDrawer(),
                          onOpenCommandPalette: () =>
                              _showCommandPalette(state),
                        ),
                        Expanded(
                          child: AnimatedSwitcher(
                            duration: const Duration(milliseconds: 220),
                            switchInCurve: Curves.easeOutCubic,
                            switchOutCurve: Curves.easeInCubic,
                            transitionBuilder: (child, animation) {
                              return FadeTransition(
                                opacity: animation,
                                child: SlideTransition(
                                  position: Tween<Offset>(
                                    begin: const Offset(0, 0.012),
                                    end: Offset.zero,
                                  ).animate(animation),
                                  child: child,
                                ),
                              );
                            },
                            child: KeyedSubtree(
                              key: ValueKey(selected),
                              child: _pageFor(selected, urdfPath),
                            ),
                          ),
                        ),
                        _StatusBar(message: state.message, busy: state.busy),
                      ],
                    ),
                  ),
                  const AssistantDock(),
                ],
              );
            },
          ),
        ),
      ),
    );
  }

  Future<void> _exportDiagnostics(AppState state) async {
    try {
      final res = await state.api.postJson('/diagnostics/export', {});
      final path = res['path']?.toString();
      if (!mounted) return;
      ScaffoldMessenger.of(context)
        ..clearSnackBars()
        ..showSnackBar(
          SnackBar(
            content: Text(
              path == null
                  ? 'Diagnostics exported.'
                  : 'Diagnostics exported to $path',
            ),
            action: path == null
                ? null
                : SnackBarAction(
                    label: 'Copy path',
                    onPressed: () => copyToClipboard(context, path),
                  ),
          ),
        );
    } catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(context)
        ..clearSnackBars()
        ..showSnackBar(
          SnackBar(content: Text('Diagnostics export failed: $e')),
        );
    }
  }

  Widget _pageFor(AppPage page, String? urdfPath) {
    return switch (page) {
      AppPage.home => HomePanel(urdfPath: urdfPath),
      AppPage.robot => Panel(
        title: 'Robot Setup',
        icon: Icons.precision_manufacturing_outlined,
        child: RobotPanel(urdfPath: urdfPath),
      ),
      AppPage.obsAction => const Panel(
        title: 'Observation / Action Space',
        icon: Icons.schema_outlined,
        child: ObservationActionPanel(),
      ),
      AppPage.rewards => const Panel(
        title: 'Reward Builder',
        icon: Icons.functions,
        child: RewardPanel(),
      ),
      AppPage.training => Panel(
        title: 'Training',
        icon: Icons.school_outlined,
        child: TrainingPanel(urdfPath: urdfPath),
      ),
      AppPage.evaluation => const Panel(
        title: 'Evaluation',
        icon: Icons.fact_check_outlined,
        child: EvaluationPanel(),
      ),
      AppPage.settings => const Panel(
        title: 'Settings',
        icon: Icons.settings_outlined,
        child: SettingsPanel(),
      ),
      AppPage.logs => const Panel(
        title: 'Logs',
        icon: Icons.terminal,
        child: LogsPanel(),
      ),
    };
  }
}

class _Header extends ConsumerWidget {
  const _Header({
    required this.selectedLabel,
    required this.onOpenNotifications,
    required this.onOpenCommandPalette,
  });
  final String selectedLabel;
  final VoidCallback onOpenNotifications;
  final VoidCallback onOpenCommandPalette;

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final state = ref.watch(appStateProvider);
    final center = ref.watch(notificationCenterProvider);
    final scheme = Theme.of(context).colorScheme;
    final online = state.health?['ok'] == true;
    final renderer = state.health?['renderer']?.toString();

    return Container(
      height: 50,
      padding: const EdgeInsets.symmetric(horizontal: 16),
      decoration: BoxDecoration(
        border: Border(bottom: BorderSide(color: scheme.outlineVariant)),
      ),
      child: Row(
        children: [
          const Text(
            'EasyRTG',
            style: TextStyle(fontSize: 17, fontWeight: FontWeight.w800),
          ),
          if (state.currentProjectName != null) ...[
            const SizedBox(width: 8),
            Container(
              padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 3),
              decoration: BoxDecoration(
                borderRadius: BorderRadius.circular(6),
                color: scheme.primary.withValues(alpha: 0.1),
              ),
              child: Row(
                mainAxisSize: MainAxisSize.min,
                children: [
                  Icon(Icons.folder_outlined, size: 12, color: scheme.primary),
                  const SizedBox(width: 5),
                  Text(
                    state.currentProjectName!,
                    style: TextStyle(
                      fontSize: 12,
                      fontWeight: FontWeight.w600,
                      color: scheme.primary,
                    ),
                  ),
                ],
              ),
            ),
          ],
          const SizedBox(width: 10),
          Icon(
            Icons.chevron_right,
            size: 17,
            color: scheme.onSurface.withValues(alpha: 0.4),
          ),
          const SizedBox(width: 4),
          Expanded(
            child: Text(
              selectedLabel,
              overflow: TextOverflow.ellipsis,
              style: TextStyle(
                fontWeight: FontWeight.w600,
                color: scheme.onSurface.withValues(alpha: 0.7),
              ),
            ),
          ),
          Row(
            mainAxisSize: MainAxisSize.min,
            children: [
              StatusDot(
                label: online
                    ? (renderer == null || renderer.isEmpty
                          ? 'Backend online'
                          : renderer)
                    : 'Backend offline',
                online: online,
              ),
              const SizedBox(width: 6),
              IconButton(
                tooltip: 'Command palette (Ctrl+K)',
                onPressed: onOpenCommandPalette,
                icon: const Icon(Icons.search, size: 20),
              ),
              IconButton(
                tooltip: center.unread > 0
                    ? '${center.unread} new notification(s)'
                    : 'Agent notifications',
                onPressed: onOpenNotifications,
                icon: Badge(
                  isLabelVisible: center.unread > 0,
                  label: Text('${center.unread}'),
                  child: const Icon(Icons.notifications_outlined, size: 20),
                ),
              ),
              IconButton(
                tooltip: 'Refresh backend state',
                onPressed: state.busy ? null : () => state.refreshAll(),
                icon: const Icon(Icons.refresh, size: 20),
              ),
            ],
          ),
        ],
      ),
    );
  }
}

class _StatusBar extends StatelessWidget {
  const _StatusBar({required this.message, required this.busy});
  final String message;
  final bool busy;

  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;
    return Container(
      height: 32,
      padding: const EdgeInsets.symmetric(horizontal: 12),
      decoration: BoxDecoration(
        border: Border(top: BorderSide(color: scheme.outlineVariant)),
      ),
      child: Row(
        children: [
          if (busy) ...[
            const SizedBox(
              width: 13,
              height: 13,
              child: CircularProgressIndicator(strokeWidth: 2),
            ),
            const SizedBox(width: 8),
          ] else ...[
            Icon(
              Icons.info_outline,
              size: 14,
              color: scheme.onSurface.withValues(alpha: 0.45),
            ),
            const SizedBox(width: 8),
          ],
          Expanded(
            child: Text(
              message,
              overflow: TextOverflow.ellipsis,
              style: TextStyle(
                fontSize: 12.5,
                color: scheme.onSurface.withValues(alpha: 0.8),
              ),
            ),
          ),
          CopyIconButton(
            text: message,
            tooltip: 'Copy status message',
            size: 14,
          ),
        ],
      ),
    );
  }
}
