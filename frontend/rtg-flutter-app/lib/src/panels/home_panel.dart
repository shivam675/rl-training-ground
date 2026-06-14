import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../app_state.dart';
import '../nav.dart';
import '../theme/app_theme.dart';
import '../theme/easy_colors.dart';
import '../widgets/common.dart';
import 'dashboard_panel.dart';
import 'simulation_panel.dart';
import 'training_panel.dart';

/// Home cockpit: a large live viewport beside a status/control sidebar. The
/// chat moved to the co-pilot dock; the 4-quadrant grid is gone. While setup is
/// incomplete the Training panel collapses to a thin bar so the setup checklist
/// gets the room; once training unlocks it smoothly opens into the full control.
class HomePanel extends StatelessWidget {
  const HomePanel({super.key, this.urdfPath});

  final String? urdfPath;

  @override
  Widget build(BuildContext context) {
    return Row(
      children: [
        const Expanded(
          flex: 5,
          child: Panel(
            title: '3D Simulation',
            icon: Icons.view_in_ar,
            child: SimulationViewport(),
          ),
        ),
        Expanded(flex: 3, child: _HomeSidebar(urdfPath: urdfPath)),
      ],
    );
  }
}

/// Setup checklist + Training, where Training's height animates with the setup
/// state: a thin bar until the environment is valid, then it opens.
class _HomeSidebar extends ConsumerStatefulWidget {
  const _HomeSidebar({required this.urdfPath});

  final String? urdfPath;

  @override
  ConsumerState<_HomeSidebar> createState() => _HomeSidebarState();
}

class _HomeSidebarState extends ConsumerState<_HomeSidebar>
    with SingleTickerProviderStateMixin {
  static const _collapsed = 58.0;
  static const _expandedFraction = 0.55;

  late final AnimationController _controller = AnimationController(
    vsync: this,
    duration: const Duration(milliseconds: 360),
    value: ref.read(appStateProvider).canStartTraining ? 1.0 : 0.0,
  );

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final unlocked = ref.watch(appStateProvider).canStartTraining;
    // Drive the open/close animation toward the current setup state. forward()/
    // reverse() are no-ops once at the target, so calling every build is safe.
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (!mounted) return;
      if (unlocked) {
        _controller.forward();
      } else {
        _controller.reverse();
      }
    });

    return LayoutBuilder(
      builder: (context, constraints) {
        final height = constraints.maxHeight;
        return AnimatedBuilder(
          animation: _controller,
          builder: (context, _) {
            final t = Curves.easeOutCubic.transform(_controller.value);
            final trainingH =
                _collapsed + (height * _expandedFraction - _collapsed) * t;
            final setupH = (height - trainingH).clamp(0.0, height);
            return Column(
              children: [
                SizedBox(
                  height: setupH,
                  child: const Panel(
                    title: 'Setup & Status',
                    icon: Icons.checklist_rounded,
                    child: DashboardPanel(),
                  ),
                ),
                SizedBox(
                  height: trainingH,
                  child: Panel(
                    title: 'Training',
                    icon: Icons.school_outlined,
                    child: TrainingControlCard(urdfPath: widget.urdfPath),
                  ),
                ),
              ],
            );
          },
        );
      },
    );
  }
}

/// Training controls for the Home sidebar. Locked: a short "finish setup" hint.
/// Unlocked: the recommended algorithm, auto-tuning (with apply wired straight
/// into the next run), start/stop, and the live mean-reward chart.
class TrainingControlCard extends ConsumerStatefulWidget {
  const TrainingControlCard({super.key, this.urdfPath});

  final String? urdfPath;

  @override
  ConsumerState<TrainingControlCard> createState() =>
      _TrainingControlCardState();
}

class _TrainingControlCardState extends ConsumerState<TrainingControlCard> {
  final timestepsController = TextEditingController();

  @override
  void initState() {
    super.initState();
    timestepsController.text = ref
        .read(appStateProvider)
        .trainingTimesteps
        .toString();
  }

  @override
  void dispose() {
    timestepsController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final state = ref.watch(appStateProvider);
    final scheme = Theme.of(context).colorScheme;
    final colors = context.colors;
    final algorithm = state.trainingAlgorithm;
    final trainingParams = state.trainingParams;
    final training = state.trainingStatus ?? {};
    final running = training['active'] == true;
    final timestep = (training['timestep'] as num?)?.toInt() ?? 0;
    final total = (training['total_timesteps'] as num?)?.toInt() ?? 0;
    final reward = (training['episode_reward'] as num?)?.toDouble();
    final blockers = state.trainingBlockers();
    final hasTelemetry = state.telemetry.length >= 2;

    void go(AppPage page) => ref.read(navIndexProvider.notifier).state = page;

    if (blockers.isNotEmpty) {
      return Padding(
        padding: const EdgeInsets.all(14),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Container(
              padding: const EdgeInsets.all(12),
              decoration: BoxDecoration(
                borderRadius: BorderRadius.circular(10),
                color: colors.warning.withValues(alpha: 0.08),
                border: Border.all(
                  color: colors.warning.withValues(alpha: 0.35),
                ),
              ),
              child: Row(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Icon(Icons.lock_outline, size: 18, color: colors.warning),
                  const SizedBox(width: 10),
                  Expanded(
                    child: Text(
                      'Finish setup to start training. Next: ${blockers.first}.',
                      style: const TextStyle(fontSize: 12.5, height: 1.35),
                    ),
                  ),
                ],
              ),
            ),
          ],
        ),
      );
    }

    final canStart =
        !state.busy &&
        !running &&
        !state.tuningActive &&
        !state.evaluationActive;
    return ListView(
      padding: const EdgeInsets.all(14),
      children: [
        Row(
          children: [
            Icon(
              running ? Icons.play_circle_outline : Icons.check_circle_outline,
              size: 18,
              color: running ? colors.success : scheme.primary,
            ),
            const SizedBox(width: 8),
            Expanded(
              child: Text(
                running
                    ? 'Training in progress'
                    : (hasTelemetry ? 'Last run complete' : 'Ready to train'),
                style: const TextStyle(
                  fontSize: 13,
                  fontWeight: FontWeight.w700,
                ),
              ),
            ),
            TextButton.icon(
              onPressed: () => go(AppPage.training),
              icon: const Icon(Icons.open_in_full, size: 14),
              label: const Text('Full controls'),
            ),
          ],
        ),
        const SizedBox(height: 8),
        // Strongest-algorithm suggestion.
        AdvisorCard(
          advisor: state.advisor,
          currentAlgorithm: algorithm,
          onPickAlgorithm: state.setTrainingAlgorithm,
        ),
        const SizedBox(height: 12),
        Wrap(
          spacing: 8,
          runSpacing: 8,
          crossAxisAlignment: WrapCrossAlignment.center,
          children: [
            DropdownMenu<String>(
              initialSelection: algorithm,
              label: const Text('Algorithm'),
              width: 120,
              inputDecorationTheme: Theme.of(context).inputDecorationTheme,
              dropdownMenuEntries: const [
                DropdownMenuEntry(value: 'PPO', label: 'PPO'),
                DropdownMenuEntry(value: 'SAC', label: 'SAC'),
                DropdownMenuEntry(value: 'TD3', label: 'TD3'),
                DropdownMenuEntry(value: 'A2C', label: 'A2C'),
              ],
              onSelected: (value) =>
                  state.setTrainingAlgorithm(value ?? algorithm),
            ),
            SizedBox(
              width: 110,
              child: TextField(
                controller: timestepsController,
                keyboardType: TextInputType.number,
                style: monoStyle(context, fontSize: 13),
                decoration: const InputDecoration(labelText: 'Timesteps'),
                onChanged: (value) => state.setTrainingTimesteps(
                  int.tryParse(value.trim()) ?? state.trainingTimesteps,
                ),
              ),
            ),
            if (running)
              OutlinedButton.icon(
                onPressed: state.stopTraining,
                icon: const Icon(Icons.stop, size: 16),
                label: const Text('Stop'),
              )
            else
              FilledButton.icon(
                onPressed: canStart
                    ? () => state.startTraining(
                        widget.urdfPath,
                        algorithm: algorithm,
                        totalTimesteps:
                            int.tryParse(timestepsController.text.trim()) ??
                            10000,
                        hyperparams: trainingParams,
                      )
                    : null,
                icon: const Icon(Icons.school, size: 16),
                label: Text('Start $algorithm'),
              ),
          ],
        ),
        if (state.tuningActive)
          Padding(
            padding: const EdgeInsets.only(top: 8),
            child: Text(
              'Tuning in progress — training starts when it finishes.',
              style: TextStyle(
                fontSize: 11.5,
                color: scheme.onSurface.withValues(alpha: 0.6),
              ),
            ),
          ),
        const SizedBox(height: 12),
        Wrap(
          spacing: 10,
          runSpacing: 10,
          children: [
            StatChip(
              label: 'Timestep',
              value: total > 0 ? '$timestep / $total' : '$timestep',
              icon: Icons.timer_outlined,
            ),
            if (reward != null)
              StatChip(
                label: 'Mean reward',
                value: reward.toStringAsFixed(2),
                icon: Icons.emoji_events_outlined,
                color: colors.success,
              ),
          ],
        ),
        if (running && total > 0) ...[
          const SizedBox(height: 12),
          ClipRRect(
            borderRadius: BorderRadius.circular(4),
            child: LinearProgressIndicator(
              value: (timestep / total).clamp(0.0, 1.0),
              minHeight: 6,
            ),
          ),
        ],
        const SizedBox(height: 16),
        const SectionHeader('Auto-tune (Optuna)'),
        TuningCard(
          state: state,
          algorithm: algorithm,
          onApplyBest: (params) {
            state.setTrainingParams({...state.trainingParams, ...params});
            ScaffoldMessenger.of(context)
              ..clearSnackBars()
              ..showSnackBar(
                const SnackBar(
                  duration: Duration(milliseconds: 1600),
                  content: Text('Best tuning params applied to the next run.'),
                ),
              );
          },
        ),
        if (hasTelemetry) ...[
          const SizedBox(height: 16),
          const SectionHeader('Live telemetry'),
          TelemetryCharts(points: state.telemetry),
        ],
        if (!running && hasTelemetry) ...[
          const SizedBox(height: 12),
          OutlinedButton.icon(
            onPressed: () => go(AppPage.evaluation),
            icon: const Icon(Icons.fact_check_outlined, size: 16),
            label: const Text('Evaluate this run'),
          ),
        ],
      ],
    );
  }
}
