import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../agent/chat_view.dart';
import '../app_state.dart';
import '../nav.dart';
import '../theme/app_theme.dart';
import '../theme/easy_colors.dart';
import '../theme/tokens.dart';
import '../widgets/common.dart';
import 'dashboard_panel.dart';
import 'training_panel.dart';

/// Home cockpit: the assistant is the main canvas, with the setup/status +
/// training controls in a side column. The 3D viewport moved to its own page
/// (entered during Isaac training live-view / evaluation) — there is no live
/// PyBullet viewport on Home anymore. While setup is incomplete the Training
/// panel collapses to a thin bar so the setup checklist gets the room; once
/// training unlocks it smoothly opens into the full control.
class HomePanel extends StatelessWidget {
  const HomePanel({super.key, this.urdfPath});

  final String? urdfPath;

  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;
    return Row(
      children: [
        const Expanded(flex: 5, child: AssistantChat()),
        VerticalDivider(width: 1, color: scheme.outlineVariant),
        Expanded(flex: 3, child: _HomeSidebar(urdfPath: urdfPath)),
      ],
    );
  }
}

/// Setup checklist + Training, each exactly as tall as its own content, in one
/// scrolling column.
///
/// This replaced an `AnimationController` that interpolated the two panels
/// between a fixed collapsed height and 55% of the viewport. Every fixed height
/// here was a guess about content it could not see, and the guess was wrong in
/// both directions: too much while Training was locked (a void under the goal
/// chips) and too little for the checklist once the "No robot loaded" empty
/// state was on screen, which clipped a checklist row in half. Content sizing
/// gets the collapse-while-locked behaviour for free — a locked Training panel
/// IS just its one-line banner — so the controller, the two magic fractions and
/// the ticker mixin all went with it.
class _HomeSidebar extends StatelessWidget {
  const _HomeSidebar({required this.urdfPath});

  final String? urdfPath;

  @override
  Widget build(BuildContext context) {
    // Scrolls as ONE column. Each Panel is content-sized, so the pair can
    // exceed the viewport (a live telemetry chart plus a long warning list) and
    // the user scrolls past both instead of hunting for which of two nested
    // scrollers has the row they want.
    return SingleChildScrollView(
      child: Column(
        children: [
          const Panel(
            title: 'Setup & Status',
            icon: Icons.checklist_rounded,
            child: DashboardPanel(),
          ),
          Panel(
            title: 'Training',
            icon: Icons.school_outlined,
            child: TrainingControlCard(urdfPath: urdfPath),
          ),
        ],
      ),
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
      // shrinkWrap + no physics: _HomeSidebar owns the scroll now, so this list
      // reports its content height and hands the gesture up instead of nesting
      // a second scroller inside the first.
      return ListView(
        shrinkWrap: true,
        physics: const NeverScrollableScrollPhysics(),
        padding: const EdgeInsets.all(Insets.page),
        children: [
          Container(
            padding: const EdgeInsets.all(12),
            decoration: BoxDecoration(
              borderRadius: BorderRadius.circular(Radii.md),
              color: colors.warning.withValues(alpha: 0.08),
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
          if (state.configProblems.isNotEmpty) ...[
            const SizedBox(height: 10),
            Align(
              alignment: Alignment.centerLeft,
              child: TextButton.icon(
                onPressed: () => go(AppPage.rewards),
                icon: const Icon(Icons.arrow_forward, size: 15),
                label: const Text('Review the reward'),
              ),
            ),
          ],
        ],
      );
    }

    final canStart =
        !state.busy &&
        !running &&
        !state.tuningActive &&
        !state.evaluationActive;
    return ListView(
      shrinkWrap: true,
      physics: const NeverScrollableScrollPhysics(),
      padding: const EdgeInsets.all(Insets.page),
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
                    ? () async {
                        final live = state.trainingLiveView;
                        await state.startTraining(
                          widget.urdfPath,
                          algorithm: algorithm,
                          totalTimesteps:
                              int.tryParse(timestepsController.text.trim()) ??
                              10000,
                          hyperparams: trainingParams,
                        );
                        if (live && context.mounted) go(AppPage.viewer);
                      }
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
