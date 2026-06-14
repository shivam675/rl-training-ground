import 'package:fl_chart/fl_chart.dart';
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../app_state.dart';
import '../nav.dart';
import '../theme/app_theme.dart';
import '../theme/easy_colors.dart';
import '../widgets/common.dart';

class TrainingPanel extends ConsumerStatefulWidget {
  const TrainingPanel({super.key, this.urdfPath});
  final String? urdfPath;

  @override
  ConsumerState<TrainingPanel> createState() => _TrainingPanelState();
}

class _TrainingPanelState extends ConsumerState<TrainingPanel> {
  final timestepsController = TextEditingController();

  @override
  void initState() {
    super.initState();
    // Seed from the shared, persisted controls so algorithm/timesteps survive
    // navigation. Algorithm/params are read live from state in build().
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
    final algorithm = state.trainingAlgorithm;
    final trainingParams = state.trainingParams;
    final training = state.trainingStatus ?? {};
    final running = training['active'] == true;
    final statusMessage = '${training['message'] ?? 'idle'}';
    final timestep = (training['timestep'] as num?)?.toInt() ?? 0;
    final totalTimesteps = (training['total_timesteps'] as num?)?.toInt() ?? 0;
    final blockers = state.trainingBlockers();
    final canStart =
        !state.busy &&
        !running &&
        blockers.isEmpty &&
        !state.tuningActive &&
        !state.evaluationActive;

    return ListView(
      padding: const EdgeInsets.all(14),
      children: [
        if (blockers.isNotEmpty) ...[
          _SetupGateCard(blockers: blockers),
          const SizedBox(height: 14),
        ],
        if (state.tuningActive || state.evaluationActive) ...[
          Container(
            padding: const EdgeInsets.all(12),
            decoration: BoxDecoration(
              borderRadius: BorderRadius.circular(8),
              color: context.colors.info.withValues(alpha: 0.08),
              border: Border.all(
                color: context.colors.info.withValues(alpha: 0.35),
              ),
            ),
            child: Row(
              children: [
                SizedBox(
                  width: 16,
                  height: 16,
                  child: CircularProgressIndicator(
                    strokeWidth: 2,
                    color: context.colors.info,
                  ),
                ),
                const SizedBox(width: 10),
                Expanded(
                  child: Text(
                    '${state.blockingJobLabel ?? 'A job'} in progress — training '
                    'is locked until it finishes (only one job runs at a time).',
                    style: const TextStyle(fontSize: 12.5, height: 1.35),
                  ),
                ),
              ],
            ),
          ),
          const SizedBox(height: 14),
        ],
        const SectionHeader('Controls'),
        Wrap(
          spacing: 8,
          runSpacing: 8,
          crossAxisAlignment: WrapCrossAlignment.center,
          children: [
            DropdownMenu<String>(
              initialSelection: algorithm,
              label: const Text('Algorithm'),
              width: 130,
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
              width: 120,
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
              icon: const Icon(Icons.school),
              label: Text('Start $algorithm'),
            ),
            OutlinedButton.icon(
              onPressed: state.stopTraining,
              icon: const Icon(Icons.stop),
              label: const Text('Stop'),
            ),
          ],
        ),
        const SizedBox(height: 12),
        _TrainingParamsBar(
          params: trainingParams,
          onClear: () => state.setTrainingParams(const {
            'learning_rate': 0.0003,
            'batch_size': 64,
            'gamma': 0.99,
            'n_steps': 256,
          }),
        ),
        const SizedBox(height: 16),
        const SectionHeader('Status'),
        Wrap(
          spacing: 10,
          runSpacing: 10,
          children: [
            StatChip(
              label: 'Training',
              value: statusMessage,
              icon: running
                  ? Icons.play_circle_outline
                  : Icons.pause_circle_outline,
              color: running ? context.colors.success : null,
            ),
            StatChip(
              label: 'Timestep',
              value: '$timestep',
              icon: Icons.timer_outlined,
            ),
            if (training['episode_reward'] != null)
              StatChip(
                label: 'Mean reward',
                value: (training['episode_reward'] as num).toStringAsFixed(2),
                icon: Icons.emoji_events_outlined,
              ),
            if (training['fps'] != null)
              StatChip(
                label: 'FPS',
                value: '${training['fps']}',
                icon: Icons.speed,
              ),
            StatChip(
              label: 'Observation size',
              value: '${state.obsVectorSize}',
              icon: Icons.visibility_outlined,
            ),
            StatChip(
              label: 'Action size',
              value: '${state.actionVectorSize}',
              icon: Icons.gamepad_outlined,
            ),
          ],
        ),
        if (running && totalTimesteps > 0) ...[
          const SizedBox(height: 12),
          ClipRRect(
            borderRadius: BorderRadius.circular(4),
            child: LinearProgressIndicator(
              value: (timestep / totalTimesteps).clamp(0.0, 1.0),
              minHeight: 6,
            ),
          ),
          const SizedBox(height: 4),
          Text(
            '$timestep / $totalTimesteps timesteps '
            '(${(100 * timestep / totalTimesteps).toStringAsFixed(0)}%)',
            style: TextStyle(
              fontSize: 11.5,
              color: scheme.onSurface.withValues(alpha: 0.6),
            ),
          ),
        ],
        if (state.telemetry.isNotEmpty) ...[
          const SizedBox(height: 16),
          const SectionHeader('Live telemetry'),
          TelemetryCharts(points: state.telemetry),
        ],
        const SizedBox(height: 16),
        const SectionHeader('Algorithm advisor'),
        AdvisorCard(
          advisor: state.advisor,
          currentAlgorithm: algorithm,
          onPickAlgorithm: state.setTrainingAlgorithm,
        ),
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
                  content: Text('Best tuning params applied to training.'),
                ),
              );
          },
        ),
        const SizedBox(height: 16),
        const SectionHeader('Environment'),
        const _EnvSummaryStrip(),
        const SizedBox(height: 12),
        Container(
          padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 10),
          decoration: BoxDecoration(
            borderRadius: BorderRadius.circular(8),
            color: scheme.primary.withValues(alpha: 0.06),
            border: Border.all(color: scheme.primary.withValues(alpha: 0.25)),
          ),
          child: Row(
            children: [
              Icon(Icons.info_outline, size: 16, color: scheme.primary),
              const SizedBox(width: 10),
              Expanded(
                child: Text(
                  'DQN is disabled for continuous action spaces. A2C, SAC and TD3 endpoints are available in the backend API.',
                  style: TextStyle(
                    fontSize: 12.5,
                    height: 1.4,
                    color: scheme.onSurface.withValues(alpha: 0.8),
                  ),
                ),
              ),
            ],
          ),
        ),
      ],
    );
  }
}

class _TrainingParamsBar extends StatelessWidget {
  const _TrainingParamsBar({required this.params, required this.onClear});

  final Map<String, dynamic> params;
  final VoidCallback onClear;

  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;
    final summary = params.entries
        .where((entry) => entry.value != null)
        .map((entry) => '${entry.key}=${entry.value}')
        .join(' · ');
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 9),
      decoration: BoxDecoration(
        borderRadius: BorderRadius.circular(8),
        color: scheme.primary.withValues(alpha: 0.05),
        border: Border.all(color: scheme.primary.withValues(alpha: 0.22)),
      ),
      child: Row(
        children: [
          Icon(Icons.tune, size: 16, color: scheme.primary),
          const SizedBox(width: 8),
          Expanded(
            child: Text(
              summary,
              maxLines: 2,
              overflow: TextOverflow.ellipsis,
              style: monoStyle(context, fontSize: 11.5),
            ),
          ),
          TextButton.icon(
            onPressed: onClear,
            icon: const Icon(Icons.restart_alt, size: 15),
            label: const Text('Defaults'),
          ),
        ],
      ),
    );
  }
}

class _SetupGateCard extends StatelessWidget {
  const _SetupGateCard({required this.blockers});

  final List<String> blockers;

  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;
    return Container(
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        borderRadius: BorderRadius.circular(8),
        color: context.colors.warning.withValues(alpha: 0.08),
        border: Border.all(
          color: context.colors.warning.withValues(alpha: 0.35),
        ),
      ),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Icon(Icons.lock_outline, size: 18, color: context.colors.warning),
          const SizedBox(width: 10),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                const Text(
                  'Training locked',
                  style: TextStyle(fontSize: 13, fontWeight: FontWeight.w700),
                ),
                const SizedBox(height: 4),
                Text(
                  blockers.map((item) => '• $item').join('\n'),
                  style: TextStyle(
                    fontSize: 12,
                    height: 1.35,
                    color: scheme.onSurface.withValues(alpha: 0.72),
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}

class AdvisorCard extends StatelessWidget {
  const AdvisorCard({
    super.key,
    required this.advisor,
    required this.currentAlgorithm,
    required this.onPickAlgorithm,
  });

  final Map<String, dynamic>? advisor;
  final String currentAlgorithm;
  final ValueChanged<String> onPickAlgorithm;

  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;
    if (advisor == null) {
      return Text(
        'Advisor unavailable — refresh backend state.',
        style: TextStyle(
          fontSize: 12,
          color: scheme.onSurface.withValues(alpha: 0.5),
        ),
      );
    }
    final recommended = '${advisor!['recommended']}';
    final reasons = (advisor!['reasons'] as List? ?? []).cast<dynamic>();
    final presets = ((advisor!['presets'] as Map?) ?? {})
        .cast<String, dynamic>();
    final algoPresets = ((presets[currentAlgorithm] as Map?) ?? {})
        .cast<String, dynamic>();

    return Container(
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        borderRadius: BorderRadius.circular(10),
        border: Border.all(color: scheme.outlineVariant),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Icon(Icons.lightbulb_outline, size: 17, color: scheme.primary),
              const SizedBox(width: 8),
              Text(
                'Recommended: $recommended',
                style: const TextStyle(
                  fontWeight: FontWeight.w700,
                  fontSize: 13,
                ),
              ),
              const SizedBox(width: 10),
              if (recommended != currentAlgorithm)
                TextButton(
                  onPressed: () => onPickAlgorithm(recommended),
                  child: Text('Use $recommended'),
                ),
            ],
          ),
          const SizedBox(height: 6),
          for (final reason in reasons)
            Padding(
              padding: const EdgeInsets.only(left: 25, bottom: 3),
              child: Text(
                '• $reason',
                style: TextStyle(
                  fontSize: 12,
                  height: 1.35,
                  color: scheme.onSurface.withValues(alpha: 0.7),
                ),
              ),
            ),
          if (algoPresets.isNotEmpty) ...[
            const SizedBox(height: 8),
            Padding(
              padding: const EdgeInsets.only(left: 25),
              child: Wrap(
                spacing: 8,
                runSpacing: 6,
                crossAxisAlignment: WrapCrossAlignment.center,
                children: [
                  Text(
                    '$currentAlgorithm presets:',
                    style: TextStyle(
                      fontSize: 12,
                      color: scheme.onSurface.withValues(alpha: 0.6),
                    ),
                  ),
                  for (final entry in algoPresets.entries)
                    Tooltip(
                      message: entry.value.toString(),
                      child: ActionChip(
                        label: Text(
                          entry.key,
                          style: const TextStyle(fontSize: 11.5),
                        ),
                        onPressed: () => copyToClipboard(
                          context,
                          '${entry.value}',
                          label:
                              'Copied ${entry.key} preset for $currentAlgorithm',
                        ),
                      ),
                    ),
                ],
              ),
            ),
          ],
        ],
      ),
    );
  }
}

class TuningCard extends StatefulWidget {
  const TuningCard({
    super.key,
    required this.state,
    required this.algorithm,
    required this.onApplyBest,
  });

  final AppState state;
  final String algorithm;
  final ValueChanged<Map<String, dynamic>> onApplyBest;

  @override
  State<TuningCard> createState() => TuningCardState();
}

class TuningCardState extends State<TuningCard> {
  int trials = 8;

  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;
    final status = widget.state.tuningStatus ?? {};
    final active = status['active'] == true;
    final best = status['best_params'];
    final trainingActive = widget.state.trainingStatus?['active'] == true;
    final evalActive = widget.state.evaluationStatus?['active'] == true;
    final setupReady = widget.state.canStartTraining;

    return Container(
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        borderRadius: BorderRadius.circular(10),
        border: Border.all(color: scheme.outlineVariant),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Wrap(
            spacing: 10,
            runSpacing: 8,
            crossAxisAlignment: WrapCrossAlignment.center,
            children: [
              Text(
                'Trials:',
                style: TextStyle(
                  fontSize: 12.5,
                  color: scheme.onSurface.withValues(alpha: 0.7),
                ),
              ),
              SegmentedButton<int>(
                showSelectedIcon: false,
                segments: const [
                  ButtonSegment(value: 4, label: Text('4')),
                  ButtonSegment(value: 8, label: Text('8')),
                  ButtonSegment(value: 16, label: Text('16')),
                ],
                selected: {trials},
                onSelectionChanged: (sel) => setState(() => trials = sel.first),
              ),
              if (active)
                OutlinedButton.icon(
                  onPressed: widget.state.stopTuning,
                  icon: const Icon(Icons.stop, size: 16),
                  label: const Text('Stop tuning'),
                )
              else
                FilledButton.tonalIcon(
                  onPressed: trainingActive || evalActive || !setupReady
                      ? null
                      : () => widget.state.startTuning(
                          algorithm: widget.algorithm,
                          nTrials: trials,
                        ),
                  icon: const Icon(Icons.auto_fix_high, size: 16),
                  label: Text('Tune ${widget.algorithm}'),
                ),
            ],
          ),
          if (active) ...[
            const SizedBox(height: 10),
            Row(
              children: [
                const SizedBox(
                  width: 13,
                  height: 13,
                  child: CircularProgressIndicator(strokeWidth: 2),
                ),
                const SizedBox(width: 8),
                Text(
                  '${status['message'] ?? 'running'}',
                  style: const TextStyle(fontSize: 12.5),
                ),
              ],
            ),
          ],
          if (!setupReady) ...[
            const SizedBox(height: 8),
            Text(
              'Tuning unlocks after the current environment setup is saved and valid.',
              style: TextStyle(
                fontSize: 11.5,
                color: scheme.onSurface.withValues(alpha: 0.55),
              ),
            ),
          ],
          if (best != null) ...[
            const SizedBox(height: 10),
            Row(
              children: [
                Icon(
                  Icons.emoji_events,
                  size: 15,
                  color: context.colors.success,
                ),
                const SizedBox(width: 6),
                Expanded(
                  child: Text(
                    'Best score ${(status['best_value'] as num?)?.toStringAsFixed(2)} · $best',
                    maxLines: 3,
                    overflow: TextOverflow.ellipsis,
                    style: monoStyle(context, fontSize: 11.5),
                  ),
                ),
                CopyIconButton(text: '$best', tooltip: 'Copy best parameters'),
                const SizedBox(width: 4),
                FilledButton.tonalIcon(
                  onPressed: best is Map
                      ? () => widget.onApplyBest(best.cast<String, dynamic>())
                      : null,
                  icon: const Icon(Icons.input, size: 15),
                  label: const Text('Apply to training'),
                ),
              ],
            ),
          ],
          const SizedBox(height: 4),
          Text(
            'Runs short training trials with sampled hyperparameters and '
            'scores each by rollout reward. Use the best parameters for your '
            'full run.',
            style: TextStyle(
              fontSize: 11,
              color: scheme.onSurface.withValues(alpha: 0.5),
            ),
          ),
        ],
      ),
    );
  }
}

/// Live charts for the active/last training run.
class TelemetryCharts extends StatelessWidget {
  const TelemetryCharts({super.key, required this.points});

  final List<Map<String, dynamic>> points;

  List<FlSpot> _spots(String key) {
    final spots = <FlSpot>[];
    for (final point in points) {
      final x = (point['timestep'] as num?)?.toDouble();
      final y = (point[key] as num?)?.toDouble();
      if (x == null || y == null || !y.isFinite) continue;
      spots.add(FlSpot(x, y));
    }
    return spots;
  }

  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;
    final reward = _spots('reward_mean');
    final fps = _spots('fps');
    return Column(
      children: [
        if (reward.length >= 2)
          _ChartCard(
            title: 'Mean episode reward',
            color: scheme.primary,
            spots: reward,
          )
        else
          Padding(
            padding: const EdgeInsets.symmetric(vertical: 8),
            child: Text(
              'Reward curve appears after the first episodes complete…',
              style: TextStyle(
                fontSize: 12,
                color: scheme.onSurface.withValues(alpha: 0.55),
              ),
            ),
          ),
        if (fps.length >= 2) ...[
          const SizedBox(height: 10),
          _ChartCard(
            title: 'Simulation speed (steps/s)',
            color: context.colors.info,
            spots: fps,
            height: 110,
          ),
        ],
      ],
    );
  }
}

class _ChartCard extends StatelessWidget {
  const _ChartCard({
    required this.title,
    required this.color,
    required this.spots,
    this.height = 170,
  });

  final String title;
  final Color color;
  final List<FlSpot> spots;
  final double height;

  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;
    final labelStyle = TextStyle(
      fontSize: 10,
      color: scheme.onSurface.withValues(alpha: 0.5),
    );
    return Container(
      padding: const EdgeInsets.fromLTRB(10, 10, 14, 6),
      decoration: BoxDecoration(
        borderRadius: BorderRadius.circular(10),
        border: Border.all(color: scheme.outlineVariant),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Container(
                width: 8,
                height: 8,
                decoration: BoxDecoration(shape: BoxShape.circle, color: color),
              ),
              const SizedBox(width: 7),
              Text(
                title,
                style: const TextStyle(
                  fontSize: 12,
                  fontWeight: FontWeight.w600,
                ),
              ),
              const Spacer(),
              Text(
                spots.isEmpty ? '' : spots.last.y.toStringAsFixed(2),
                style: monoStyle(context, fontSize: 12, color: color),
              ),
            ],
          ),
          const SizedBox(height: 8),
          SizedBox(
            height: height,
            child: LineChart(
              duration: Duration.zero,
              LineChartData(
                gridData: FlGridData(
                  drawVerticalLine: false,
                  getDrawingHorizontalLine: (value) => FlLine(
                    color: scheme.outlineVariant.withValues(alpha: 0.5),
                    strokeWidth: 0.6,
                  ),
                ),
                titlesData: FlTitlesData(
                  topTitles: const AxisTitles(),
                  rightTitles: const AxisTitles(),
                  leftTitles: AxisTitles(
                    sideTitles: SideTitles(
                      showTitles: true,
                      reservedSize: 44,
                      getTitlesWidget: (value, meta) =>
                          Text(meta.formattedValue, style: labelStyle),
                    ),
                  ),
                  bottomTitles: AxisTitles(
                    sideTitles: SideTitles(
                      showTitles: true,
                      reservedSize: 20,
                      getTitlesWidget: (value, meta) =>
                          Text(meta.formattedValue, style: labelStyle),
                    ),
                  ),
                ),
                borderData: FlBorderData(show: false),
                lineTouchData: LineTouchData(
                  touchTooltipData: LineTouchTooltipData(
                    getTooltipItems: (touched) => [
                      for (final spot in touched)
                        LineTooltipItem(
                          '${spot.y.toStringAsFixed(2)}\n@ ${spot.x.toInt()}',
                          const TextStyle(fontSize: 11, color: Colors.white),
                        ),
                    ],
                  ),
                ),
                lineBarsData: [
                  LineChartBarData(
                    spots: spots,
                    color: color,
                    barWidth: 2,
                    isCurved: false,
                    dotData: const FlDotData(show: false),
                    belowBarData: BarAreaData(
                      show: true,
                      color: color.withValues(alpha: 0.08),
                    ),
                  ),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }
}

/// Compact one-line summary of the configured environment, linking to the
/// Build tabs. Replaces the read-only obs/action lists that duplicated those
/// pages; the effective sizes update live as sources are toggled in Build.
class _EnvSummaryStrip extends ConsumerWidget {
  const _EnvSummaryStrip();

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final state = ref.watch(appStateProvider);
    final scheme = Theme.of(context).colorScheme;
    final rewardTerms = ((state.envConfig?['rewards'] as List?) ?? [])
        .where((r) => r is Map && r['enabled'] == true)
        .length;
    void go(AppPage page) => ref.read(navIndexProvider.notifier).state = page;
    return Container(
      padding: const EdgeInsets.fromLTRB(12, 4, 6, 4),
      decoration: BoxDecoration(
        borderRadius: BorderRadius.circular(10),
        border: Border.all(color: scheme.outlineVariant),
      ),
      child: Row(
        children: [
          Icon(Icons.schema_outlined, size: 16, color: scheme.primary),
          const SizedBox(width: 8),
          Expanded(
            child: Text(
              'obs ${state.obsVectorSize}  ·  act ${state.actionVectorSize}'
              '  ·  reward $rewardTerms ${rewardTerms == 1 ? 'term' : 'terms'}',
              overflow: TextOverflow.ellipsis,
              style: monoStyle(context, fontSize: 12.5),
            ),
          ),
          TextButton(
            onPressed: () => go(AppPage.obsAction),
            child: const Text('Edit spaces'),
          ),
          TextButton(
            onPressed: () => go(AppPage.rewards),
            child: const Text('Edit reward'),
          ),
        ],
      ),
    );
  }
}
