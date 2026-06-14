import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../app_state.dart';
import '../nav.dart';
import '../theme/app_theme.dart';
import '../theme/easy_colors.dart';
import '../widgets/common.dart';

class EvaluationPanel extends ConsumerStatefulWidget {
  const EvaluationPanel({super.key});

  @override
  ConsumerState<EvaluationPanel> createState() => _EvaluationPanelState();
}

class _EvaluationPanelState extends ConsumerState<EvaluationPanel> {
  final selected = <String>{};
  int episodes = 3;

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addPostFrameCallback(
      (_) => ref.read(appStateProvider).loadRuns(),
    );
  }

  @override
  Widget build(BuildContext context) {
    final state = ref.watch(appStateProvider);
    final scheme = Theme.of(context).colorScheme;
    final evalStatus = state.evaluationStatus ?? {};
    final evalActive = evalStatus['active'] == true;
    final result = evalStatus['result'] as Map<String, dynamic>?;
    final hasProject =
        state.currentProjectId != null && state.currentProjectId!.isNotEmpty;
    final scopeLabel = state.runsShowAll || !hasProject
        ? 'All projects'
        : (state.currentProjectName ?? 'This project');

    return ListView(
      padding: const EdgeInsets.all(14),
      children: [
        SectionHeader(
          'Training runs (${state.runs.length}) · $scopeLabel',
          trailing: Row(
            mainAxisSize: MainAxisSize.min,
            children: [
              if (selected.length >= 2)
                TextButton.icon(
                  onPressed: () => _showComparison(context, state),
                  icon: const Icon(Icons.compare_arrows, size: 16),
                  label: Text('Compare (${selected.length})'),
                ),
              if (selected.isNotEmpty)
                TextButton.icon(
                  onPressed: () =>
                      _confirmAndDelete(context, state, selected.toList()),
                  icon: Icon(
                    Icons.delete_outline,
                    size: 16,
                    color: context.colors.danger,
                  ),
                  label: Text(
                    'Delete (${selected.length})',
                    style: TextStyle(color: context.colors.danger),
                  ),
                ),
              if (hasProject)
                TextButton.icon(
                  onPressed: () => state.setRunsShowAll(!state.runsShowAll),
                  icon: Icon(
                    state.runsShowAll
                        ? Icons.filter_alt_off_outlined
                        : Icons.filter_alt_outlined,
                    size: 16,
                  ),
                  label: Text(
                    state.runsShowAll ? 'All projects' : 'This project',
                  ),
                ),
              IconButton(
                tooltip: 'Refresh runs',
                visualDensity: VisualDensity.compact,
                onPressed: state.loadRuns,
                icon: const Icon(Icons.refresh, size: 18),
              ),
            ],
          ),
        ),
        if (evalActive) ...[
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 10),
            decoration: BoxDecoration(
              borderRadius: BorderRadius.circular(8),
              color: scheme.primary.withValues(alpha: 0.07),
              border: Border.all(color: scheme.primary.withValues(alpha: 0.3)),
            ),
            child: Row(
              children: [
                const SizedBox(
                  width: 14,
                  height: 14,
                  child: CircularProgressIndicator(strokeWidth: 2),
                ),
                const SizedBox(width: 10),
                Expanded(
                  child: Text(
                    'Evaluating ${evalStatus['run_name']} · '
                    '${evalStatus['message'] ?? ''}',
                    style: const TextStyle(fontSize: 12.5),
                  ),
                ),
                if (evalStatus['visualize'] == true) ...[
                  FilledButton.tonalIcon(
                    onPressed: () => ref.read(navIndexProvider.notifier).state =
                        AppPage.home,
                    icon: const Icon(Icons.smart_display_outlined, size: 16),
                    label: const Text('Watch live'),
                  ),
                  const SizedBox(width: 8),
                ],
                OutlinedButton.icon(
                  // Always actionable while an evaluation is active — a Cancel
                  // gated on the generic `busy` flag could be dead just because
                  // an unrelated request was in flight.
                  onPressed: state.stopEvaluation,
                  style: OutlinedButton.styleFrom(
                    foregroundColor: scheme.error,
                    side: BorderSide(
                      color: scheme.error.withValues(alpha: 0.5),
                    ),
                  ),
                  icon: const Icon(Icons.stop_circle_outlined, size: 16),
                  label: const Text('Cancel'),
                ),
              ],
            ),
          ),
          const SizedBox(height: 12),
        ],
        if (state.runs.isEmpty)
          EmptyState(
            icon: Icons.fact_check_outlined,
            title: state.runsShowAll || !hasProject
                ? 'No training runs yet'
                : 'No runs for this project yet',
            subtitle: state.runsShowAll || !hasProject
                ? 'Finish a training run on the Training tab and it will appear '
                      'here, ready to evaluate and compare.'
                : 'Runs you train while this project is open appear here. '
                      'Use "All projects" to see every run.',
          )
        else
          for (final run in state.runs)
            _RunCard(
              run: run,
              selected: selected.contains(run['name']),
              evalBusy:
                  evalActive || state.trainingActive || state.tuningActive,
              episodes: episodes,
              showProject: state.runsShowAll || !hasProject,
              onSelect: (value) => setState(() {
                final name = run['name'].toString();
                value ? selected.add(name) : selected.remove(name);
              }),
              onEvaluate: () => state.startEvaluation(
                run['name'].toString(),
                episodes: episodes,
              ),
              onExport: () async {
                final path = await state.exportRun(run['name'].toString());
                if (path != null && context.mounted) {
                  copyToClipboard(
                    context,
                    path,
                    label: 'Bundle created — path copied',
                  );
                }
              },
              onDelete: () =>
                  _confirmAndDelete(context, state, [run['name'].toString()]),
            ),
        const SizedBox(height: 8),
        Row(
          children: [
            Text(
              'Episodes per evaluation:',
              style: TextStyle(
                fontSize: 12.5,
                color: scheme.onSurface.withValues(alpha: 0.7),
              ),
            ),
            const SizedBox(width: 8),
            SegmentedButton<int>(
              showSelectedIcon: false,
              segments: const [
                ButtonSegment(value: 1, label: Text('1')),
                ButtonSegment(value: 3, label: Text('3')),
                ButtonSegment(value: 5, label: Text('5')),
                ButtonSegment(value: 10, label: Text('10')),
              ],
              selected: {episodes},
              onSelectionChanged: (sel) => setState(() => episodes = sel.first),
            ),
          ],
        ),
        if (result != null) ...[
          const SizedBox(height: 16),
          SectionHeader(
            'Last result · ${result['run_name'] ?? ''}',
            trailing: CopyIconButton(
              text: result.toString(),
              tooltip: 'Copy result',
            ),
          ),
          Wrap(
            spacing: 10,
            runSpacing: 10,
            children: [
              StatChip(
                label: 'Mean reward',
                value: ((result['mean_reward'] as num?) ?? 0).toStringAsFixed(
                  2,
                ),
                icon: Icons.emoji_events_outlined,
              ),
              StatChip(
                label: 'Mean length',
                value: ((result['mean_length'] as num?) ?? 0).toStringAsFixed(
                  0,
                ),
                icon: Icons.straighten,
              ),
              StatChip(
                label: 'Episodes',
                value: '${(result['episodes'] as List? ?? []).length}',
                icon: Icons.repeat,
              ),
            ],
          ),
          const SizedBox(height: 10),
          Card(
            child: SingleChildScrollView(
              scrollDirection: Axis.horizontal,
              child: DataTable(
                columns: const [
                  DataColumn(label: Text('Episode'), numeric: true),
                  DataColumn(label: Text('Reward'), numeric: true),
                  DataColumn(label: Text('Length'), numeric: true),
                ],
                rows: [
                  for (final episode in (result['episodes'] as List? ?? []))
                    DataRow(
                      cells: [
                        DataCell(Text('${episode['episode']}')),
                        DataCell(
                          Text(
                            ((episode['reward'] as num?) ?? 0).toStringAsFixed(
                              2,
                            ),
                            style: monoStyle(context, fontSize: 12),
                          ),
                        ),
                        DataCell(
                          Text(
                            '${episode['length']}',
                            style: monoStyle(context, fontSize: 12),
                          ),
                        ),
                      ],
                    ),
                ],
              ),
            ),
          ),
        ],
      ],
    );
  }

  Future<void> _confirmAndDelete(
    BuildContext context,
    AppState state,
    List<String> names,
  ) async {
    if (names.isEmpty) return;
    final ok = await showDialog<bool>(
      context: context,
      builder: (ctx) => AlertDialog(
        icon: Icon(Icons.warning_amber_rounded, color: context.colors.danger),
        title: Text(
          names.length == 1
              ? 'Delete this run?'
              : 'Delete ${names.length} runs?',
        ),
        content: const Text(
          'This permanently removes the run folder — model, telemetry, config '
          'and evaluations. This cannot be undone.',
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.of(ctx).pop(false),
            child: const Text('Cancel'),
          ),
          FilledButton(
            style: FilledButton.styleFrom(
              backgroundColor: context.colors.danger,
            ),
            onPressed: () => Navigator.of(ctx).pop(true),
            child: const Text('Delete'),
          ),
        ],
      ),
    );
    if (ok != true) return;
    for (final name in names) {
      await state.deleteRun(name);
    }
    if (mounted) setState(() => selected.removeAll(names));
  }

  Future<void> _showComparison(BuildContext context, AppState state) async {
    final names = selected.toList()..sort();
    final details = [
      for (final run in state.runs)
        if (names.contains(run['name'])) run,
    ];
    if (!context.mounted) return;
    await showDialog<void>(
      context: context,
      builder: (context) => Dialog(
        child: ConstrainedBox(
          constraints: const BoxConstraints(maxWidth: 760, maxHeight: 520),
          child: Padding(
            padding: const EdgeInsets.all(16),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              mainAxisSize: MainAxisSize.min,
              children: [
                Row(
                  children: [
                    const Expanded(
                      child: Text(
                        'Run comparison',
                        style: TextStyle(
                          fontWeight: FontWeight.w700,
                          fontSize: 15,
                        ),
                      ),
                    ),
                    IconButton(
                      onPressed: () => Navigator.of(context).pop(),
                      icon: const Icon(Icons.close, size: 18),
                    ),
                  ],
                ),
                const SizedBox(height: 8),
                Flexible(
                  child: SingleChildScrollView(
                    child: SingleChildScrollView(
                      scrollDirection: Axis.horizontal,
                      child: DataTable(
                        columns: [
                          const DataColumn(label: Text('Metric')),
                          for (final run in details)
                            DataColumn(label: Text('${run['name']}')),
                        ],
                        rows: [
                          _compareRow(
                            'Algorithm',
                            details,
                            (r) => r['algorithm'],
                          ),
                          _compareRow(
                            'Timesteps',
                            details,
                            (r) => r['total_timesteps'],
                          ),
                          _compareRow(
                            'Learning rate',
                            details,
                            (r) => r['learning_rate'],
                          ),
                          _compareRow(
                            'Best train reward',
                            details,
                            (r) => _fmt(r['reward_best']),
                          ),
                          _compareRow(
                            'Last train reward',
                            details,
                            (r) => _fmt(r['reward_last']),
                          ),
                          _compareRow(
                            'Best eval mean',
                            details,
                            (r) => _fmt(r['eval_best_mean']),
                          ),
                          _compareRow(
                            'Evaluations',
                            details,
                            (r) => r['eval_count'] ?? 0,
                          ),
                          _compareRow(
                            'Model saved',
                            details,
                            (r) => r['model_saved'] == true ? 'yes' : 'no',
                          ),
                        ],
                      ),
                    ),
                  ),
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }

  static String _fmt(dynamic value) =>
      value is num ? value.toStringAsFixed(2) : '—';

  DataRow _compareRow(
    String label,
    List<Map<String, dynamic>> runs,
    dynamic Function(Map<String, dynamic>) pick,
  ) {
    return DataRow(
      cells: [
        DataCell(
          Text(label, style: const TextStyle(fontWeight: FontWeight.w600)),
        ),
        for (final run in runs) DataCell(Text('${pick(run) ?? '—'}')),
      ],
    );
  }
}

class _RunCard extends StatelessWidget {
  const _RunCard({
    required this.run,
    required this.selected,
    required this.evalBusy,
    required this.episodes,
    required this.showProject,
    required this.onSelect,
    required this.onEvaluate,
    required this.onExport,
    required this.onDelete,
  });

  final Map<String, dynamic> run;
  final bool selected;
  final bool evalBusy;
  final int episodes;
  final bool showProject;
  final ValueChanged<bool> onSelect;
  final VoidCallback onEvaluate;
  final VoidCallback onExport;
  final VoidCallback onDelete;

  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;
    final hasModel = run['model_saved'] == true;
    return Container(
      margin: const EdgeInsets.only(bottom: 8),
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
      decoration: BoxDecoration(
        borderRadius: BorderRadius.circular(10),
        border: Border.all(
          color: selected ? scheme.primary : scheme.outlineVariant,
          width: selected ? 1.4 : 1,
        ),
      ),
      child: Row(
        children: [
          Checkbox(
            value: selected,
            visualDensity: VisualDensity.compact,
            onChanged: (value) => onSelect(value ?? false),
          ),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Row(
                  children: [
                    Flexible(
                      child: Text(
                        '${run['name']}',
                        overflow: TextOverflow.ellipsis,
                        style: monoStyle(
                          context,
                          fontSize: 12.5,
                        ).copyWith(fontWeight: FontWeight.w700),
                      ),
                    ),
                    const SizedBox(width: 8),
                    if (run['algorithm'] != null)
                      Container(
                        padding: const EdgeInsets.symmetric(
                          horizontal: 6,
                          vertical: 1,
                        ),
                        decoration: BoxDecoration(
                          borderRadius: BorderRadius.circular(5),
                          color: scheme.primary.withValues(alpha: 0.12),
                        ),
                        child: Text(
                          '${run['algorithm']}',
                          style: TextStyle(
                            fontSize: 10.5,
                            fontWeight: FontWeight.w700,
                            color: scheme.primary,
                          ),
                        ),
                      ),
                  ],
                ),
                const SizedBox(height: 2),
                Text(
                  [
                    if (showProject && run['project_name'] != null)
                      '📁 ${run['project_name']}',
                    if (run['total_timesteps'] != null)
                      '${run['total_timesteps']} steps',
                    if (run['reward_best'] != null)
                      'best ${(run['reward_best'] as num).toStringAsFixed(2)}',
                    if (run['eval_best_mean'] != null)
                      'eval ${(run['eval_best_mean'] as num).toStringAsFixed(2)}',
                    if (!hasModel) 'no model',
                  ].join(' · '),
                  style: TextStyle(
                    fontSize: 11.5,
                    color: scheme.onSurface.withValues(alpha: 0.6),
                  ),
                ),
              ],
            ),
          ),
          IconButton(
            tooltip: 'Delete run',
            visualDensity: VisualDensity.compact,
            onPressed: onDelete,
            icon: Icon(
              Icons.delete_outline,
              size: 18,
              color: context.colors.danger,
            ),
          ),
          IconButton(
            tooltip: 'Export bundle (model + config + telemetry)',
            visualDensity: VisualDensity.compact,
            onPressed: hasModel ? onExport : null,
            icon: const Icon(Icons.archive_outlined, size: 18),
          ),
          const SizedBox(width: 4),
          FilledButton.tonalIcon(
            onPressed: hasModel && !evalBusy ? onEvaluate : null,
            icon: const Icon(Icons.play_arrow, size: 16),
            label: Text('Evaluate ×$episodes'),
          ),
        ],
      ),
    );
  }
}
