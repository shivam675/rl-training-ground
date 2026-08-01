import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../app_state.dart';
import '../theme/app_theme.dart';
import '../theme/easy_colors.dart';
import '../theme/tokens.dart';
import '../widgets/common.dart';

class RewardPanel extends ConsumerWidget {
  const RewardPanel({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final state = ref.watch(appStateProvider);
    final scheme = Theme.of(context).colorScheme;
    final catalog = (state.observations?['reward_components'] as List? ?? [])
        .cast<dynamic>();
    final configured = (state.envConfig?['rewards'] as List? ?? [])
        .cast<dynamic>();
    final lastResult = state.lastRewardResult;

    String labelFor(String key) {
      for (final item in catalog) {
        if (item['key'] == key) return '${item['label']}';
      }
      return key;
    }

    return ListView(
      padding: const EdgeInsets.all(Insets.page),
      children: [
        SectionHeader(
          'Reward components',
          trailing: FilledButton.icon(
            onPressed: state.busy ? null : state.testReward,
            icon: const Icon(Icons.functions, size: 16),
            label: const Text('Test reward'),
          ),
        ),
        Text(
          'The reward the agent wrote for this robot and goal. Every term is an '
          'expression over the robot state — edit the expression or the weight '
          'and it persists into the training config. Test against the current '
          'robot pose any time.',
          style: TextStyle(
            fontSize: 12,
            color: scheme.onSurface.withValues(alpha: 0.6),
          ),
        ),
        const SizedBox(height: 8),
        if (lastResult != null) ...[
          _RewardResultCard(result: lastResult),
          const SizedBox(height: 12),
        ],
        if (configured.isEmpty)
          const EmptyState(
            icon: Icons.functions,
            title: 'No reward function yet',
            subtitle:
                'Tell the agent in chat what you want this robot to do. It '
                'writes a reward function for that goal and this robot, and it '
                'appears here — fully editable.',
          )
        else
          for (final component in configured)
            _RewardEditor(
              component: (component as Map).cast<String, dynamic>(),
              label: labelFor('${component['key']}'),
              state: state,
            ),
      ],
    );
  }
}

class _RewardResultCard extends StatelessWidget {
  const _RewardResultCard({required this.result});

  final Map<String, dynamic> result;

  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;
    final warnings = (result['warnings'] as List? ?? [])
        .map((item) => item.toString())
        .toList();
    final terms = (result['terms'] as List? ?? [])
        .whereType<Map>()
        .map((item) => item.cast<String, dynamic>())
        .toList();
    final reward = (result['reward'] as num?)?.toDouble() ?? 0.0;
    final formula = result['formula']?.toString() ?? '0.0';
    return Container(
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        borderRadius: BorderRadius.circular(8),
        color: scheme.primary.withValues(alpha: 0.07),
        border: Border.all(color: scheme.primary.withValues(alpha: 0.3)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Icon(Icons.calculate_outlined, size: 18, color: scheme.primary),
              const SizedBox(width: 8),
              Text(
                'Reward ${reward.toStringAsFixed(4)}',
                style: const TextStyle(
                  fontSize: 13,
                  fontWeight: FontWeight.w700,
                ),
              ),
              const Spacer(),
              CopyIconButton(text: result.toString(), tooltip: 'Copy result'),
            ],
          ),
          const SizedBox(height: 6),
          SelectableText(formula, style: monoStyle(context, fontSize: 11.5)),
          if (warnings.isNotEmpty) ...[
            const SizedBox(height: 8),
            for (final warning in warnings)
              Text(
                'Review: $warning',
                style: TextStyle(fontSize: 12, color: context.colors.warning),
              ),
          ],
          if (terms.isNotEmpty) ...[
            const SizedBox(height: 8),
            SingleChildScrollView(
              scrollDirection: Axis.horizontal,
              child: DataTable(
                headingRowHeight: 28,
                dataRowMinHeight: 28,
                dataRowMaxHeight: 32,
                columns: const [
                  DataColumn(label: Text('Term')),
                  DataColumn(label: Text('Raw'), numeric: true),
                  DataColumn(label: Text('Weight'), numeric: true),
                  DataColumn(label: Text('Value'), numeric: true),
                ],
                rows: [
                  for (final term in terms)
                    DataRow(
                      cells: [
                        DataCell(Text('${term['key']}')),
                        DataCell(Text(_fmt(term['raw']))),
                        DataCell(Text(_fmt(term['weight']))),
                        DataCell(Text(_fmt(term['value']))),
                      ],
                    ),
                ],
              ),
            ),
          ],
        ],
      ),
    );
  }

  static String _fmt(dynamic value) {
    final number = value is num ? value.toDouble() : double.tryParse('$value');
    return number == null ? '$value' : number.toStringAsFixed(4);
  }
}

class _RewardEditor extends StatefulWidget {
  const _RewardEditor({
    required this.component,
    required this.label,
    required this.state,
  });

  final Map<String, dynamic> component;
  final String label;
  final AppState state;

  @override
  State<_RewardEditor> createState() => _RewardEditorState();
}

class _RewardEditorState extends State<_RewardEditor> {
  bool expanded = false;

  Map<String, dynamic> get params =>
      ((widget.component['params'] as Map?) ?? {}).cast<String, dynamic>();

  String get keyName => '${widget.component['key']}';

  void patch(Map<String, dynamic> changes) {
    widget.state.patchConfig({
      'rewards': [
        {'key': keyName, ...changes},
      ],
    });
  }

  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;
    final enabled = widget.component['enabled'] == true;
    final weight = (widget.component['weight'] as num?)?.toDouble() ?? 1.0;
    final expr = '${params['expr'] ?? ''}';
    // Only the handful of PyBullet-only components carry other params; an
    // authored term carries `expr` and nothing else.
    final editableParams = params.entries
        .where((entry) => entry.key != 'expr')
        .toList();
    final feedback = widget.state.feedbackForReward(keyName);

    return Container(
      margin: const EdgeInsets.only(bottom: 8),
      padding: const EdgeInsets.fromLTRB(12, 6, 12, 6),
      decoration: BoxDecoration(
        borderRadius: BorderRadius.circular(10),
        border: Border.all(color: scheme.outlineVariant),
        color: enabled ? null : scheme.onSurface.withValues(alpha: 0.03),
      ),
      child: Column(
        // Without this the expanded params/code editor shrink-wraps and the
        // default center alignment floats it into the middle of the row.
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          Row(
            children: [
              Switch(
                value: enabled,
                onChanged: (value) => patch({'enabled': value}),
              ),
              const SizedBox(width: 8),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      widget.label,
                      style: const TextStyle(
                        fontSize: 13,
                        fontWeight: FontWeight.w600,
                      ),
                    ),
                    Text(
                      // The expression IS the term, so show it collapsed rather
                      // than making the key stand in for a meaning it no longer
                      // carries (the key is only a telemetry label now).
                      expr.isEmpty ? keyName : expr,
                      maxLines: 1,
                      overflow: TextOverflow.ellipsis,
                      style: monoStyle(
                        context,
                        fontSize: 10.5,
                        color: scheme.onSurface.withValues(alpha: 0.5),
                      ),
                    ),
                  ],
                ),
              ),
              SizedBox(
                width: 90,
                child: TextFormField(
                  key: ValueKey('w$weight'),
                  initialValue: weight.toString(),
                  enabled: enabled,
                  style: monoStyle(context, fontSize: 12),
                  decoration: const InputDecoration(
                    labelText: 'weight',
                    isDense: true,
                  ),
                  onFieldSubmitted: (text) {
                    final parsed = double.tryParse(text.trim());
                    if (parsed != null) patch({'weight': parsed});
                  },
                ),
              ),
              IconButton(
                tooltip: expanded ? 'Hide expression' : 'Edit expression',
                visualDensity: VisualDensity.compact,
                onPressed: () => setState(() => expanded = !expanded),
                icon: Icon(expanded ? Icons.expand_less : Icons.tune, size: 18),
              ),
            ],
          ),
          if (feedback.isNotEmpty)
            Padding(
              padding: const EdgeInsets.only(left: 60, bottom: 6),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  for (final item in feedback)
                    Text(
                      item,
                      style: TextStyle(
                        fontSize: 11.5,
                        color: context.colors.warning,
                      ),
                    ),
                ],
              ),
            ),
          if (expanded) ...[
            const Divider(),
            _ExpressionEditor(
              initialExpr: expr,
              enabled: enabled,
              onSave: (text) => patch({
                'params': {'expr': text},
              }),
            ),
            if (editableParams.isNotEmpty)
              Padding(
                padding: const EdgeInsets.only(bottom: 8),
                child: Wrap(
                  spacing: 10,
                  runSpacing: 10,
                  children: [
                    for (final entry in editableParams)
                      _ParamField(
                        name: entry.key,
                        value: entry.value,
                        enabled: enabled,
                        onSubmitted: (value) => patch({
                          'params': {entry.key: value},
                        }),
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

class _ParamField extends StatelessWidget {
  const _ParamField({
    required this.name,
    required this.value,
    required this.enabled,
    required this.onSubmitted,
  });

  final String name;
  final dynamic value;
  final bool enabled;
  final ValueChanged<dynamic> onSubmitted;

  @override
  Widget build(BuildContext context) {
    final isList = value is List;
    final display = isList ? (value as List).join(', ') : '$value';
    return SizedBox(
      width: isList ? 200 : 130,
      child: TextFormField(
        key: ValueKey('$name$display'),
        initialValue: display,
        enabled: enabled,
        style: monoStyle(context, fontSize: 12),
        decoration: InputDecoration(
          labelText: name,
          isDense: true,
          helperText: isList ? 'comma-separated' : null,
          helperStyle: const TextStyle(fontSize: 10),
        ),
        onFieldSubmitted: (text) {
          if (isList) {
            final parts = text
                .split(',')
                .map((s) => num.tryParse(s.trim()))
                .whereType<num>()
                .toList();
            onSubmitted(parts);
          } else {
            onSubmitted(num.tryParse(text.trim()) ?? text.trim());
          }
        },
      ),
    );
  }
}

class _ExpressionEditor extends StatefulWidget {
  const _ExpressionEditor({
    required this.initialExpr,
    required this.enabled,
    required this.onSave,
  });

  final String initialExpr;
  final bool enabled;
  final ValueChanged<String> onSave;

  @override
  State<_ExpressionEditor> createState() => _ExpressionEditorState();
}

class _ExpressionEditorState extends State<_ExpressionEditor> {
  late final controller = TextEditingController(text: widget.initialExpr);

  @override
  void dispose() {
    controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;
    return Padding(
      padding: const EdgeInsets.only(bottom: 8),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          TextField(
            controller: controller,
            enabled: widget.enabled,
            minLines: 2,
            maxLines: 6,
            style: monoStyle(context, fontSize: 12.5),
            decoration: const InputDecoration(
              labelText: 'expression — evaluated every step, times the weight',
              alignLabelWithHint: true,
            ),
          ),
          const SizedBox(height: 8),
          Row(
            children: [
              FilledButton.icon(
                onPressed: widget.enabled
                    ? () => widget.onSave(controller.text.trim())
                    : null,
                icon: const Icon(Icons.save_outlined, size: 16),
                label: const Text('Save expression'),
              ),
            ],
          ),
          const SizedBox(height: 4),
          Text(
            'State: base_x/y/z, base_vx/vy/vz, speed, roll_rate, pitch_rate, '
            'yaw_rate, upright, height_ratio, episode_progress, joint_pos, '
            'joint_vel, joint_limit_use, action, prev_action, and on a robot '
            'with feet contact, contact_force, air_time, contact_time, '
            'foot_speed. Functions: abs min max clamp mean sum norm sqrt exp '
            'log tanh sign step where. Reduce per-joint and per-limb values '
            'with mean(), not sum() — sum scales with how many DOFs the robot '
            'has, so the same weight means something else on another robot. '
            'Saving re-checks the expression and audits the weights; anything '
            'wrong appears next to the term.',
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
