import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../app_state.dart';
import '../theme/app_theme.dart';
import '../theme/easy_colors.dart';
import '../widgets/common.dart';

class ObservationActionPanel extends ConsumerWidget {
  const ObservationActionPanel({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final state = ref.watch(appStateProvider);
    final scheme = Theme.of(context).colorScheme;
    final obs = state.observations ?? {};
    final sources = (obs['sources'] as List? ?? []).cast<dynamic>();
    final configObs = (state.envConfig?['observations'] as List? ?? [])
        .cast<dynamic>();
    final configActions = (state.envConfig?['actions'] as List? ?? [])
        .cast<dynamic>();
    final liveActions = (state.actions?['actions'] as List? ?? [])
        .cast<dynamic>();

    bool obsEnabled(String key) {
      final entry = configObs.firstWhere(
        (o) => o['key'] == key,
        orElse: () => null,
      );
      return entry == null ? false : entry['enabled'] == true;
    }

    Map<String, dynamic>? liveAction(int jointIndex) {
      for (final item in liveActions) {
        if (item['joint_index'] == jointIndex) {
          return (item as Map).cast<String, dynamic>();
        }
      }
      return null;
    }

    return ListView(
      padding: const EdgeInsets.all(14),
      children: [
        if (state.configProblems.isNotEmpty) ...[
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
            decoration: BoxDecoration(
              borderRadius: BorderRadius.circular(8),
              color: context.colors.warning.withValues(alpha: 0.08),
              border: Border.all(
                color: context.colors.warning.withValues(alpha: 0.4),
              ),
            ),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                for (final problem in state.configProblems)
                  Text(
                    '⚠ $problem',
                    style: TextStyle(
                      fontSize: 12,
                      color: context.colors.warning,
                    ),
                  ),
              ],
            ),
          ),
          const SizedBox(height: 12),
        ],
        SectionHeader(
          'Observations',
          trailing: StatChip(
            label: 'Vector size',
            value: '${state.obsVectorSize}',
            icon: Icons.visibility_outlined,
          ),
        ),
        Text(
          'Toggle which signals the policy observes. Changes persist to the '
          'environment config used for training.',
          style: TextStyle(
            fontSize: 12,
            color: scheme.onSurface.withValues(alpha: 0.6),
          ),
        ),
        const SizedBox(height: 6),
        for (final item in sources)
          SwitchListTile(
            dense: true,
            title: Text('${item['label']}'),
            subtitle: Text(
              item['placeholder'] == true
                  ? 'Not yet supported'
                  : 'Size: ${item['size']}',
              style: TextStyle(
                fontSize: 11.5,
                color: scheme.onSurface.withValues(alpha: 0.55),
              ),
            ),
            secondary: CopyIconButton(
              text: '${item['key']}',
              tooltip: 'Copy key "${item['key']}"',
              size: 14,
            ),
            value: obsEnabled('${item['key']}'),
            onChanged: item['placeholder'] == true
                ? null
                : (value) => state.patchConfig({
                    'observations': [
                      {'key': item['key'], 'enabled': value},
                    ],
                  }),
          ),
        const SizedBox(height: 16),
        SectionHeader(
          'Actions',
          trailing: StatChip(
            label: 'Vector size',
            value: '${state.actionVectorSize}',
            icon: Icons.gamepad_outlined,
          ),
        ),
        Wrap(
          spacing: 8,
          runSpacing: 8,
          children: [
            OutlinedButton.icon(
              onPressed: state.zeroAction,
              icon: const Icon(Icons.exposure_zero),
              label: const Text('Zero action'),
            ),
            OutlinedButton.icon(
              onPressed: state.randomAction,
              icon: const Icon(Icons.shuffle),
              label: const Text('Safe random'),
            ),
          ],
        ),
        const SizedBox(height: 8),
        if (configActions.isEmpty)
          const EmptyState(
            icon: Icons.gamepad_outlined,
            title: 'No actions available',
            subtitle: 'Load a robot with movable joints to configure actions.',
          )
        else
          for (final action in configActions)
            _ActionEditor(
              action: (action as Map).cast<String, dynamic>(),
              live: liveAction((action['joint_index'] as num).toInt()),
              onPatch: (patch) => state.patchConfig({
                'actions': [patch],
              }),
            ),
      ],
    );
  }
}

class _ActionEditor extends StatelessWidget {
  const _ActionEditor({
    required this.action,
    required this.live,
    required this.onPatch,
  });

  final Map<String, dynamic> action;
  final Map<String, dynamic>? live;
  final ValueChanged<Map<String, dynamic>> onPatch;

  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;
    final jointIndex = (action['joint_index'] as num).toInt();
    final enabled = action['enabled'] == true;
    final name = live?['joint_name'] ?? 'joint $jointIndex';
    return Container(
      margin: const EdgeInsets.only(bottom: 8),
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
      decoration: BoxDecoration(
        borderRadius: BorderRadius.circular(10),
        border: Border.all(color: scheme.outlineVariant),
        color: enabled ? null : scheme.onSurface.withValues(alpha: 0.03),
      ),
      child: Row(
        children: [
          Switch(
            value: enabled,
            onChanged: (value) =>
                onPatch({'joint_index': jointIndex, 'enabled': value}),
          ),
          const SizedBox(width: 8),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  '$name  (#$jointIndex)',
                  style: monoStyle(
                    context,
                    fontSize: 12.5,
                  ).copyWith(fontWeight: FontWeight.w700),
                ),
                if (live != null)
                  Text(
                    'limits ${live!['lower_limit']} … ${live!['upper_limit']} · '
                    'force ${live!['max_force']}',
                    style: TextStyle(
                      fontSize: 11,
                      color: scheme.onSurface.withValues(alpha: 0.55),
                    ),
                  ),
              ],
            ),
          ),
          DropdownButton<String>(
            value: '${action['control_mode'] ?? 'position'}',
            underline: const SizedBox.shrink(),
            style: TextStyle(fontSize: 12, color: scheme.onSurface),
            items: const [
              DropdownMenuItem(value: 'position', child: Text('position')),
              DropdownMenuItem(value: 'velocity', child: Text('velocity')),
              DropdownMenuItem(value: 'torque', child: Text('torque')),
            ],
            onChanged: enabled
                ? (value) => onPatch({
                    'joint_index': jointIndex,
                    'control_mode': value,
                  })
                : null,
          ),
          const SizedBox(width: 10),
          _ScaleField(
            label: 'low',
            value: (action['scale_low'] as num?)?.toDouble() ?? -1.0,
            enabled: enabled,
            onSubmitted: (v) =>
                onPatch({'joint_index': jointIndex, 'scale_low': v}),
          ),
          const SizedBox(width: 6),
          _ScaleField(
            label: 'high',
            value: (action['scale_high'] as num?)?.toDouble() ?? 1.0,
            enabled: enabled,
            onSubmitted: (v) =>
                onPatch({'joint_index': jointIndex, 'scale_high': v}),
          ),
        ],
      ),
    );
  }
}

class _ScaleField extends StatelessWidget {
  const _ScaleField({
    required this.label,
    required this.value,
    required this.enabled,
    required this.onSubmitted,
  });

  final String label;
  final double value;
  final bool enabled;
  final ValueChanged<double> onSubmitted;

  @override
  Widget build(BuildContext context) {
    return SizedBox(
      width: 74,
      child: TextFormField(
        key: ValueKey('$label$value'),
        initialValue: value.toString(),
        enabled: enabled,
        style: monoStyle(context, fontSize: 12),
        decoration: InputDecoration(labelText: label, isDense: true),
        onFieldSubmitted: (text) {
          final parsed = double.tryParse(text.trim());
          if (parsed != null) onSubmitted(parsed);
        },
      ),
    );
  }
}
