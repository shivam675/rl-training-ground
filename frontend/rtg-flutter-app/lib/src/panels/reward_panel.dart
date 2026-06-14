import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../app_state.dart';
import '../theme/app_theme.dart';
import '../theme/easy_colors.dart';
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
    final lastResult = state.message.startsWith('Reward')
        ? state.message
        : null;

    String labelFor(String key) {
      for (final item in catalog) {
        if (item['key'] == key) return '${item['label']}';
      }
      return key;
    }

    return ListView(
      padding: const EdgeInsets.all(14),
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
          'Enable components and tune weights — changes persist into the '
          'training config. Test against the current robot pose any time.',
          style: TextStyle(
            fontSize: 12,
            color: scheme.onSurface.withValues(alpha: 0.6),
          ),
        ),
        const SizedBox(height: 8),
        if (lastResult != null) ...[
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 10),
            decoration: BoxDecoration(
              borderRadius: BorderRadius.circular(8),
              color: scheme.primary.withValues(alpha: 0.07),
              border: Border.all(color: scheme.primary.withValues(alpha: 0.3)),
            ),
            child: Row(
              children: [
                Icon(Icons.calculate_outlined, size: 18, color: scheme.primary),
                const SizedBox(width: 10),
                Expanded(
                  child: SelectableText(
                    lastResult,
                    style: const TextStyle(fontSize: 12.5, height: 1.4),
                  ),
                ),
                CopyIconButton(text: lastResult, tooltip: 'Copy result'),
              ],
            ),
          ),
          const SizedBox(height: 12),
        ],
        if (configured.isEmpty)
          const EmptyState(
            icon: Icons.functions,
            title: 'No reward components',
            subtitle:
                'Load a robot first — reward components are part of the '
                'environment config.',
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
    final isCustom = keyName == 'custom_python';
    final editableParams = params.entries
        .where((entry) => entry.key != 'code')
        .toList();

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
                      keyName,
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
              if (isCustom || editableParams.isNotEmpty)
                IconButton(
                  tooltip: expanded ? 'Hide parameters' : 'Edit parameters',
                  visualDensity: VisualDensity.compact,
                  onPressed: () => setState(() => expanded = !expanded),
                  icon: Icon(
                    expanded ? Icons.expand_less : Icons.tune,
                    size: 18,
                  ),
                ),
            ],
          ),
          if (expanded) ...[
            const Divider(),
            if (isCustom)
              _CustomCodeEditor(
                initialCode: '${params['code'] ?? ''}',
                state: widget.state,
                onSave: (code) => patch({
                  'params': {'code': code},
                }),
              )
            else
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

class _CustomCodeEditor extends StatefulWidget {
  const _CustomCodeEditor({
    required this.initialCode,
    required this.state,
    required this.onSave,
  });

  final String initialCode;
  final AppState state;
  final ValueChanged<String> onSave;

  @override
  State<_CustomCodeEditor> createState() => _CustomCodeEditorState();
}

class _CustomCodeEditorState extends State<_CustomCodeEditor> {
  late final controller = TextEditingController(
    text: widget.initialCode.isEmpty
        ? 'def reward(obs, action, ctx):\n'
              '    # obs: same vector the policy sees (enabled observations)\n'
              '    # ctx: base_position[x,y,z], base_orientation[x,y,z,w],\n'
              '    #      base_linear_velocity, base_angular_velocity,\n'
              '    #      joint_positions[], joint_velocities[], prev_action[], sim_time\n'
              '    # Tip: prefer the built-in toggles; only use this for goals\n'
              '    #      they cannot express, and avoid double-counting them.\n'
              '    return ctx["base_linear_velocity"][0]  # e.g. reward forward speed\n'
        : widget.initialCode,
  );
  String? validation;
  bool validating = false;

  @override
  void dispose() {
    controller.dispose();
    super.dispose();
  }

  Future<void> validate() async {
    setState(() {
      validating = true;
      validation = null;
    });
    try {
      final result = await widget.state.validateCustomReward(controller.text);
      setState(() {
        validation = result['ok'] == true
            ? '✓ Valid — sample value: ${result['value']}'
            : '✗ ${result['error']}';
      });
    } catch (e) {
      setState(() => validation = '✗ $e');
    } finally {
      if (mounted) setState(() => validating = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;
    final valid = validation?.startsWith('✓') ?? false;
    return Padding(
      padding: const EdgeInsets.only(bottom: 8),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          TextField(
            controller: controller,
            minLines: 4,
            maxLines: 12,
            style: monoStyle(context, fontSize: 12.5),
            decoration: const InputDecoration(
              labelText: 'def reward(obs, action, ctx) — runs every step',
              alignLabelWithHint: true,
            ),
          ),
          const SizedBox(height: 8),
          Row(
            children: [
              OutlinedButton.icon(
                onPressed: validating ? null : validate,
                icon: validating
                    ? const SizedBox(
                        width: 14,
                        height: 14,
                        child: CircularProgressIndicator(strokeWidth: 2),
                      )
                    : const Icon(Icons.rule, size: 16),
                label: const Text('Validate (sandboxed)'),
              ),
              const SizedBox(width: 8),
              FilledButton.icon(
                onPressed: () => widget.onSave(controller.text),
                icon: const Icon(Icons.save_outlined, size: 16),
                label: const Text('Save code'),
              ),
              const SizedBox(width: 12),
              if (validation != null)
                Expanded(
                  child: Text(
                    validation!,
                    maxLines: 2,
                    overflow: TextOverflow.ellipsis,
                    style: TextStyle(
                      fontSize: 12,
                      color: valid
                          ? context.colors.success
                          : context.colors.danger,
                    ),
                  ),
                ),
            ],
          ),
          const SizedBox(height: 4),
          Text(
            'Code is sandbox-tested before use (CPU/memory limits catch '
            'infinite loops). During training it runs in-process; failures '
            'score 0 and warn instead of crashing.',
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
