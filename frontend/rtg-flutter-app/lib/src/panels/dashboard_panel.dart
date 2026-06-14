import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../agent/assistant_dock.dart';
import '../agent/chat_controller.dart';
import '../app_state.dart';
import '../nav.dart';
import '../theme/easy_colors.dart';
import '../widgets/common.dart';

/// Compact dashboard summary: a one-glance robot snapshot + a setup checklist
/// that links to the relevant tabs. Replaces the full Robot Inspector on the
/// home page, whose load form and joint table duplicated the Robot Setup tab.
class DashboardPanel extends ConsumerWidget {
  const DashboardPanel({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final state = ref.watch(appStateProvider);
    final scheme = Theme.of(context).colorScheme;
    final info = state.robotInfo ?? const {};
    final hasRobot = state.hasRobot;
    final name = info['name']?.toString();
    final jointCount =
        (info['joint_count'] as num?)?.toInt() ??
        (info['joints'] as List?)?.length ??
        0;
    final path = info['source_path']?.toString() ?? info['path']?.toString();

    void go(AppPage page) => ref.read(navIndexProvider.notifier).state = page;

    final steps = <_SetupStep>[
      _SetupStep(
        label: 'Robot loaded',
        done: hasRobot,
        detail: hasRobot ? (name ?? 'loaded') : 'Load a URDF file',
        tab: AppPage.robot,
      ),
      _SetupStep(
        label: 'Observations enabled',
        done: state.hasEnabledObservations,
        detail: 'What the policy senses',
        tab: AppPage.obsAction,
      ),
      _SetupStep(
        label: 'Actions enabled',
        done: state.hasEnabledActions,
        detail: 'Joints the policy controls',
        tab: AppPage.obsAction,
      ),
      _SetupStep(
        label: 'Reward set',
        done: state.hasEnabledRewards,
        detail: 'What the robot learns',
        tab: AppPage.rewards,
      ),
      _SetupStep(
        label: 'Configuration saved',
        done: state.envConfigSaved,
        detail: 'Saved automatically as you edit',
        tab: AppPage.rewards,
      ),
    ];

    final unlocked = state.canStartTraining;
    final doneCount = steps.where((s) => s.done).length;

    void askGoal(String goal) {
      ref.read(chatControllerProvider).sendChat('Make this robot $goal.');
      setDockExpanded(ref, true);
    }

    return ListView(
      padding: const EdgeInsets.all(14),
      children: [
        const SectionHeader('Robot'),
        if (!hasRobot)
          EmptyState(
            icon: Icons.precision_manufacturing_outlined,
            title: 'No robot loaded',
            subtitle:
                'Open Robot Setup to load a URDF, or ask the assistant to help.',
            action: Column(
              mainAxisSize: MainAxisSize.min,
              children: [
                FilledButton.icon(
                  onPressed: state.busy
                      ? null
                      : () => state.loadUrdf(
                          path: 'r2d2.urdf',
                          basePosition: const [0.0, 0.0, 0.5],
                          fixedBase: false,
                          addPlane: true,
                        ),
                  icon: const Icon(Icons.bolt, size: 16),
                  label: const Text('Try a sample robot (R2D2)'),
                ),
                const SizedBox(height: 8),
                OutlinedButton.icon(
                  onPressed: () => go(AppPage.robot),
                  icon: const Icon(Icons.upload_file, size: 16),
                  label: const Text('Browse for a URDF…'),
                ),
              ],
            ),
          )
        else ...[
          Wrap(
            spacing: 10,
            runSpacing: 10,
            children: [
              StatChip(
                label: 'Name',
                value: name ?? 'robot',
                icon: Icons.smart_toy_outlined,
              ),
              StatChip(
                label: 'Joints',
                value: '$jointCount',
                icon: Icons.device_hub,
              ),
            ],
          ),
          if (path != null && path.isNotEmpty) ...[
            const SizedBox(height: 10),
            CopyableValue(value: path),
          ],
        ],
        const SizedBox(height: 16),
        SectionHeader(
          'Setup checklist',
          trailing: Row(
            mainAxisSize: MainAxisSize.min,
            children: [
              SizedBox(
                width: 16,
                height: 16,
                child: CircularProgressIndicator(
                  value: doneCount / steps.length,
                  strokeWidth: 2.5,
                  color: unlocked ? context.colors.success : scheme.primary,
                  backgroundColor: scheme.outlineVariant,
                ),
              ),
              const SizedBox(width: 6),
              Text(
                '$doneCount/${steps.length}',
                style: TextStyle(
                  fontSize: 11.5,
                  fontWeight: FontWeight.w700,
                  color: scheme.onSurface.withValues(alpha: 0.6),
                ),
              ),
              const SizedBox(width: 2),
              IconButton(
                tooltip: 'Ask the assistant',
                visualDensity: VisualDensity.compact,
                iconSize: 18,
                onPressed: () => setDockExpanded(ref, true),
                icon: const Icon(Icons.smart_toy_outlined),
              ),
            ],
          ),
        ),
        _UnlockBanner(unlocked: unlocked, onTrain: () => go(AppPage.training)),
        const SizedBox(height: 8),
        for (final step in steps)
          _SetupRow(step: step, onTap: () => go(step.tab)),
        if (state.configProblems.isNotEmpty) ...[
          const SizedBox(height: 12),
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
        ],
        const SizedBox(height: 14),
        if (hasRobot) ...[
          Text(
            'Tell the assistant what to learn:',
            style: TextStyle(
              fontSize: 12,
              fontWeight: FontWeight.w600,
              color: scheme.onSurface.withValues(alpha: 0.7),
            ),
          ),
          const SizedBox(height: 8),
          Wrap(
            spacing: 8,
            runSpacing: 8,
            children: [
              for (final goal in const [
                'walk',
                'balance',
                'stand',
                'reach a target',
              ])
                ActionChip(
                  avatar: Icon(
                    Icons.auto_awesome,
                    size: 15,
                    color: scheme.primary,
                  ),
                  label: Text(goal[0].toUpperCase() + goal.substring(1)),
                  onPressed: () => askGoal(goal),
                ),
            ],
          ),
        ] else
          Text(
            'Or tell the assistant what the robot should learn — it drafts the '
            'observations, actions and reward for you, all editable before you '
            'train.',
            style: TextStyle(
              fontSize: 11.5,
              height: 1.4,
              color: scheme.onSurface.withValues(alpha: 0.55),
            ),
          ),
      ],
    );
  }
}

class _SetupStep {
  const _SetupStep({
    required this.label,
    required this.done,
    required this.detail,
    required this.tab,
  });

  final String label;
  final bool done;
  final String detail;
  final AppPage tab;
}

class _SetupRow extends StatelessWidget {
  const _SetupRow({required this.step, required this.onTap});

  final _SetupStep step;
  final VoidCallback onTap;

  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;
    final color = step.done ? context.colors.success : scheme.onSurface;
    return InkWell(
      borderRadius: BorderRadius.circular(8),
      onTap: onTap,
      child: Padding(
        padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 7),
        child: Row(
          children: [
            Icon(
              step.done ? Icons.check_circle : Icons.radio_button_unchecked,
              size: 18,
              color: step.done
                  ? color
                  : scheme.onSurface.withValues(alpha: 0.35),
            ),
            const SizedBox(width: 10),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    step.label,
                    style: TextStyle(
                      fontSize: 13,
                      fontWeight: FontWeight.w600,
                      color: step.done
                          ? scheme.onSurface
                          : scheme.onSurface.withValues(alpha: 0.85),
                    ),
                  ),
                  Text(
                    step.detail,
                    maxLines: 1,
                    overflow: TextOverflow.ellipsis,
                    style: TextStyle(
                      fontSize: 11,
                      color: scheme.onSurface.withValues(alpha: 0.5),
                    ),
                  ),
                ],
              ),
            ),
            Icon(
              Icons.chevron_right,
              size: 18,
              color: scheme.onSurface.withValues(alpha: 0.35),
            ),
          ],
        ),
      ),
    );
  }
}

class _UnlockBanner extends StatelessWidget {
  const _UnlockBanner({required this.unlocked, required this.onTrain});

  final bool unlocked;
  final VoidCallback onTrain;

  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;
    final color = unlocked ? context.colors.success : scheme.primary;
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 10),
      decoration: BoxDecoration(
        borderRadius: BorderRadius.circular(10),
        color: color.withValues(alpha: 0.08),
        border: Border.all(color: color.withValues(alpha: 0.35)),
      ),
      child: Row(
        children: [
          Icon(
            unlocked ? Icons.lock_open_rounded : Icons.lock_outline,
            size: 18,
            color: color,
          ),
          const SizedBox(width: 10),
          Expanded(
            child: Text(
              unlocked
                  ? 'Setup complete — Training and Evaluation are unlocked.'
                  : 'Complete the checklist to unlock Training and Evaluation.',
              style: TextStyle(
                fontSize: 12.5,
                fontWeight: FontWeight.w600,
                color: color,
              ),
            ),
          ),
          if (unlocked) ...[
            const SizedBox(width: 8),
            FilledButton.icon(
              onPressed: onTrain,
              icon: const Icon(Icons.school, size: 15),
              label: const Text('Train'),
            ),
          ],
        ],
      ),
    );
  }
}
