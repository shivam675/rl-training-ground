import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../notifications.dart';
import '../theme/easy_colors.dart';
import 'common.dart';

Color severityColor(BuildContext context, String severity) {
  final colors = context.colors;
  return switch (severity) {
    'success' => colors.success,
    'warning' => colors.warning,
    'error' => colors.danger,
    _ => colors.info,
  };
}

IconData severityIcon(String severity) => switch (severity) {
  'success' => Icons.check_circle_outline,
  'warning' => Icons.warning_amber_rounded,
  'error' => Icons.error_outline,
  _ => Icons.info_outline,
};

class NotificationDrawer extends ConsumerWidget {
  const NotificationDrawer({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final center = ref.watch(notificationCenterProvider);
    final scheme = Theme.of(context).colorScheme;
    return Drawer(
      width: 400,
      child: SafeArea(
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            Padding(
              padding: const EdgeInsets.fromLTRB(16, 12, 8, 8),
              child: Row(
                children: [
                  Icon(
                    Icons.notifications_outlined,
                    size: 19,
                    color: scheme.primary,
                  ),
                  const SizedBox(width: 8),
                  const Expanded(
                    child: Text(
                      'Agent notifications',
                      style: TextStyle(
                        fontWeight: FontWeight.w700,
                        fontSize: 14.5,
                      ),
                    ),
                  ),
                  IconButton(
                    tooltip: 'Clear all',
                    visualDensity: VisualDensity.compact,
                    onPressed: center.notifications.isEmpty
                        ? null
                        : center.clear,
                    icon: const Icon(Icons.delete_sweep_outlined, size: 18),
                  ),
                  IconButton(
                    tooltip: 'Close',
                    visualDensity: VisualDensity.compact,
                    onPressed: () => Navigator.of(context).pop(),
                    icon: const Icon(Icons.close, size: 18),
                  ),
                ],
              ),
            ),
            const Divider(),
            Expanded(
              child: center.notifications.isEmpty
                  ? const EmptyState(
                      icon: Icons.notifications_none,
                      title: 'No notifications yet',
                      subtitle:
                          'The agent will post updates here: robot loads, '
                          'training progress, results and suggested next steps.',
                    )
                  : ListView.builder(
                      padding: const EdgeInsets.all(10),
                      itemCount: center.notifications.length,
                      itemBuilder: (context, index) => _NotificationCard(
                        notification: center.notifications[index],
                      ),
                    ),
            ),
          ],
        ),
      ),
    );
  }
}

class _NotificationCard extends StatelessWidget {
  const _NotificationCard({required this.notification});

  final AgentNotification notification;

  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;
    final color = severityColor(context, notification.severity);
    final time = notification.time;
    final timeLabel =
        '${time.hour.toString().padLeft(2, '0')}:${time.minute.toString().padLeft(2, '0')}';
    return Container(
      margin: const EdgeInsets.only(bottom: 8),
      padding: const EdgeInsets.fromLTRB(12, 10, 6, 10),
      decoration: BoxDecoration(
        borderRadius: BorderRadius.circular(10),
        color: color.withValues(alpha: 0.05),
        border: Border.all(color: color.withValues(alpha: 0.3)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Icon(severityIcon(notification.severity), size: 16, color: color),
              const SizedBox(width: 8),
              Expanded(
                child: Text(
                  notification.title,
                  style: const TextStyle(
                    fontWeight: FontWeight.w700,
                    fontSize: 13,
                  ),
                ),
              ),
              Text(
                timeLabel,
                style: TextStyle(
                  fontSize: 10.5,
                  color: scheme.onSurface.withValues(alpha: 0.45),
                ),
              ),
              CopyIconButton(
                text: notification.asText(),
                tooltip: 'Copy notification',
                size: 13,
              ),
            ],
          ),
          if (notification.body.isNotEmpty) ...[
            const SizedBox(height: 4),
            Padding(
              padding: const EdgeInsets.only(left: 24),
              child: SelectableText(
                notification.body,
                style: TextStyle(
                  fontSize: 12.5,
                  height: 1.4,
                  color: scheme.onSurface.withValues(alpha: 0.8),
                ),
              ),
            ),
          ],
          if (notification.nextSteps.isNotEmpty) ...[
            const SizedBox(height: 6),
            Padding(
              padding: const EdgeInsets.only(left: 24),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    'NEXT STEPS',
                    style: TextStyle(
                      fontSize: 10,
                      fontWeight: FontWeight.w700,
                      letterSpacing: 1,
                      color: color,
                    ),
                  ),
                  const SizedBox(height: 3),
                  for (final step in notification.nextSteps)
                    Padding(
                      padding: const EdgeInsets.only(bottom: 2),
                      child: Row(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Text(
                            '•  ',
                            style: TextStyle(fontSize: 12, color: color),
                          ),
                          Expanded(
                            child: Text(
                              step,
                              style: TextStyle(
                                fontSize: 12,
                                height: 1.35,
                                color: scheme.onSurface.withValues(alpha: 0.75),
                              ),
                            ),
                          ),
                        ],
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
