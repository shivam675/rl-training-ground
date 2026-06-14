import 'package:file_picker/file_picker.dart';
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../app_state.dart';
import '../theme/app_theme.dart';
import '../theme/easy_colors.dart';
import '../widgets/common.dart';

class RobotPanel extends ConsumerStatefulWidget {
  const RobotPanel({super.key, this.urdfPath});
  final String? urdfPath;

  @override
  ConsumerState<RobotPanel> createState() => _RobotPanelState();
}

class _RobotPanelState extends ConsumerState<RobotPanel> {
  final pathController = TextEditingController();
  final gravityController = TextEditingController(text: '-9.81');
  bool fixedBase = false;
  bool addPlane = true;

  @override
  void dispose() {
    pathController.dispose();
    gravityController.dispose();
    super.dispose();
  }

  String jointsAsCsv(List<dynamic> joints) {
    final buffer = StringBuffer('index,name,type,lower_limit,upper_limit\n');
    for (final item in joints) {
      buffer.writeln(
        '${item['index']},${item['name']},${item['type']},'
        '${item['lower_limit']},${item['upper_limit']}',
      );
    }
    return buffer.toString();
  }

  @override
  Widget build(BuildContext context) {
    final state = ref.watch(appStateProvider);
    final scheme = Theme.of(context).colorScheme;
    final info = state.robotInfo ?? {};
    final joints = (info['joints'] as List? ?? []).cast<dynamic>();
    final warnings = (info['warnings'] as List? ?? []).cast<dynamic>();
    final hasRobot = info['name'] != null;
    final loadedPath =
        info['source_path']?.toString() ?? info['path']?.toString();

    return ListView(
      padding: const EdgeInsets.all(14),
      children: [
        const SectionHeader('Load model'),
        Row(
          children: [
            Expanded(
              child: TextField(
                controller: pathController,
                style: monoStyle(context, fontSize: 13),
                decoration: const InputDecoration(
                  labelText: 'URDF path',
                  prefixIcon: Icon(Icons.description_outlined, size: 18),
                ),
                onSubmitted: (_) => _load(state),
              ),
            ),
            const SizedBox(width: 8),
            IconButton.filledTonal(
              tooltip: 'Browse for a .urdf file',
              icon: const Icon(Icons.folder_open),
              onPressed: () async {
                final result = await FilePicker.platform.pickFiles(
                  type: FileType.custom,
                  allowedExtensions: ['urdf'],
                );
                final path = result?.files.single.path;
                if (path != null) setState(() => pathController.text = path);
              },
            ),
            const SizedBox(width: 8),
            FilledButton.icon(
              onPressed: state.busy ? null : () => _load(state),
              icon: const Icon(Icons.play_arrow),
              label: const Text('Load'),
            ),
          ],
        ),
        const SizedBox(height: 12),
        Wrap(
          spacing: 10,
          runSpacing: 10,
          crossAxisAlignment: WrapCrossAlignment.center,
          children: [
            FilterChip(
              label: const Text('Fixed base'),
              tooltip: 'Anchor the robot base so it cannot move',
              selected: fixedBase,
              onSelected: (v) => setState(() => fixedBase = v),
            ),
            FilterChip(
              label: const Text('Ground plane'),
              tooltip: 'Add a flat ground plane to the world',
              selected: addPlane,
              onSelected: (v) => setState(() => addPlane = v),
            ),
            SizedBox(
              width: 130,
              child: TextField(
                controller: gravityController,
                style: monoStyle(context, fontSize: 13),
                decoration: const InputDecoration(
                  labelText: 'Gravity Z (m/s²)',
                ),
              ),
            ),
            OutlinedButton.icon(
              onPressed: () => state.setGravity(
                double.tryParse(gravityController.text) ?? -9.81,
              ),
              icon: const Icon(Icons.south, size: 16),
              label: const Text('Apply gravity'),
            ),
            OutlinedButton.icon(
              onPressed: state.resetSimulation,
              icon: const Icon(Icons.restart_alt, size: 16),
              label: const Text('Reset world'),
            ),
          ],
        ),
        const SizedBox(height: 18),
        if (!hasRobot)
          const EmptyState(
            icon: Icons.precision_manufacturing_outlined,
            title: 'No robot loaded yet',
            subtitle:
                'Pick a URDF file above and press Load to spawn it in the simulation.',
          )
        else ...[
          const SectionHeader('Robot'),
          Wrap(
            spacing: 10,
            runSpacing: 10,
            children: [
              StatChip(
                label: 'Name',
                value: '${info['name']}',
                icon: Icons.smart_toy_outlined,
              ),
              StatChip(
                label: 'Joints',
                value: '${info['joint_count'] ?? joints.length}',
                icon: Icons.device_hub,
              ),
            ],
          ),
          if (loadedPath != null && loadedPath.isNotEmpty) ...[
            const SizedBox(height: 10),
            CopyableValue(value: loadedPath),
          ],
          if (state.robotDynamicsHasIssues) ...[
            const SizedBox(height: 12),
            Builder(
              builder: (context) {
                final dyn = state.robotDynamics ?? {};
                final hasErrors = ((dyn['error_count'] as num?) ?? 0) > 0;
                final accent = hasErrors
                    ? scheme.error
                    : context.colors.warning;
                return Container(
                  padding: const EdgeInsets.symmetric(
                    horizontal: 12,
                    vertical: 10,
                  ),
                  decoration: BoxDecoration(
                    borderRadius: BorderRadius.circular(8),
                    color: accent.withValues(alpha: 0.08),
                    border: Border.all(color: accent.withValues(alpha: 0.4)),
                  ),
                  child: Row(
                    children: [
                      Icon(
                        hasErrors
                            ? Icons.error_outline
                            : Icons.warning_amber_rounded,
                        size: 18,
                        color: accent,
                      ),
                      const SizedBox(width: 10),
                      Expanded(
                        child: Text(
                          '${dyn['summary'] ?? 'Robot physics issues detected.'} '
                          'Auto-fix clamps masses and rebuilds bad inertia.',
                          style: const TextStyle(fontSize: 12.5, height: 1.35),
                        ),
                      ),
                      const SizedBox(width: 10),
                      FilledButton.icon(
                        onPressed: state.busy ? null : state.fixRobotDynamics,
                        icon: const Icon(Icons.healing, size: 16),
                        label: const Text('Auto-fix dynamics'),
                      ),
                    ],
                  ),
                );
              },
            ),
          ],
          if (warnings.isNotEmpty) ...[
            const SizedBox(height: 12),
            for (final warning in warnings)
              Padding(
                padding: const EdgeInsets.only(bottom: 6),
                child: Container(
                  padding: const EdgeInsets.symmetric(
                    horizontal: 12,
                    vertical: 8,
                  ),
                  decoration: BoxDecoration(
                    borderRadius: BorderRadius.circular(8),
                    color: context.colors.warning.withValues(alpha: 0.08),
                    border: Border.all(
                      color: context.colors.warning.withValues(alpha: 0.4),
                    ),
                  ),
                  child: Row(
                    children: [
                      Icon(
                        Icons.warning_amber_rounded,
                        size: 16,
                        color: context.colors.warning,
                      ),
                      const SizedBox(width: 8),
                      Expanded(
                        child: Text(
                          '$warning',
                          style: TextStyle(
                            fontSize: 12.5,
                            color: context.colors.warning,
                          ),
                        ),
                      ),
                      CopyIconButton(text: '$warning', tooltip: 'Copy warning'),
                    ],
                  ),
                ),
              ),
          ],
          const SizedBox(height: 14),
          SectionHeader(
            'Joints (${joints.length})',
            trailing: joints.isEmpty
                ? null
                : TextButton.icon(
                    onPressed: () => copyToClipboard(
                      context,
                      jointsAsCsv(joints),
                      label: 'Copied ${joints.length} joints as CSV',
                    ),
                    icon: const Icon(Icons.table_view, size: 15),
                    label: const Text('Copy as CSV'),
                  ),
          ),
          if (joints.isEmpty)
            Text(
              'This robot exposes no joints.',
              style: TextStyle(color: scheme.onSurface.withValues(alpha: 0.6)),
            )
          else
            Card(
              child: SingleChildScrollView(
                scrollDirection: Axis.horizontal,
                child: DataTable(
                  showCheckboxColumn: false,
                  columns: const [
                    DataColumn(label: Text('#'), numeric: true),
                    DataColumn(label: Text('Joint')),
                    DataColumn(label: Text('Type')),
                    DataColumn(label: Text('Limits')),
                    DataColumn(label: Text('')),
                  ],
                  rows: [
                    for (final item in joints)
                      DataRow(
                        cells: [
                          DataCell(
                            Text(
                              '${item['index']}',
                              style: monoStyle(context, fontSize: 12),
                            ),
                          ),
                          DataCell(
                            Text(
                              '${item['name']}',
                              style: monoStyle(context, fontSize: 12),
                            ),
                          ),
                          DataCell(_JointTypeBadge(type: '${item['type']}')),
                          DataCell(
                            Text(
                              '${item['lower_limit']} … ${item['upper_limit']}',
                              style: monoStyle(context, fontSize: 12),
                            ),
                          ),
                          DataCell(
                            CopyIconButton(
                              text: '${item['name']}',
                              tooltip: 'Copy joint name',
                              size: 14,
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

  void _load(AppState state) {
    state.loadUrdf(
      path: pathController.text.trim(),
      basePosition: const [0.0, 0.0, 0.5],
      fixedBase: fixedBase,
      addPlane: addPlane,
    );
  }
}

class _JointTypeBadge extends StatelessWidget {
  const _JointTypeBadge({required this.type});

  final String type;

  @override
  Widget build(BuildContext context) {
    final color = switch (type.toLowerCase()) {
      'revolute' => const Color(0xff4f9cff),
      'prismatic' => const Color(0xff9a7bff),
      'continuous' => const Color(0xff5fe089),
      'fixed' => const Color(0xff8a9bb0),
      _ => Theme.of(context).colorScheme.primary,
    };
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 3),
      decoration: BoxDecoration(
        borderRadius: BorderRadius.circular(6),
        color: color.withValues(alpha: 0.12),
        border: Border.all(color: color.withValues(alpha: 0.4)),
      ),
      child: Text(
        type,
        style: TextStyle(
          fontSize: 11,
          fontWeight: FontWeight.w600,
          color: color,
        ),
      ),
    );
  }
}
