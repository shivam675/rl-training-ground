import 'package:file_picker/file_picker.dart';
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../app_state.dart';
import '../theme/app_theme.dart';
import '../theme/easy_colors.dart';
import '../theme/tokens.dart';
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

    // Two columns, and progressive disclosure inside them. Loading a robot is
    // one action; everything else here is reference material that used to be
    // dumped on the user at once — 45 joint rows, world physics knobs and a
    // warning list, before they had done anything.
    //
    // Left: identity. Who loaded, from where, and how to swap it. Answers "is
    // this the right robot?" without scrolling.
    // Right: everything you might drill into, each behind a disclosure.
    if (hasRobot) {
      return LayoutBuilder(
        builder: (context, constraints) {
          final identity = _identityColumn(state, info, joints, loadedPath);
          final detail = _detailChildren(state, scheme, joints, warnings);
          // Below this the two columns are narrower than their own content, so
          // stack them instead. 920 is roughly where the joint table stops
          // fitting beside the identity column.
          if (constraints.maxWidth < 920) {
            return ListView(
              padding: const EdgeInsets.all(Insets.page),
              children: [
                identity,
                const SizedBox(height: Insets.xl),
                ...detail,
              ],
            );
          }
          return Row(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              SizedBox(
                // 380, not 340: three identity tiles across 340 minus the page
                // gutter gave each one 97px, which is under what "MOVABLE" and
                // its icon need. The path line below reads better wider too.
                width: 380,
                child: SingleChildScrollView(
                  padding: const EdgeInsets.all(Insets.page),
                  child: identity,
                ),
              ),
              VerticalDivider(width: 1, color: scheme.outlineVariant),
              Expanded(
                child: ListView(
                  padding: const EdgeInsets.all(Insets.page),
                  children: detail,
                ),
              ),
            ],
          );
        },
      );
    }

    // Nothing loaded: one full-width invitation, no columns to fill.
    return ListView(
      padding: const EdgeInsets.all(Insets.page),
      children: [
        EmptyState(
          icon: Icons.precision_manufacturing_outlined,
          title: 'Load a robot',
          subtitle:
              'Pick a URDF file. Meshes and package:// paths are resolved '
              'automatically.',
          action: Wrap(
            spacing: 10,
            runSpacing: 10,
            alignment: WrapAlignment.center,
            children: [
              FilledButton.icon(
                onPressed: state.busy ? null : _browseThenLoad,
                icon: const Icon(Icons.folder_open, size: 16),
                label: const Text('Browse files'),
              ),
              OutlinedButton.icon(
                onPressed: state.busy
                    ? null
                    : () {
                        pathController.text = kSampleRobotPath;
                        _load(state);
                      },
                icon: const Icon(Icons.bolt, size: 16),
                label: const Text('Sample robot ($kSampleRobotName)'),
              ),
            ],
          ),
        ),
        const SizedBox(height: Insets.lg),
        _DisclosureGroup(
          children: [
            _Disclosure(
              title: 'Load from a path',
              subtitle: 'Type or paste a .urdf location',
              child: _loadForm(state),
            ),
          ],
        ),
      ],
    );
  }

  int _movableCount(List<dynamic> joints) =>
      joints.where((j) => '${j['type']}'.toLowerCase() != 'fixed').length;

  /// Left column: who is loaded, from where, and how to swap it.
  Widget _identityColumn(
    AppState state,
    Map<dynamic, dynamic> info,
    List<dynamic> joints,
    String? loadedPath,
  ) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      mainAxisSize: MainAxisSize.min,
      children: [
        // Row of equal thirds, not a Wrap. Three tiles want 327px and this
        // column offers 292, so Wrap dropped MOVABLE onto a second line —
        // which read as a layout accident, because it is one. Expanded thirds
        // stay on one line at any column width; StatChip ellipsizes.
        Row(
          children: [
            for (final chip in [
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
              // The number that actually matters for training: how many joints
              // a policy could drive. go1 reports 45 joints and only 12 move.
              StatChip(
                label: 'Movable',
                value: '${_movableCount(joints)}',
                icon: Icons.settings_input_component,
              ),
            ].indexed)
              Expanded(
                child: Padding(
                  padding: EdgeInsets.only(left: chip.$1 == 0 ? 0 : Insets.sm),
                  child: chip.$2,
                ),
              ),
          ],
        ),
        if (loadedPath != null && loadedPath.isNotEmpty) ...[
          const SizedBox(height: Insets.md),
          CopyableValue(value: loadedPath, maxLines: 2),
        ],
        const SizedBox(height: Insets.lg),
        OutlinedButton.icon(
          onPressed: state.busy ? null : _browseThenLoad,
          icon: const Icon(Icons.swap_horiz, size: 16),
          label: const Text('Load a different robot'),
        ),
      ],
    );
  }

  /// Right column: the advisory banner, then everything drillable.
  List<Widget> _detailChildren(
    AppState state,
    ColorScheme scheme,
    List<dynamic> joints,
    List<dynamic> warnings,
  ) {
    return [
      if (state.robotDynamicsHasIssues) ...[
        _dynamicsBanner(state, scheme),
        const SizedBox(height: Insets.lg),
      ],
      _DisclosureGroup(
        children: [
          if (joints.isNotEmpty)
            _Disclosure(
              title: 'Joints',
              subtitle: '${joints.length} in the URDF · limits and types',
              child: _jointsTable(joints),
            ),
          if (warnings.isNotEmpty)
            _Disclosure(
              title: 'URDF warnings',
              subtitle:
                  '${warnings.length} advisory — the robot loaded regardless',
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  for (final warning in warnings)
                    Padding(
                      padding: const EdgeInsets.only(bottom: 6),
                      child: Row(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Icon(
                            Icons.warning_amber_rounded,
                            size: 15,
                            color: context.colors.warning,
                          ),
                          const SizedBox(width: 8),
                          Expanded(
                            child: Text(
                              '$warning',
                              style: const TextStyle(
                                fontSize: 12.5,
                                height: 1.35,
                              ),
                            ),
                          ),
                        ],
                      ),
                    ),
                ],
              ),
            ),
          _Disclosure(
            title: 'World',
            subtitle: 'Base anchoring, ground plane, gravity',
            child: _worldForm(state),
          ),
          _Disclosure(
            title: 'Load from a path',
            subtitle: 'Type or paste a .urdf location',
            child: _loadForm(state),
          ),
        ],
      ),
    ];
  }

  Widget _dynamicsBanner(AppState state, ColorScheme scheme) {
    final dyn = state.robotDynamics ?? {};
    final hasErrors = ((dyn['error_count'] as num?) ?? 0) > 0;
    final accent = hasErrors ? scheme.error : context.colors.warning;
    final summary = dyn['summary'] ?? 'Robot physics issues detected.';
    return Container(
      padding: const EdgeInsets.symmetric(
        horizontal: Insets.lg,
        vertical: Insets.md,
      ),
      decoration: BoxDecoration(
        borderRadius: BorderRadius.circular(Radii.md),
        color: accent.withValues(alpha: 0.08),
      ),
      child: Row(
        children: [
          Icon(
            hasErrors ? Icons.error_outline : Icons.warning_amber_rounded,
            size: 18,
            color: accent,
          ),
          const SizedBox(width: Insets.md),
          Expanded(
            child: Text(
              state.robotDynamicsFixable
                  ? '$summary Auto-fix clamps masses and rebuilds bad inertia.'
                  : '$summary Advisory only — these need a URDF edit, not '
                        'auto-fixable.',
              style: const TextStyle(fontSize: 12.5, height: 1.35),
            ),
          ),
          if (state.robotDynamicsFixable) ...[
            const SizedBox(width: Insets.md),
            FilledButton.icon(
              onPressed: state.busy ? null : state.fixRobotDynamics,
              icon: const Icon(Icons.healing, size: 16),
              label: const Text('Auto-fix dynamics'),
            ),
          ],
        ],
      ),
    );
  }

  Widget _jointsTable(List<dynamic> joints) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Align(
          alignment: Alignment.centerRight,
          child: TextButton.icon(
            onPressed: () => copyToClipboard(
              context,
              jointsAsCsv(joints),
              label: 'Copied ${joints.length} joints as CSV',
            ),
            icon: const Icon(Icons.table_view, size: 15),
            label: const Text('Copy as CSV'),
          ),
        ),
        const SizedBox(height: Insets.sm),
        SingleChildScrollView(
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
      ],
    );
  }

  /// Input left, controls right — the same split as [_worldForm].
  Widget _loadForm(AppState state) {
    return _InputsThenControls(
      inputs: TextField(
        controller: pathController,
        style: monoStyle(context, fontSize: 13),
        decoration: const InputDecoration(
          labelText: 'URDF path',
          prefixIcon: Icon(Icons.description_outlined, size: 18),
        ),
        onSubmitted: (_) => _load(state),
      ),
      // The field is the whole point of this form, so let it take the slack
      // instead of sizing it and leaving a gap.
      inputsFlex: true,
      controls: [
        IconButton.filledTonal(
          tooltip: 'Browse for a .urdf file',
          icon: const Icon(Icons.folder_open),
          onPressed: _browse,
        ),
        FilledButton.icon(
          onPressed: state.busy ? null : () => _load(state),
          icon: const Icon(Icons.play_arrow),
          label: const Text('Load'),
        ),
      ],
    );
  }

  Widget _worldForm(AppState state) {
    return _InputsThenControls(
      inputs: SizedBox(
        width: 150,
        child: TextField(
          controller: gravityController,
          style: monoStyle(context, fontSize: 13),
          decoration: const InputDecoration(labelText: 'Gravity Z (m/s²)'),
        ),
      ),
      controls: [
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
    );
  }

  Future<String?> _browse() async {
    final result = await FilePicker.platform.pickFiles(
      type: FileType.custom,
      allowedExtensions: ['urdf'],
    );
    final path = result?.files.single.path;
    if (path != null && mounted) setState(() => pathController.text = path);
    return path;
  }

  Future<void> _browseThenLoad() async {
    // One gesture, not two: picking a file and then hunting for a Load button
    // is the step everyone forgets.
    if (await _browse() != null && mounted) {
      _load(ref.read(appStateProvider));
    }
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

/// One form row: everything you type on the left, everything you click on the
/// right. Chips, toggles and buttons were previously interleaved with the
/// fields in a single [Wrap], so "Gravity Z" sat between two toggles and a
/// button — nothing lined up between the two disclosures.
class _InputsThenControls extends StatelessWidget {
  const _InputsThenControls({
    required this.inputs,
    required this.controls,
    this.inputsFlex = false,
  });

  final Widget inputs;
  final List<Widget> controls;

  /// Let the input soak up the free width instead of the gap doing it.
  final bool inputsFlex;

  @override
  Widget build(BuildContext context) {
    // Wrap for the controls, not a Row: the disclosure is 340px narrower on a
    // stacked layout and four buttons must be able to take a second line
    // rather than overflow. WrapAlignment.end keeps them on the right edge.
    final trailing = Wrap(
      alignment: WrapAlignment.end,
      spacing: Insets.sm,
      runSpacing: Insets.sm,
      crossAxisAlignment: WrapCrossAlignment.center,
      children: controls,
    );
    return Row(
      crossAxisAlignment: CrossAxisAlignment.center,
      children: [
        if (inputsFlex) Expanded(child: inputs) else inputs,
        const SizedBox(width: Insets.lg),
        if (inputsFlex) trailing else Expanded(child: trailing),
      ],
    );
  }
}

/// Stacks disclosures into one surface with hairlines between them, so four
/// collapsed sections read as a single list rather than four floating cards.
class _DisclosureGroup extends StatelessWidget {
  const _DisclosureGroup({required this.children});

  final List<Widget> children;

  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;
    return DecoratedBox(
      decoration: BoxDecoration(
        color: scheme.surfaceContainerLow,
        borderRadius: BorderRadius.circular(Radii.md),
        boxShadow: Shadows.sm,
      ),
      child: Column(
        children: [
          for (var i = 0; i < children.length; i++) ...[
            if (i > 0) Divider(height: 1, color: scheme.outlineVariant),
            Padding(
              padding: const EdgeInsets.symmetric(horizontal: Insets.lg),
              child: children[i],
            ),
          ],
        ],
      ),
    );
  }
}

/// A collapsed section. Reference material lives behind one of these so the
/// page opens with what the user needs, not everything the backend knows.
class _Disclosure extends StatelessWidget {
  const _Disclosure({required this.title, required this.child, this.subtitle});

  final String title;
  final String? subtitle;
  final Widget child;

  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;
    return Theme(
      // ExpansionTile draws its own divider lines; the surface is shadow-only.
      data: Theme.of(context).copyWith(dividerColor: Colors.transparent),
      child: ExpansionTile(
        tilePadding: const EdgeInsets.symmetric(vertical: Insets.xs),
        // `top` is the fix: with bottom-only padding the expanded content
        // butted straight up against the title/subtitle, so World's toggles and
        // the URDF field looked welded to the heading that opened them.
        childrenPadding: const EdgeInsets.only(
          top: Insets.md,
          bottom: Insets.lg,
        ),
        expandedCrossAxisAlignment: CrossAxisAlignment.start,
        shape: const Border(),
        collapsedShape: const Border(),
        title: Text(
          title,
          style: const TextStyle(fontSize: 13.5, fontWeight: FontWeight.w600),
        ),
        subtitle: subtitle == null
            ? null
            : Text(
                subtitle!,
                style: TextStyle(
                  fontSize: 11.5,
                  color: scheme.onSurfaceVariant,
                ),
              ),
        children: [child],
      ),
    );
  }
}

class _JointTypeBadge extends StatelessWidget {
  const _JointTypeBadge({required this.type});

  final String type;

  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;
    // One distinction actually matters here — can the policy drive this joint
    // or not. Four hues for four URDF spellings was decoration, and D4 reserves
    // colour for meaning. Weight carries it instead.
    final movable = type.toLowerCase() != 'fixed';
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 3),
      decoration: BoxDecoration(
        borderRadius: BorderRadius.circular(6),
        color: scheme.surfaceContainerHighest,
      ),
      child: Text(
        type,
        style: TextStyle(
          fontSize: 11,
          fontWeight: movable ? FontWeight.w600 : FontWeight.w400,
          color: movable ? scheme.onSurface : scheme.onSurfaceVariant,
        ),
      ),
    );
  }
}
