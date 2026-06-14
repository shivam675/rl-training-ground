import 'package:flutter/material.dart';

/// One entry in the command palette.
class PaletteCommand {
  const PaletteCommand({
    required this.label,
    required this.icon,
    required this.run,
    this.hint,
  });

  final String label;
  final IconData icon;
  final VoidCallback run;

  /// Short category shown on the right (e.g. "Go to", "Project").
  final String? hint;
}

/// A `Ctrl+K` command palette: filter pages and actions, run with a click or
/// Enter. One keyboard-driven surface for navigation + project actions.
Future<void> showCommandPalette(
  BuildContext context,
  List<PaletteCommand> commands,
) {
  return showDialog<void>(
    context: context,
    barrierColor: Colors.black.withValues(alpha: 0.35),
    builder: (_) => _CommandPalette(commands: commands),
  );
}

class _CommandPalette extends StatefulWidget {
  const _CommandPalette({required this.commands});

  final List<PaletteCommand> commands;

  @override
  State<_CommandPalette> createState() => _CommandPaletteState();
}

class _CommandPaletteState extends State<_CommandPalette> {
  final controller = TextEditingController();
  String _query = '';

  @override
  void dispose() {
    controller.dispose();
    super.dispose();
  }

  List<PaletteCommand> get _filtered {
    final q = _query.trim().toLowerCase();
    if (q.isEmpty) return widget.commands;
    return widget.commands
        .where(
          (c) =>
              c.label.toLowerCase().contains(q) ||
              (c.hint?.toLowerCase().contains(q) ?? false),
        )
        .toList();
  }

  void _run(PaletteCommand command) {
    Navigator.of(context).pop();
    command.run();
  }

  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;
    final filtered = _filtered;
    return Dialog(
      alignment: Alignment.topCenter,
      insetPadding: const EdgeInsets.only(top: 90, left: 20, right: 20),
      child: ConstrainedBox(
        constraints: const BoxConstraints(maxWidth: 560, maxHeight: 460),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Padding(
              padding: const EdgeInsets.fromLTRB(12, 12, 12, 8),
              child: TextField(
                controller: controller,
                autofocus: true,
                decoration: const InputDecoration(
                  prefixIcon: Icon(Icons.search, size: 18),
                  hintText: 'Jump to a page or run an action…',
                ),
                onChanged: (value) => setState(() => _query = value),
                onSubmitted: (_) {
                  if (filtered.isNotEmpty) _run(filtered.first);
                },
              ),
            ),
            const Divider(height: 1),
            Flexible(
              child: filtered.isEmpty
                  ? Padding(
                      padding: const EdgeInsets.all(24),
                      child: Text(
                        'No matching commands',
                        style: TextStyle(
                          color: scheme.onSurface.withValues(alpha: 0.6),
                        ),
                      ),
                    )
                  : ListView.builder(
                      shrinkWrap: true,
                      padding: const EdgeInsets.symmetric(vertical: 6),
                      itemCount: filtered.length,
                      itemBuilder: (context, index) {
                        final command = filtered[index];
                        return ListTile(
                          dense: true,
                          leading: Icon(
                            command.icon,
                            size: 19,
                            color: scheme.primary,
                          ),
                          title: Text(command.label),
                          trailing: command.hint == null
                              ? null
                              : Text(
                                  command.hint!,
                                  style: TextStyle(
                                    fontSize: 11,
                                    color: scheme.onSurface.withValues(
                                      alpha: 0.5,
                                    ),
                                  ),
                                ),
                          onTap: () => _run(command),
                        );
                      },
                    ),
            ),
          ],
        ),
      ),
    );
  }
}
