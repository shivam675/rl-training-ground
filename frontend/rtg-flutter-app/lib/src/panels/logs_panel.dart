import 'dart:async';
import 'dart:convert';

import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../app_state.dart';
import '../theme/app_theme.dart';
import '../widgets/common.dart';

class LogsPanel extends ConsumerStatefulWidget {
  const LogsPanel({super.key});

  @override
  ConsumerState<LogsPanel> createState() => _LogsPanelState();
}

class _LogsPanelState extends ConsumerState<LogsPanel> {
  final _scroll = ScrollController();
  List<String> _lines = [];
  bool _loading = false;
  Timer? _timer;

  @override
  void initState() {
    super.initState();
    _load();
    // Live tail while the user is on this tab (the timer dies with the page).
    _timer = Timer.periodic(const Duration(seconds: 3), (_) => _load());
  }

  @override
  void dispose() {
    _timer?.cancel();
    _scroll.dispose();
    super.dispose();
  }

  Future<void> _load() async {
    if (_loading) return;
    _loading = true;
    final lines = await ref.read(appStateProvider).fetchBackendLogs(lines: 600);
    _loading = false;
    if (!mounted) return;
    final atBottom =
        !_scroll.hasClients ||
        _scroll.position.maxScrollExtent - _scroll.position.pixels < 80;
    setState(() => _lines = lines);
    // Keep following the tail only if the user hadn't scrolled up to read.
    if (atBottom) {
      WidgetsBinding.instance.addPostFrameCallback((_) {
        if (_scroll.hasClients) {
          _scroll.jumpTo(_scroll.position.maxScrollExtent);
        }
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    final state = ref.watch(appStateProvider);
    final scheme = Theme.of(context).colorScheme;
    final dark = scheme.brightness == Brightness.dark;
    final logText = _lines.isEmpty
        ? 'No backend log output yet.'
        : _lines.join('\n');
    final trainingJson = const JsonEncoder.withIndent(
      '  ',
    ).convert(state.trainingStatus ?? {});

    return ListView(
      padding: const EdgeInsets.all(14),
      children: [
        SectionHeader(
          'Backend log',
          trailing: Row(
            mainAxisSize: MainAxisSize.min,
            children: [
              IconButton(
                tooltip: 'Refresh now',
                visualDensity: VisualDensity.compact,
                iconSize: 18,
                onPressed: _load,
                icon: const Icon(Icons.refresh),
              ),
              CopyIconButton(text: logText, tooltip: 'Copy backend log'),
            ],
          ),
        ),
        Container(
          height: 380,
          width: double.infinity,
          padding: const EdgeInsets.all(12),
          decoration: BoxDecoration(
            borderRadius: BorderRadius.circular(8),
            color: dark ? const Color(0xff10141a) : const Color(0xff1d232b),
            border: Border.all(color: scheme.outlineVariant),
          ),
          child: Scrollbar(
            controller: _scroll,
            child: SingleChildScrollView(
              controller: _scroll,
              child: SelectableText(
                logText,
                style: monoStyle(
                  context,
                  fontSize: 11.5,
                  color: const Color(0xffcfd8e3),
                ),
              ),
            ),
          ),
        ),
        const SizedBox(height: 4),
        Text(
          'Live tail of app_settings/logs/backend.log — auto-refreshes every 3s, '
          'last 600 lines.',
          style: TextStyle(
            fontSize: 11.5,
            color: scheme.onSurface.withValues(alpha: 0.5),
          ),
        ),
        const SizedBox(height: 18),
        SectionHeader(
          'Runtime status',
          trailing: CopyIconButton(
            text: state.message,
            tooltip: 'Copy status message',
          ),
        ),
        Container(
          padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 10),
          decoration: BoxDecoration(
            borderRadius: BorderRadius.circular(8),
            color: scheme.onSurface.withValues(alpha: 0.04),
            border: Border.all(color: scheme.outlineVariant),
          ),
          child: SelectableText(
            state.message,
            style: const TextStyle(fontSize: 13, height: 1.45),
          ),
        ),
        const SizedBox(height: 16),
        SectionHeader(
          'Training status (JSON)',
          trailing: TextButton.icon(
            onPressed: () => copyToClipboard(
              context,
              trainingJson,
              label: 'Copied training status JSON',
            ),
            icon: const Icon(Icons.copy_rounded, size: 15),
            label: const Text('Copy JSON'),
          ),
        ),
        Container(
          width: double.infinity,
          padding: const EdgeInsets.all(12),
          decoration: BoxDecoration(
            borderRadius: BorderRadius.circular(8),
            color: scheme.brightness == Brightness.dark
                ? const Color(0xff10141a)
                : const Color(0xffeef1f5),
            border: Border.all(color: scheme.outlineVariant),
          ),
          child: SelectableText(
            trainingJson,
            style: monoStyle(context, fontSize: 12),
          ),
        ),
      ],
    );
  }
}
