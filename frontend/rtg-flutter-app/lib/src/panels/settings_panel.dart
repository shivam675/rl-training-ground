import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../app_state.dart';
import '../theme/app_theme.dart';
import '../theme/easy_colors.dart';
import '../theme/theme_controller.dart';
import '../widgets/common.dart';

class SettingsPanel extends ConsumerStatefulWidget {
  const SettingsPanel({super.key});

  @override
  ConsumerState<SettingsPanel> createState() => _SettingsPanelState();
}

class _SettingsPanelState extends ConsumerState<SettingsPanel> {
  // Which provider is active. Both providers' settings are always kept and
  // saved, so switching never loses a key, model or tuning value.
  String activeProvider = 'ollama';

  // --- Ollama provider ---
  final oProviderController = TextEditingController(
    text: 'Local/Remote Ollama',
  );
  final oBaseUrlController = TextEditingController(
    text: 'http://localhost:11434',
  );
  final oModelController = TextEditingController(text: 'llama3.1');
  final oTokenController = TextEditingController();
  final oTimeoutController = TextEditingController(text: '30');
  double oTemperature = 0.3;
  double oTopP = 0.9;
  double oNumPredict = 512;
  bool oThinking = true;

  // --- OpenAI-compatible provider (OpenAI, NVIDIA NIM, vLLM, …) ---
  final aProviderController = TextEditingController(text: 'OpenAI-compatible');
  final aBaseUrlController = TextEditingController(
    text: 'https://integrate.api.nvidia.com/v1',
  );
  final aModelController = TextEditingController(
    text: 'nvidia/nemotron-3-ultra-550b-a55b',
  );
  final aKeyController = TextEditingController();
  final aTimeoutController = TextEditingController(text: '120');
  double aTemperature = 0.6;
  double aTopP = 0.95;
  double aMaxTokens = 4096;
  double aReasoningBudget = 16384;
  bool aThinking = true;

  bool showSecret = false;
  bool hydrated = false;
  String? capabilityResult;
  bool checkingCapabilities = false;

  Future<void> _checkCapabilities(AppState state) async {
    setState(() {
      checkingCapabilities = true;
      capabilityResult = null;
    });
    try {
      final res = await state.checkModelCapabilities();
      setState(() {
        final supportsTools = res['supports_tools'] == true;
        capabilityResult = supportsTools
            ? '✓ ${res['model']} supports tool calling'
                  '${res['context_length'] != null ? ' · context ${res['context_length']}' : ''}'
            : '✗ ${res['model']} does NOT support tools — the agent can chat '
                  'but cannot operate the app. Pick a tool-capable model.';
      });
    } catch (e) {
      setState(
        () => capabilityResult =
            '✗ ${e.toString().replaceFirst('Exception: ', '')}',
      );
    } finally {
      if (mounted) setState(() => checkingCapabilities = false);
    }
  }

  @override
  void dispose() {
    for (final c in [
      oProviderController,
      oBaseUrlController,
      oModelController,
      oTokenController,
      oTimeoutController,
      aProviderController,
      aBaseUrlController,
      aModelController,
      aKeyController,
      aTimeoutController,
    ]) {
      c.dispose();
    }
    super.dispose();
  }

  void hydrateFromState(AppState state) {
    final s = state.agentSettings;
    if (hydrated || s == null) return;
    activeProvider = s['active_provider']?.toString() == 'openai'
        ? 'openai'
        : 'ollama';
    final o = (s['ollama'] as Map?)?.cast<String, dynamic>() ?? {};
    oProviderController.text =
        o['provider_name']?.toString() ?? oProviderController.text;
    oBaseUrlController.text =
        o['base_url']?.toString() ?? oBaseUrlController.text;
    oModelController.text =
        o['model_name']?.toString() ?? oModelController.text;
    oTokenController.text = o['bearer_token']?.toString() ?? '';
    oTimeoutController.text =
        (o['timeout_seconds'] as num?)?.toString() ?? oTimeoutController.text;
    oTemperature = (o['temperature'] as num?)?.toDouble() ?? oTemperature;
    oTopP = (o['top_p'] as num?)?.toDouble() ?? oTopP;
    oNumPredict = (o['num_predict'] as num?)?.toDouble() ?? oNumPredict;
    oThinking = o['enable_thinking'] as bool? ?? oThinking;

    final a = (s['openai'] as Map?)?.cast<String, dynamic>() ?? {};
    aProviderController.text =
        a['provider_name']?.toString() ?? aProviderController.text;
    aBaseUrlController.text =
        a['base_url']?.toString() ?? aBaseUrlController.text;
    aModelController.text =
        a['model_name']?.toString() ?? aModelController.text;
    aKeyController.text = a['api_key']?.toString() ?? '';
    aTimeoutController.text =
        (a['timeout_seconds'] as num?)?.toString() ?? aTimeoutController.text;
    aTemperature = (a['temperature'] as num?)?.toDouble() ?? aTemperature;
    aTopP = (a['top_p'] as num?)?.toDouble() ?? aTopP;
    aMaxTokens = (a['max_tokens'] as num?)?.toDouble() ?? aMaxTokens;
    aReasoningBudget =
        (a['reasoning_budget'] as num?)?.toDouble() ?? aReasoningBudget;
    aThinking = a['enable_thinking'] as bool? ?? aThinking;
    hydrated = true;
  }

  @override
  Widget build(BuildContext context) {
    final state = ref.watch(appStateProvider);
    final theme = ref.watch(themeControllerProvider);
    final scheme = Theme.of(context).colorScheme;
    hydrateFromState(state);
    return ListView(
      padding: const EdgeInsets.all(14),
      children: [
        const SectionHeader('Appearance'),
        Row(
          children: [
            Text(
              'Theme mode',
              style: TextStyle(
                fontSize: 13,
                color: scheme.onSurface.withValues(alpha: 0.75),
              ),
            ),
            const SizedBox(width: 16),
            SegmentedButton<ThemeMode>(
              showSelectedIcon: false,
              segments: const [
                ButtonSegment(
                  value: ThemeMode.dark,
                  icon: Icon(Icons.dark_mode_outlined, size: 16),
                  label: Text('Dark'),
                ),
                ButtonSegment(
                  value: ThemeMode.light,
                  icon: Icon(Icons.light_mode_outlined, size: 16),
                  label: Text('Light'),
                ),
                ButtonSegment(
                  value: ThemeMode.system,
                  icon: Icon(Icons.brightness_auto_outlined, size: 16),
                  label: Text('System'),
                ),
              ],
              selected: {theme.mode},
              onSelectionChanged: (selection) => theme.setMode(selection.first),
            ),
          ],
        ),
        const SizedBox(height: 14),
        Text(
          'Accent color',
          style: TextStyle(
            fontSize: 13,
            color: scheme.onSurface.withValues(alpha: 0.75),
          ),
        ),
        const SizedBox(height: 8),
        Wrap(
          spacing: 10,
          runSpacing: 10,
          children: [
            for (var i = 0; i < accentPresets.length; i++)
              _AccentSwatch(
                preset: accentPresets[i],
                selected: theme.accentIndex == i,
                onTap: () => theme.setAccent(i),
              ),
          ],
        ),
        const SizedBox(height: 18),
        const SectionHeader('Viewport'),
        Text(
          'Stream resolution scale · ${state.streamResolutionScale.toStringAsFixed(2)}x',
          style: const TextStyle(fontSize: 13, fontWeight: FontWeight.w600),
        ),
        Slider(
          value: state.streamResolutionScale,
          min: 0.5,
          max: 1.5,
          divisions: 10,
          label: '${state.streamResolutionScale.toStringAsFixed(2)}x',
          onChanged: state.setStreamResolutionScale,
        ),
        Text(
          'Lower scale improves FPS. Higher scale improves viewport clarity.',
          style: TextStyle(
            fontSize: 12,
            color: scheme.onSurface.withValues(alpha: 0.55),
          ),
        ),
        const SizedBox(height: 18),
        const SectionHeader('Agent behavior'),
        Row(
          children: [
            Text(
              'Autonomy',
              style: TextStyle(
                fontSize: 13,
                color: scheme.onSurface.withValues(alpha: 0.75),
              ),
            ),
            const SizedBox(width: 16),
            SegmentedButton<String>(
              showSelectedIcon: false,
              segments: const [
                ButtonSegment(
                  value: 'act',
                  icon: Icon(Icons.bolt, size: 16),
                  label: Text('Act freely'),
                ),
                ButtonSegment(
                  value: 'ask',
                  icon: Icon(Icons.pan_tool_outlined, size: 16),
                  label: Text('Ask first'),
                ),
              ],
              selected: {state.agentAutonomy},
              onSelectionChanged: (selection) =>
                  state.setAgentAutonomy(selection.first),
            ),
          ],
        ),
        const SizedBox(height: 6),
        Text(
          state.agentAutonomy == 'ask'
              ? 'State-changing tools (load robot, start training, …) pause '
                    'for your confirmation — a Run button appears in chat.'
              : 'The agent executes state-changing tools immediately.',
          style: TextStyle(
            fontSize: 12,
            color: scheme.onSurface.withValues(alpha: 0.55),
          ),
        ),
        const SizedBox(height: 18),
        const SectionHeader('Agent provider'),
        SegmentedButton<String>(
          showSelectedIcon: false,
          segments: const [
            ButtonSegment(
              value: 'ollama',
              icon: Icon(Icons.dns_outlined, size: 16),
              label: Text('Ollama'),
            ),
            ButtonSegment(
              value: 'openai',
              icon: Icon(Icons.cloud_outlined, size: 16),
              label: Text('OpenAI-compatible'),
            ),
          ],
          selected: {activeProvider},
          onSelectionChanged: (selection) =>
              setState(() => activeProvider = selection.first),
        ),
        const SizedBox(height: 6),
        Text(
          'The active provider runs the agent. Both providers keep their own '
          'URL, key, model and tuning — switching never clears them. Save to apply.',
          style: TextStyle(
            fontSize: 12,
            color: scheme.onSurface.withValues(alpha: 0.55),
          ),
        ),
        const SizedBox(height: 12),
        if (activeProvider == 'ollama')
          ..._ollamaFields(scheme)
        else
          ..._openaiFields(scheme),
        const SizedBox(height: 14),
        Wrap(
          spacing: 8,
          runSpacing: 8,
          children: [
            FilledButton.icon(
              onPressed: state.busy ? null : () => _save(state),
              icon: const Icon(Icons.save_outlined),
              label: const Text('Save settings'),
            ),
            OutlinedButton.icon(
              onPressed: checkingCapabilities
                  ? null
                  : () => _checkCapabilities(state),
              icon: checkingCapabilities
                  ? const SizedBox(
                      width: 14,
                      height: 14,
                      child: CircularProgressIndicator(strokeWidth: 2),
                    )
                  : const Icon(Icons.checklist, size: 16),
              label: const Text('Check model capabilities'),
            ),
            OutlinedButton.icon(
              onPressed: () => _resetDefaults(state),
              icon: const Icon(Icons.restart_alt),
              label: const Text('Reset defaults'),
            ),
            OutlinedButton.icon(
              onPressed: () async {
                try {
                  final res = await state.api.postJson(
                    '/diagnostics/export',
                    {},
                  );
                  final path = res['path']?.toString() ?? '';
                  if (context.mounted && path.isNotEmpty) {
                    copyToClipboard(
                      context,
                      path,
                      label: 'Diagnostics bundle created — path copied',
                    );
                  }
                } catch (e) {
                  if (context.mounted) {
                    ScaffoldMessenger.of(context).showSnackBar(
                      SnackBar(content: Text('Export failed: $e')),
                    );
                  }
                }
              },
              icon: const Icon(Icons.archive_outlined, size: 16),
              label: const Text('Export diagnostics'),
            ),
          ],
        ),
        if (capabilityResult != null) ...[
          const SizedBox(height: 10),
          Text(
            capabilityResult!,
            style: TextStyle(
              fontSize: 12.5,
              height: 1.4,
              color: capabilityResult!.startsWith('✓')
                  ? context.colors.success
                  : context.colors.danger,
            ),
          ),
        ],
      ],
    );
  }

  List<Widget> _ollamaFields(ColorScheme scheme) {
    return [
      TextField(
        controller: oProviderController,
        decoration: const InputDecoration(
          labelText: 'Provider name',
          prefixIcon: Icon(Icons.dns_outlined, size: 18),
        ),
      ),
      const SizedBox(height: 10),
      TextField(
        controller: oBaseUrlController,
        style: monoStyle(context, fontSize: 13),
        decoration: const InputDecoration(
          labelText: 'Base URL',
          prefixIcon: Icon(Icons.link, size: 18),
        ),
      ),
      const SizedBox(height: 10),
      Row(
        children: [
          Expanded(
            child: TextField(
              controller: oModelController,
              style: monoStyle(context, fontSize: 13),
              decoration: const InputDecoration(
                labelText: 'Model',
                prefixIcon: Icon(Icons.memory, size: 18),
              ),
            ),
          ),
          const SizedBox(width: 10),
          Expanded(
            child: _secretField(oTokenController, 'Bearer token (optional)'),
          ),
        ],
      ),
      const SizedBox(height: 12),
      Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Expanded(
            child: TextField(
              controller: oTimeoutController,
              keyboardType: TextInputType.number,
              decoration: const InputDecoration(
                labelText: 'Timeout seconds',
                prefixIcon: Icon(Icons.timer_outlined, size: 18),
              ),
            ),
          ),
          const SizedBox(width: 10),
          Expanded(
            child: _LabeledSlider(
              label: 'Temperature',
              value: oTemperature,
              min: 0,
              max: 1,
              divisions: 20,
              display: oTemperature.toStringAsFixed(2),
              onChanged: (v) => setState(() => oTemperature = v),
            ),
          ),
        ],
      ),
      Row(
        children: [
          Expanded(
            child: _LabeledSlider(
              label: 'Top-p',
              value: oTopP,
              min: 0,
              max: 1,
              divisions: 20,
              display: oTopP.toStringAsFixed(2),
              onChanged: (v) => setState(() => oTopP = v),
            ),
          ),
          const SizedBox(width: 10),
          Expanded(
            child: _LabeledSlider(
              label: 'Max tokens',
              value: oNumPredict,
              min: 64,
              max: 8192,
              divisions: 127,
              display: '${oNumPredict.round()}',
              onChanged: (v) => setState(() => oNumPredict = v),
            ),
          ),
        ],
      ),
      _thinkingTile(
        value: oThinking,
        onChanged: (v) => setState(() => oThinking = v),
        scheme: scheme,
      ),
    ];
  }

  List<Widget> _openaiFields(ColorScheme scheme) {
    return [
      TextField(
        controller: aProviderController,
        decoration: const InputDecoration(
          labelText: 'Provider name',
          prefixIcon: Icon(Icons.cloud_outlined, size: 18),
        ),
      ),
      const SizedBox(height: 10),
      TextField(
        controller: aBaseUrlController,
        style: monoStyle(context, fontSize: 13),
        decoration: const InputDecoration(
          labelText: 'Base URL (…/v1)',
          prefixIcon: Icon(Icons.link, size: 18),
        ),
      ),
      const SizedBox(height: 10),
      Row(
        children: [
          Expanded(
            child: TextField(
              controller: aModelController,
              style: monoStyle(context, fontSize: 13),
              decoration: const InputDecoration(
                labelText: 'Model',
                prefixIcon: Icon(Icons.memory, size: 18),
              ),
            ),
          ),
          const SizedBox(width: 10),
          Expanded(child: _secretField(aKeyController, 'API key')),
        ],
      ),
      const SizedBox(height: 12),
      Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Expanded(
            child: TextField(
              controller: aTimeoutController,
              keyboardType: TextInputType.number,
              decoration: const InputDecoration(
                labelText: 'Timeout seconds',
                prefixIcon: Icon(Icons.timer_outlined, size: 18),
              ),
            ),
          ),
          const SizedBox(width: 10),
          Expanded(
            child: _LabeledSlider(
              label: 'Temperature',
              value: aTemperature,
              min: 0,
              max: 2,
              divisions: 40,
              display: aTemperature.toStringAsFixed(2),
              onChanged: (v) => setState(() => aTemperature = v),
            ),
          ),
        ],
      ),
      Row(
        children: [
          Expanded(
            child: _LabeledSlider(
              label: 'Top-p',
              value: aTopP,
              min: 0,
              max: 1,
              divisions: 20,
              display: aTopP.toStringAsFixed(2),
              onChanged: (v) => setState(() => aTopP = v),
            ),
          ),
          const SizedBox(width: 10),
          Expanded(
            child: _LabeledSlider(
              label: 'Max tokens',
              value: aMaxTokens,
              min: 256,
              max: 32768,
              divisions: 127,
              display: '${aMaxTokens.round()}',
              onChanged: (v) => setState(() => aMaxTokens = v),
            ),
          ),
        ],
      ),
      _LabeledSlider(
        label: 'Reasoning budget (thinking tokens)',
        value: aReasoningBudget,
        min: 0,
        max: 32768,
        divisions: 128,
        display: '${aReasoningBudget.round()}',
        onChanged: (v) => setState(() => aReasoningBudget = v),
      ),
      _thinkingTile(
        value: aThinking,
        onChanged: (v) => setState(() => aThinking = v),
        scheme: scheme,
      ),
    ];
  }

  Widget _secretField(TextEditingController controller, String label) {
    return TextField(
      controller: controller,
      obscureText: !showSecret,
      decoration: InputDecoration(
        labelText: label,
        prefixIcon: const Icon(Icons.key_outlined, size: 18),
        suffixIcon: IconButton(
          tooltip: showSecret ? 'Hide' : 'Show',
          iconSize: 17,
          onPressed: () => setState(() => showSecret = !showSecret),
          icon: Icon(
            showSecret
                ? Icons.visibility_off_outlined
                : Icons.visibility_outlined,
          ),
        ),
      ),
    );
  }

  Widget _thinkingTile({
    required bool value,
    required ValueChanged<bool> onChanged,
    required ColorScheme scheme,
  }) {
    return SwitchListTile(
      contentPadding: EdgeInsets.zero,
      dense: true,
      value: value,
      onChanged: onChanged,
      title: const Text(
        'Show model thinking',
        style: TextStyle(fontSize: 13, fontWeight: FontWeight.w600),
      ),
      subtitle: Text(
        value
            ? 'Reasoning models show a collapsible thinking section per reply.'
            : 'No-think: the model skips reasoning (faster). Save to apply.',
        style: TextStyle(
          fontSize: 12,
          color: scheme.onSurface.withValues(alpha: 0.55),
        ),
      ),
    );
  }

  Map<String, dynamic> _ollamaPayload() => {
    'provider_name': oProviderController.text.trim(),
    'base_url': oBaseUrlController.text.trim(),
    'bearer_token': oTokenController.text,
    'model_name': oModelController.text.trim(),
    'temperature': oTemperature,
    'top_p': oTopP,
    'num_predict': oNumPredict.round(),
    'timeout_seconds': double.tryParse(oTimeoutController.text.trim()) ?? 30,
    'enable_thinking': oThinking,
    'system_prompt_override': '',
  };

  Map<String, dynamic> _openaiPayload() => {
    'provider_name': aProviderController.text.trim(),
    'base_url': aBaseUrlController.text.trim(),
    'api_key': aKeyController.text,
    'model_name': aModelController.text.trim(),
    'temperature': aTemperature,
    'top_p': aTopP,
    'max_tokens': aMaxTokens.round(),
    'reasoning_budget': aReasoningBudget.round(),
    'timeout_seconds': double.tryParse(aTimeoutController.text.trim()) ?? 120,
    'enable_thinking': aThinking,
    'system_prompt_override': '',
  };

  void _save(AppState state) {
    state.saveAgentSettings({
      'active_provider': activeProvider,
      'ollama': _ollamaPayload(),
      'openai': _openaiPayload(),
    });
  }

  void _resetDefaults(AppState state) {
    oProviderController.text = 'Local/Remote Ollama';
    oBaseUrlController.text = 'http://localhost:11434';
    oModelController.text = 'llama3.1';
    oTokenController.clear();
    oTimeoutController.text = '30';
    setState(() {
      activeProvider = 'ollama';
      oTemperature = 0.3;
      oTopP = 0.9;
      oNumPredict = 512;
      oThinking = true;
    });
    state.setStreamResolutionScale(1.0);
    state.saveAgentSettings({
      'active_provider': 'ollama',
      'ollama': _ollamaPayload(),
      'openai': _openaiPayload(),
    });
  }
}

class _AccentSwatch extends StatelessWidget {
  const _AccentSwatch({
    required this.preset,
    required this.selected,
    required this.onTap,
  });

  final AccentPreset preset;
  final bool selected;
  final VoidCallback onTap;

  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;
    return Tooltip(
      message: preset.name,
      child: InkWell(
        borderRadius: BorderRadius.circular(10),
        onTap: onTap,
        child: AnimatedContainer(
          duration: const Duration(milliseconds: 180),
          padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
          decoration: BoxDecoration(
            borderRadius: BorderRadius.circular(10),
            color: selected
                ? preset.seed.withValues(alpha: 0.14)
                : Colors.transparent,
            border: Border.all(
              color: selected ? preset.seed : scheme.outlineVariant,
              width: selected ? 1.6 : 1,
            ),
          ),
          child: Row(
            mainAxisSize: MainAxisSize.min,
            children: [
              AnimatedContainer(
                duration: const Duration(milliseconds: 180),
                width: 16,
                height: 16,
                decoration: BoxDecoration(
                  shape: BoxShape.circle,
                  color: preset.seed,
                  boxShadow: selected
                      ? [
                          BoxShadow(
                            color: preset.seed.withValues(alpha: 0.5),
                            blurRadius: 8,
                          ),
                        ]
                      : null,
                ),
                child: selected
                    ? const Icon(Icons.check, size: 12, color: Colors.white)
                    : null,
              ),
              const SizedBox(width: 8),
              Text(
                preset.name,
                style: TextStyle(
                  fontSize: 12.5,
                  fontWeight: selected ? FontWeight.w700 : FontWeight.w500,
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

class _LabeledSlider extends StatelessWidget {
  const _LabeledSlider({
    required this.label,
    required this.value,
    required this.min,
    required this.max,
    required this.divisions,
    required this.display,
    required this.onChanged,
  });

  final String label;
  final double value;
  final double min;
  final double max;
  final int divisions;
  final String display;
  final ValueChanged<double> onChanged;

  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;
    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        Row(
          mainAxisAlignment: MainAxisAlignment.spaceBetween,
          children: [
            Text(
              label,
              style: TextStyle(
                fontSize: 12.5,
                color: scheme.onSurface.withValues(alpha: 0.7),
              ),
            ),
            Text(
              display,
              style: monoStyle(
                context,
                fontSize: 12,
                color: scheme.primary,
              ).copyWith(fontWeight: FontWeight.w700),
            ),
          ],
        ),
        Slider(
          value: value.clamp(min, max),
          min: min,
          max: max,
          divisions: divisions,
          label: display,
          onChanged: onChanged,
        ),
      ],
    );
  }
}
