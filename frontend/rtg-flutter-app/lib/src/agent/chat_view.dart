import 'package:flutter/material.dart';
import 'package:flutter_markdown/flutter_markdown.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../app_state.dart';
import '../theme/app_theme.dart';
import '../theme/easy_colors.dart';
import '../widgets/common.dart';
import 'chat_controller.dart';

/// The agent conversation surface: header, message list and composer. State
/// lives in [ChatController] (shared), so this widget is just a renderer + the
/// scroll/focus plumbing. Usable standalone (tests, a full page) or wrapped by
/// the co-pilot dock — pass [onCollapse] to show a collapse affordance.
class AssistantChat extends ConsumerStatefulWidget {
  const AssistantChat({super.key, this.onCollapse});

  /// When non-null, a collapse chevron is shown in the header.
  final VoidCallback? onCollapse;

  @override
  ConsumerState<AssistantChat> createState() => _AssistantChatState();
}

class _AssistantChatState extends ConsumerState<AssistantChat> {
  final inputController = TextEditingController();
  final scrollController = ScrollController();
  final focusNode = FocusNode();

  // Follow the bottom only while the user hasn't scrolled away, and schedule at
  // most one (instant) scroll per frame instead of stacking animations.
  bool _autoFollow = true;
  bool _scrollScheduled = false;
  int _lastForceTick = 0;

  @override
  void initState() {
    super.initState();
    scrollController.addListener(_onScroll);
  }

  @override
  void dispose() {
    inputController.dispose();
    scrollController.dispose();
    focusNode.dispose();
    super.dispose();
  }

  void _onScroll() {
    if (!scrollController.hasClients) return;
    final position = scrollController.position;
    _autoFollow = position.maxScrollExtent - position.pixels < 60;
  }

  void scrollToBottom({bool force = false}) {
    if (force) _autoFollow = true;
    if (!_autoFollow || _scrollScheduled) return;
    _scrollScheduled = true;
    WidgetsBinding.instance.addPostFrameCallback((_) {
      _scrollScheduled = false;
      if (!mounted || !scrollController.hasClients || !_autoFollow) return;
      scrollController.jumpTo(scrollController.position.maxScrollExtent);
    });
  }

  void _send(ChatController controller) {
    final text = inputController.text;
    if (text.trim().isEmpty || controller.sending) return;
    controller.sendChat(text);
    inputController.clear();
    focusNode.requestFocus();
  }

  @override
  Widget build(BuildContext context) {
    final controller = ref.watch(chatControllerProvider);
    // Watch app state too so the connection dot updates live.
    final state = ref.watch(appStateProvider);
    final scheme = Theme.of(context).colorScheme;
    final connected = state.agentConnected;

    // Keep the view pinned: force after sends/prompts, otherwise follow unless
    // the user scrolled up.
    if (controller.forceScrollTick != _lastForceTick) {
      _lastForceTick = controller.forceScrollTick;
      WidgetsBinding.instance.addPostFrameCallback(
        (_) => scrollToBottom(force: true),
      );
    } else {
      WidgetsBinding.instance.addPostFrameCallback((_) => scrollToBottom());
    }

    final itemCount =
        controller.messages.length + (controller.showTyping ? 1 : 0);

    return Column(
      children: [
        Padding(
          padding: const EdgeInsets.fromLTRB(12, 8, 6, 6),
          child: Row(
            children: [
              if (widget.onCollapse != null)
                IconButton(
                  tooltip: 'Collapse assistant',
                  visualDensity: VisualDensity.compact,
                  iconSize: 18,
                  onPressed: widget.onCollapse,
                  icon: const Icon(Icons.chevron_right),
                ),
              Expanded(
                child: Row(
                  children: [
                    CircleAvatar(
                      radius: 14,
                      backgroundColor: scheme.primary.withValues(alpha: 0.15),
                      child: Icon(
                        Icons.smart_toy_outlined,
                        size: 16,
                        color: scheme.primary,
                      ),
                    ),
                    const SizedBox(width: 10),
                    const Flexible(
                      child: Text(
                        'Assistant',
                        maxLines: 1,
                        overflow: TextOverflow.ellipsis,
                        style: TextStyle(
                          fontSize: 14,
                          fontWeight: FontWeight.w700,
                        ),
                      ),
                    ),
                    const SizedBox(width: 10),
                    Flexible(
                      child: Tooltip(
                        message: state.agentConnectionDetail,
                        child: StatusDot(
                          label: connected ? 'Connected' : 'Offline',
                          online: connected,
                        ),
                      ),
                    ),
                  ],
                ),
              ),
              IconButton(
                tooltip: 'Copy conversation',
                visualDensity: VisualDensity.compact,
                iconSize: 18,
                onPressed: controller.hasHistory
                    ? () => copyToClipboard(
                        context,
                        controller.transcript(),
                        label: 'Copied conversation',
                      )
                    : null,
                icon: const Icon(Icons.copy_all),
              ),
              IconButton(
                tooltip: 'Clear chat',
                visualDensity: VisualDensity.compact,
                iconSize: 18,
                onPressed: controller.hasHistory ? controller.clearChat : null,
                icon: const Icon(Icons.delete_sweep_outlined),
              ),
              IconButton(
                tooltip: 'Refresh backend + agent connection',
                visualDensity: VisualDensity.compact,
                iconSize: 18,
                onPressed: () {
                  state.refreshAll();
                  state.refreshAgentHealth();
                },
                icon: const Icon(Icons.refresh),
              ),
            ],
          ),
        ),
        const Divider(height: 1),
        Expanded(
          child: ListView.builder(
            controller: scrollController,
            padding: const EdgeInsets.all(12),
            itemCount: itemCount,
            itemBuilder: (context, index) {
              if (index >= controller.messages.length) {
                return const _TypingRow();
              }
              final message = controller.messages[index];
              return switch (message.kind) {
                ChatKind.tool => _ToolActivityRow(
                  message: message,
                  onConfirm: () => controller.confirmTool(message),
                ),
                ChatKind.notice => _NoticeRow(text: message.text),
                _ => _MessageBubble(message: message),
              };
            },
          ),
        ),
        const Divider(),
        Padding(
          padding: const EdgeInsets.all(10),
          child: Row(
            crossAxisAlignment: CrossAxisAlignment.end,
            children: [
              Expanded(
                child: TextField(
                  controller: inputController,
                  focusNode: focusNode,
                  minLines: 1,
                  maxLines: 4,
                  textInputAction: TextInputAction.send,
                  decoration: InputDecoration(
                    hintText: connected
                        ? 'Ask the agent — it can operate the app for you…'
                        : 'Backend offline — messages will fail',
                    prefixIcon: const Icon(Icons.chat_bubble_outline, size: 18),
                  ),
                  onSubmitted: (_) => _send(controller),
                ),
              ),
              const SizedBox(width: 8),
              IconButton.filled(
                tooltip: 'Send (Enter)',
                onPressed: controller.sending ? null : () => _send(controller),
                icon: controller.sending
                    ? const SizedBox(
                        width: 18,
                        height: 18,
                        child: CircularProgressIndicator(strokeWidth: 2),
                      )
                    : const Icon(Icons.send_rounded),
              ),
            ],
          ),
        ),
      ],
    );
  }
}

class _ToolActivityRow extends StatelessWidget {
  const _ToolActivityRow({required this.message, required this.onConfirm});

  final ChatMessage message;
  final VoidCallback onConfirm;

  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;
    final colors = context.colors;
    final awaiting = message.needsConfirmation;
    final running = message.toolOk == null && !awaiting;
    final failed = message.toolOk == false;
    final color = awaiting
        ? colors.warning
        : failed
        ? colors.danger
        : running
        ? scheme.primary
        : colors.success;
    final label = [
      message.toolName ?? 'tool',
      if ((message.toolArgs ?? '').isNotEmpty) '(${message.toolArgs})',
    ].join(' ');
    return Padding(
      padding: const EdgeInsets.only(bottom: 8, left: 34),
      child: Row(
        children: [
          Flexible(
            child: Container(
              padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 5),
              decoration: BoxDecoration(
                borderRadius: BorderRadius.circular(8),
                color: color.withValues(alpha: 0.08),
                border: Border.all(color: color.withValues(alpha: 0.35)),
              ),
              child: Row(
                mainAxisSize: MainAxisSize.min,
                children: [
                  if (running)
                    SizedBox(
                      width: 12,
                      height: 12,
                      child: CircularProgressIndicator(
                        strokeWidth: 1.6,
                        color: color,
                      ),
                    )
                  else
                    Icon(
                      awaiting
                          ? Icons.pan_tool_outlined
                          : failed
                          ? Icons.error_outline
                          : Icons.check_circle_outline,
                      size: 14,
                      color: color,
                    ),
                  const SizedBox(width: 7),
                  Flexible(
                    child: Text(
                      failed || awaiting ? '$label — ${message.text}' : label,
                      maxLines: 2,
                      overflow: TextOverflow.ellipsis,
                      style: monoStyle(context, fontSize: 11.5, color: color),
                    ),
                  ),
                  if (awaiting) ...[
                    const SizedBox(width: 8),
                    SizedBox(
                      height: 24,
                      child: FilledButton.tonal(
                        style: FilledButton.styleFrom(
                          padding: const EdgeInsets.symmetric(horizontal: 10),
                          visualDensity: VisualDensity.compact,
                        ),
                        onPressed: onConfirm,
                        child: const Text(
                          'Run',
                          style: TextStyle(fontSize: 11),
                        ),
                      ),
                    ),
                  ],
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }
}

class _NoticeRow extends StatelessWidget {
  const _NoticeRow({required this.text});

  final String text;

  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;
    return Padding(
      padding: const EdgeInsets.only(bottom: 8),
      child: Center(
        child: Text(
          text,
          textAlign: TextAlign.center,
          style: TextStyle(
            fontSize: 11.5,
            fontStyle: FontStyle.italic,
            color: scheme.onSurface.withValues(alpha: 0.55),
          ),
        ),
      ),
    );
  }
}

/// Collapsible reasoning shown above an agent reply. Collapsed by default.
class _ThinkingSection extends StatelessWidget {
  const _ThinkingSection({
    required this.thinking,
    required this.streaming,
    required this.expanded,
    required this.onToggle,
  });

  final String thinking;
  final bool streaming;
  final bool expanded;
  final VoidCallback onToggle;

  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;
    final muted = scheme.onSurface.withValues(alpha: 0.55);
    return Container(
      margin: const EdgeInsets.only(bottom: 6),
      decoration: BoxDecoration(
        borderRadius: BorderRadius.circular(8),
        border: Border.all(color: scheme.outlineVariant),
        color: scheme.onSurface.withValues(alpha: 0.025),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          InkWell(
            borderRadius: BorderRadius.circular(8),
            onTap: onToggle,
            child: Padding(
              padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
              child: Row(
                mainAxisSize: MainAxisSize.min,
                children: [
                  if (streaming)
                    SizedBox(
                      width: 12,
                      height: 12,
                      child: CircularProgressIndicator(
                        strokeWidth: 1.6,
                        color: muted,
                      ),
                    )
                  else
                    Icon(Icons.psychology_outlined, size: 14, color: muted),
                  const SizedBox(width: 6),
                  Text(
                    streaming ? 'Thinking…' : 'Thinking',
                    style: TextStyle(
                      fontSize: 11.5,
                      fontWeight: FontWeight.w600,
                      color: muted,
                    ),
                  ),
                  const SizedBox(width: 2),
                  Icon(
                    expanded ? Icons.expand_less : Icons.expand_more,
                    size: 16,
                    color: muted,
                  ),
                ],
              ),
            ),
          ),
          if (expanded)
            Padding(
              padding: const EdgeInsets.fromLTRB(10, 0, 10, 8),
              child: SelectableText(
                thinking,
                style: monoStyle(context, fontSize: 11).copyWith(
                  color: scheme.onSurface.withValues(alpha: 0.6),
                  height: 1.45,
                ),
              ),
            ),
        ],
      ),
    );
  }
}

class _TypingRow extends StatelessWidget {
  const _TypingRow();

  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;
    return Padding(
      padding: const EdgeInsets.only(bottom: 10),
      child: Row(
        children: [
          CircleAvatar(
            radius: 13,
            backgroundColor: scheme.primary.withValues(alpha: 0.15),
            child: Icon(Icons.smart_toy, size: 15, color: scheme.primary),
          ),
          const SizedBox(width: 8),
          DecoratedBox(
            decoration: BoxDecoration(
              color: Theme.of(context).brightness == Brightness.dark
                  ? const Color(0xff262d36)
                  : const Color(0xffeef1f5),
              borderRadius: BorderRadius.circular(12),
            ),
            child: const Padding(
              padding: EdgeInsets.symmetric(horizontal: 12, vertical: 10),
              child: _TypingIndicator(),
            ),
          ),
        ],
      ),
    );
  }
}

class _MessageBubble extends StatefulWidget {
  const _MessageBubble({required this.message});

  final ChatMessage message;

  @override
  State<_MessageBubble> createState() => _MessageBubbleState();
}

class _MessageBubbleState extends State<_MessageBubble> {
  bool hovered = false;
  bool thinkingExpanded = false;

  @override
  Widget build(BuildContext context) {
    final message = widget.message;
    final isUser = message.kind == ChatKind.user;
    final hasThinking = !isUser && message.thinking.trim().isNotEmpty;
    final scheme = Theme.of(context).colorScheme;
    final dark = Theme.of(context).brightness == Brightness.dark;
    final userColor = scheme.primary.withValues(alpha: dark ? 0.22 : 0.14);
    final agentColor = dark ? const Color(0xff262d36) : const Color(0xffeef1f5);
    final codeBg = dark ? const Color(0xff161b22) : const Color(0xffe3e8ee);

    final timeLabel =
        '${message.time.hour.toString().padLeft(2, '0')}:${message.time.minute.toString().padLeft(2, '0')}';

    return MouseRegion(
      onEnter: (_) => setState(() => hovered = true),
      onExit: (_) => setState(() => hovered = false),
      child: Padding(
        padding: const EdgeInsets.only(bottom: 10),
        child: Row(
          mainAxisAlignment: isUser
              ? MainAxisAlignment.end
              : MainAxisAlignment.start,
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            if (!isUser) ...[
              CircleAvatar(
                radius: 13,
                backgroundColor: scheme.primary.withValues(alpha: 0.15),
                child: Icon(Icons.smart_toy, size: 15, color: scheme.primary),
              ),
              const SizedBox(width: 8),
            ],
            Flexible(
              child: Column(
                crossAxisAlignment: isUser
                    ? CrossAxisAlignment.end
                    : CrossAxisAlignment.start,
                children: [
                  if (hasThinking)
                    ConstrainedBox(
                      constraints: const BoxConstraints(maxWidth: 540),
                      child: _ThinkingSection(
                        thinking: message.thinking.trim(),
                        streaming: message.streaming,
                        expanded: thinkingExpanded,
                        onToggle: () => setState(
                          () => thinkingExpanded = !thinkingExpanded,
                        ),
                      ),
                    ),
                  if (message.text.isNotEmpty || !hasThinking)
                    ConstrainedBox(
                      constraints: const BoxConstraints(maxWidth: 540),
                      child: DecoratedBox(
                        decoration: BoxDecoration(
                          color: isUser ? userColor : agentColor,
                          borderRadius: BorderRadius.only(
                            topLeft: const Radius.circular(12),
                            topRight: const Radius.circular(12),
                            bottomLeft: Radius.circular(isUser ? 12 : 3),
                            bottomRight: Radius.circular(isUser ? 3 : 12),
                          ),
                        ),
                        child: Padding(
                          padding: const EdgeInsets.symmetric(
                            horizontal: 12,
                            vertical: 9,
                          ),
                          child: isUser
                              ? SelectableText(
                                  message.text,
                                  style: const TextStyle(height: 1.35),
                                )
                              : message.streaming
                              // Plain non-selectable text while tokens stream
                              // in: re-parsing markdown + rebuilding selection
                              // spans on every update freezes the UI on long
                              // replies.
                              ? Text(
                                  message.text,
                                  style: const TextStyle(height: 1.4),
                                )
                              : MarkdownBody(
                                  data: message.text,
                                  selectable: true,
                                  softLineBreak: true,
                                  styleSheet:
                                      MarkdownStyleSheet.fromTheme(
                                        Theme.of(context),
                                      ).copyWith(
                                        p: const TextStyle(height: 1.4),
                                        code: monoStyle(
                                          context,
                                        ).copyWith(backgroundColor: codeBg),
                                        codeblockDecoration: BoxDecoration(
                                          color: codeBg,
                                          borderRadius: BorderRadius.circular(
                                            8,
                                          ),
                                          border: Border.all(
                                            color: scheme.outlineVariant,
                                          ),
                                        ),
                                        codeblockPadding: const EdgeInsets.all(
                                          10,
                                        ),
                                        blockquoteDecoration: BoxDecoration(
                                          border: Border(
                                            left: BorderSide(
                                              color: scheme.primary,
                                              width: 3,
                                            ),
                                          ),
                                        ),
                                      ),
                                ),
                        ),
                      ),
                    ),
                  SizedBox(
                    height: 22,
                    child: AnimatedOpacity(
                      duration: const Duration(milliseconds: 150),
                      opacity: hovered && !message.streaming ? 1 : 0,
                      child: Row(
                        mainAxisSize: MainAxisSize.min,
                        children: [
                          Text(
                            timeLabel,
                            style: TextStyle(
                              fontSize: 10.5,
                              color: scheme.onSurface.withValues(alpha: 0.45),
                            ),
                          ),
                          CopyIconButton(
                            text: message.text,
                            tooltip: 'Copy message',
                            size: 13,
                          ),
                        ],
                      ),
                    ),
                  ),
                ],
              ),
            ),
            if (isUser) ...[
              const SizedBox(width: 8),
              CircleAvatar(
                radius: 13,
                backgroundColor: scheme.primary.withValues(alpha: 0.25),
                child: Icon(Icons.person, size: 15, color: scheme.primary),
              ),
            ],
          ],
        ),
      ),
    );
  }
}

class _TypingIndicator extends StatefulWidget {
  const _TypingIndicator();

  @override
  State<_TypingIndicator> createState() => _TypingIndicatorState();
}

class _TypingIndicatorState extends State<_TypingIndicator>
    with SingleTickerProviderStateMixin {
  late final AnimationController controller = AnimationController(
    vsync: this,
    duration: const Duration(milliseconds: 1000),
  )..repeat();

  @override
  void dispose() {
    controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final color = Theme.of(context).colorScheme.onSurface;
    return AnimatedBuilder(
      animation: controller,
      builder: (context, _) {
        return Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            for (var i = 0; i < 3; i++)
              Padding(
                padding: const EdgeInsets.symmetric(horizontal: 2.5),
                child: Opacity(
                  opacity:
                      0.25 +
                      0.75 *
                          (0.5 +
                                  0.5 *
                                      _wave(
                                        (controller.value - i * 0.18) % 1.0,
                                      ))
                              .clamp(0.0, 1.0),
                  child: Container(
                    width: 6,
                    height: 6,
                    decoration: BoxDecoration(
                      shape: BoxShape.circle,
                      color: color,
                    ),
                  ),
                ),
              ),
          ],
        );
      },
    );
  }

  double _wave(double t) => t < 0.5 ? (t * 4 - 1) : (3 - t * 4);
}
