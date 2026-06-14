import 'dart:async';
import 'dart:convert';

import 'package:flutter/foundation.dart';
import 'package:flutter_riverpod/legacy.dart';
import 'package:shared_preferences/shared_preferences.dart';

import '../app_state.dart';

enum ChatKind { user, agent, tool, notice }

class ChatMessage {
  ChatMessage({
    required this.kind,
    required this.text,
    this.thinking = '',
    this.toolName,
    this.toolArgs,
    this.toolArgsRaw,
    this.toolOk,
    DateTime? time,
  }) : time = time ?? DateTime.now();

  final ChatKind kind;
  String text;

  /// Model reasoning (qwen3 `thinking` field or inline `<think>` blocks),
  /// shown in a collapsible section in the bubble.
  String thinking;
  final String? toolName;
  final String? toolArgs;
  final Map<String, dynamic>? toolArgsRaw;
  bool? toolOk; // null while running
  bool needsConfirmation = false;

  /// While true the bubble renders cheap plain text; markdown parsing and
  /// text selection only switch on once the message stops growing.
  bool streaming = false;

  final DateTime time;

  String asTranscript() {
    return switch (kind) {
      ChatKind.user => 'You: $text',
      ChatKind.agent => 'Agent: $text',
      ChatKind.tool => '[tool] $toolName($toolArgs) → $text',
      ChatKind.notice => '[notice] $text',
    };
  }

  Map<String, dynamic> toJson() => {
    'kind': kind.name,
    'text': text,
    if (thinking.isNotEmpty) 'thinking': thinking,
    'time': time.toIso8601String(),
  };

  static ChatMessage? fromJson(Map<String, dynamic> data) {
    final kindName = data['kind']?.toString();
    ChatKind? kind;
    for (final item in ChatKind.values) {
      if (item.name == kindName) {
        kind = item;
        break;
      }
    }
    if (kind == null || kind == ChatKind.tool) return null;
    final text = data['text']?.toString() ?? '';
    final thinking = data['thinking']?.toString() ?? '';
    if (text.isEmpty && thinking.isEmpty) return null;
    return ChatMessage(
      kind: kind,
      text: text,
      thinking: thinking,
      time: DateTime.tryParse(data['time']?.toString() ?? ''),
    );
  }
}

ChatMessage welcomeMessage() => ChatMessage(
  kind: ChatKind.agent,
  text:
      'Agent ready. I can inspect the robot, test rewards, start or stop '
      'training and compare runs for you — just ask.',
);

/// Owns the agent conversation: message list, the streaming/think-tag parser,
/// persistence and the network round-trip. Lifted out of the old AgentsPanel so
/// the same conversation is shared by every surface that renders it (today the
/// co-pilot dock) instead of each widget keeping its own copy.
final chatControllerProvider = ChangeNotifierProvider<ChatController>((ref) {
  return ChatController(ref.read(appStateProvider));
});

class ChatController extends ChangeNotifier {
  ChatController(this._state) {
    _seenRobotLoadRevision = _state.robotLoadRevision;
    _state.addListener(_onAppStateChanged);
    unawaited(_loadChatHistory());
  }

  final AppState _state;
  static const _chatPrefsKey = 'easyrtg.agent_chat.v1';
  // One unified assistant; the multi-agent selector was removed.
  static const agent = 'helper';

  final messages = <ChatMessage>[welcomeMessage()];
  bool sending = false;
  bool _historyLoaded = false;
  int _seenRobotLoadRevision = 0;

  // Streamed tokens are buffered here and flushed at most every
  // [_flushInterval]: rebuilding per token starves the event loop and freezes
  // the window once replies get long.
  final _chunkBuffer = StringBuffer();
  final _thinkBuffer = StringBuffer();
  Timer? _flushTimer;
  static const _flushInterval = Duration(milliseconds: 66);

  // Inline `<think>...</think>` parser state (some models stream reasoning as
  // tags inside the content instead of a separate `thinking` field). Carries a
  // partial tag fragment across chunk boundaries.
  bool _inThink = false;
  String _tagCarry = '';

  /// Bumped when the conversation wants the view pinned to the bottom even if
  /// the user had scrolled up (after sending, or a proactive agent prompt). The
  /// view widget owns the ScrollController and watches this.
  int forceScrollTick = 0;

  bool get hasHistory => messages.length > 1;

  bool get showTyping =>
      sending &&
      (messages.isEmpty ||
          messages.last.kind != ChatKind.agent ||
          (messages.last.text.isEmpty && messages.last.thinking.isEmpty));

  @override
  void dispose() {
    _state.removeListener(_onAppStateChanged);
    _flushTimer?.cancel();
    super.dispose();
  }

  void _onAppStateChanged() {
    // A freshly loaded robot earns one proactive "what should it learn?" prompt.
    if (_state.robotLoadRevision != _seenRobotLoadRevision) {
      _seenRobotLoadRevision = _state.robotLoadRevision;
      final path = _state.lastLoadedRobotPath;
      if (path != null && path.isNotEmpty) _addRobotLoadedPrompt(path);
    }
  }

  void _requestForceScroll() => forceScrollTick++;

  /// Buffer a content chunk, splitting out any inline `<think>` reasoning.
  void _queueChunk(String chunk) {
    final visible = _separateThink(chunk);
    if (visible.isNotEmpty) _chunkBuffer.write(visible);
    _flushTimer ??= Timer(_flushInterval, _flushChunks);
  }

  /// Buffer reasoning from the model's separate `thinking` stream.
  void _queueThinking(String chunk) {
    if (chunk.isEmpty) return;
    _thinkBuffer.write(chunk);
    _flushTimer ??= Timer(_flushInterval, _flushChunks);
  }

  /// Split inline `<think>...</think>` out of [incoming]: visible text is
  /// returned, reasoning is routed into [_thinkBuffer]. A partial tag at the
  /// end is held in [_tagCarry] until the next chunk completes it.
  String _separateThink(String incoming) {
    final visible = StringBuffer();
    var s = _tagCarry + incoming;
    _tagCarry = '';
    while (s.isNotEmpty) {
      if (!_inThink) {
        final open = s.indexOf('<think>');
        if (open < 0) {
          final keep = _partialTagTail(s, '<think>');
          visible.write(s.substring(0, s.length - keep));
          _tagCarry = s.substring(s.length - keep);
          break;
        }
        visible.write(s.substring(0, open));
        s = s.substring(open + '<think>'.length);
        _inThink = true;
      } else {
        final close = s.indexOf('</think>');
        if (close < 0) {
          final keep = _partialTagTail(s, '</think>');
          _thinkBuffer.write(s.substring(0, s.length - keep));
          _tagCarry = s.substring(s.length - keep);
          break;
        }
        _thinkBuffer.write(s.substring(0, close));
        s = s.substring(close + '</think>'.length);
        _inThink = false;
      }
    }
    return visible.toString();
  }

  /// Length of the longest suffix of [s] that is a prefix of [tag] — i.e. a
  /// half-arrived tag like "<thin" we must not emit yet.
  static int _partialTagTail(String s, String tag) {
    final max = s.length < tag.length - 1 ? s.length : tag.length - 1;
    for (var len = max; len > 0; len--) {
      if (tag.startsWith(s.substring(s.length - len))) return len;
    }
    return 0;
  }

  void _flushChunks() {
    _flushTimer?.cancel();
    _flushTimer = null;
    if (_chunkBuffer.isEmpty && _thinkBuffer.isEmpty) return;
    final text = _chunkBuffer.toString();
    final think = _thinkBuffer.toString();
    _chunkBuffer.clear();
    _thinkBuffer.clear();
    final ChatMessage bubble;
    if (messages.isNotEmpty &&
        messages.last.kind == ChatKind.agent &&
        messages.last.streaming) {
      bubble = messages.last;
    } else {
      bubble = ChatMessage(kind: ChatKind.agent, text: '')..streaming = true;
      messages.add(bubble);
    }
    bubble.text += text;
    bubble.thinking += think;
    notifyListeners();
  }

  /// Stop the live bubble growing and switch it to selectable markdown.
  void _finishStreamingBubble() {
    _flushChunks();
    // Reset the inline-think parser for the next bubble.
    _inThink = false;
    _tagCarry = '';
    for (final message in messages) {
      message.streaming = false;
    }
    notifyListeners();
    _persistChat();
  }

  void clearChat() {
    messages
      ..clear()
      ..add(welcomeMessage());
    notifyListeners();
    _persistChat();
  }

  String transcript() => messages.map((m) => m.asTranscript()).join('\n\n');

  List<Map<String, String>> _historyForRequest() {
    final turns = <Map<String, String>>[];
    for (final m in messages) {
      if (m.kind == ChatKind.user) {
        turns.add({'role': 'user', 'content': m.text});
      } else if (m.kind == ChatKind.agent && m.text.isNotEmpty) {
        turns.add({'role': 'assistant', 'content': m.text});
      }
    }
    // Skip the canned welcome message; keep the recent window.
    final start = turns.length > 12 ? turns.length - 12 : 0;
    return turns.sublist(start);
  }

  Future<void> _loadChatHistory() async {
    try {
      final prefs = await SharedPreferences.getInstance();
      final raw = prefs.getString(_chatPrefsKey);
      if (raw == null || raw.isEmpty) {
        _historyLoaded = true;
        return;
      }
      final decoded = jsonDecode(raw);
      if (decoded is! List) {
        _historyLoaded = true;
        return;
      }
      final loaded = [
        for (final item in decoded)
          if (item is Map) ChatMessage.fromJson(item.cast<String, dynamic>()),
      ].whereType<ChatMessage>().toList();
      messages
        ..clear()
        ..addAll(loaded.isEmpty ? [welcomeMessage()] : loaded);
      notifyListeners();
    } catch (_) {
      // Corrupt local history should not block the chat.
    } finally {
      _historyLoaded = true;
    }
  }

  Future<void> _persistChat() async {
    if (!_historyLoaded) return;
    final stableMessages = messages
        .where((m) => !m.streaming && m.kind != ChatKind.tool)
        .toList();
    try {
      final prefs = await SharedPreferences.getInstance();
      await prefs.setString(
        _chatPrefsKey,
        jsonEncode(stableMessages.map((m) => m.toJson()).toList()),
      );
    } catch (_) {}
  }

  void _addRobotLoadedPrompt(String path) {
    final alreadyAsked = messages.any(
      (m) => m.kind == ChatKind.agent && m.text.contains(path),
    );
    if (alreadyAsked) return;
    messages.add(
      ChatMessage(
        kind: ChatKind.agent,
        text:
            'Loaded `$path`. What should this robot learn: walk, run, stand, sit Japanese-style, balance, or reach a target? Reply with one goal and I will draft the reward in Rewards.',
      ),
    );
    notifyListeners();
    _persistChat();
    _requestForceScroll();
  }

  Future<void> confirmTool(ChatMessage tool) async {
    tool.needsConfirmation = false;
    tool.toolOk = null;
    tool.text = 'running…';
    notifyListeners();
    try {
      final result = await _state.executeAgentTool(
        tool.toolName ?? '',
        tool.toolArgsRaw ?? {},
      );
      final error = result['error']?.toString();
      tool.toolOk = error == null;
      tool.text = error ?? 'done';
    } catch (e) {
      tool.toolOk = false;
      tool.text = '$e';
    }
    notifyListeners();
  }

  Future<void> sendChat(String rawText) async {
    final text = rawText.trim();
    if (text.isEmpty || sending) return;
    messages.add(ChatMessage(kind: ChatKind.user, text: text));
    sending = true;
    notifyListeners();
    _persistChat();
    _requestForceScroll();
    final history = _historyForRequest();
    try {
      await for (final event in _state.chatEvents(
        text,
        agent: agent,
        history: history,
      )) {
        switch (event['type']) {
          case 'chunk':
            _queueChunk(event['text']?.toString() ?? '');
          case 'thinking':
            _queueThinking(event['text']?.toString() ?? '');
          case 'tool_call':
            // Flush buffered text first so ordering stays correct, and end the
            // current bubble: post-tool text belongs to a new one.
            _finishStreamingBubble();
            messages.add(
              ChatMessage(
                kind: ChatKind.tool,
                text: 'running…',
                toolName: event['name']?.toString() ?? 'tool',
                toolArgs: _argsSummary(event['args']),
                toolArgsRaw: event['args'] is Map
                    ? (event['args'] as Map).cast<String, dynamic>()
                    : null,
              ),
            );
            notifyListeners();
          case 'tool_result':
            final result = event['result'];
            final needsConfirm =
                result is Map && result['requires_confirmation'] == true;
            final error = result is Map ? result['error']?.toString() : null;
            final tool = messages.lastWhere(
              (m) => m.kind == ChatKind.tool && m.toolOk == null,
              orElse: () => messages.last,
            );
            if (needsConfirm) {
              tool.needsConfirmation = true;
              tool.text = 'awaiting your confirmation';
            } else {
              tool.toolOk = error == null;
              tool.text = error ?? 'done';
            }
            notifyListeners();
          case 'notice':
            _finishStreamingBubble();
            messages.add(
              ChatMessage(
                kind: ChatKind.notice,
                text: event['text']?.toString() ?? '',
              ),
            );
            notifyListeners();
        }
      }
      _finishStreamingBubble();
      if (messages.isEmpty || messages.last.kind == ChatKind.user) {
        messages.add(ChatMessage(kind: ChatKind.agent, text: 'No response.'));
      }
      sending = false;
      notifyListeners();
      _persistChat();
    } catch (e) {
      _finishStreamingBubble();
      messages.add(
        ChatMessage(
          kind: ChatKind.agent,
          text: '⚠ ${e.toString().replaceFirst('Exception: ', '')}',
        ),
      );
      sending = false;
      notifyListeners();
      _persistChat();
    }
  }

  static String _argsSummary(dynamic args) {
    if (args is! Map || args.isEmpty) return '';
    return args.entries.map((e) => '${e.key}: ${e.value}').join(', ');
  }
}
