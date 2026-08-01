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
    this.toolResult,
    this.toolOk,
    this.providerLabel = '',
    this.diagnoseError,
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
  Map<String, dynamic>? toolResult;
  bool? toolOk; // null while running
  bool needsConfirmation = false;
  String providerLabel;

  /// The training error this notice is about. Non-null makes the bubble render a
  /// **Diagnose** button, so a failed run is one click from being investigated
  /// and fixed instead of the user having to retype the error into the chat.
  final String? diagnoseError;

  /// While true the bubble renders cheap plain text; markdown parsing and
  /// text selection only switch on once the message stops growing.
  bool streaming = false;

  /// What this reply cost: prompt/generated tokens and generation speed, summed
  /// over the provider round(s) that produced it. Zero completion tokens means no
  /// `usage` event arrived (an endpoint that ignores `include_usage`), so the
  /// footer stays hidden rather than claiming a reply was free.
  int promptTokens = 0;
  int completionTokens = 0;
  double? tps;

  bool get hasUsage => completionTokens > 0;

  final DateTime time;

  String asTranscript() {
    return switch (kind) {
      ChatKind.user => 'You: $text',
      ChatKind.agent =>
        providerLabel.isEmpty
            ? 'Agent: $text'
            : 'Agent [$providerLabel]: $text',
      ChatKind.tool => '[tool] $toolName($toolArgs) → $text',
      ChatKind.notice => '[notice] $text',
    };
  }

  Map<String, dynamic> toJson() => {
    'kind': kind.name,
    'text': text,
    if (thinking.isNotEmpty) 'thinking': thinking,
    if (providerLabel.isNotEmpty) 'providerLabel': providerLabel,
    if (diagnoseError != null) 'diagnoseError': diagnoseError,
    if (hasUsage) 'promptTokens': promptTokens,
    if (hasUsage) 'completionTokens': completionTokens,
    if (hasUsage && tps != null) 'tps': tps,
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
      providerLabel: data['providerLabel']?.toString() ?? '',
      diagnoseError: data['diagnoseError']?.toString(),
      time: DateTime.tryParse(data['time']?.toString() ?? ''),
    )
      ..promptTokens = (data['promptTokens'] as num?)?.toInt() ?? 0
      ..completionTokens = (data['completionTokens'] as num?)?.toInt() ?? 0
      ..tps = (data['tps'] as num?)?.toDouble();
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
    _state.chatController = this;
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

  /// The failure message already announced, so a status poll every few seconds
  /// does not add the same notice over and over.
  String? _seenTrainingFailure;

  // Streamed tokens are buffered here and flushed at most every
  // [_flushInterval]: rebuilding per token starves the event loop and freezes
  // the window once replies get long.
  final _chunkBuffer = StringBuffer();
  final _thinkBuffer = StringBuffer();
  String _pendingProviderLabel = '';
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
    _checkTrainingFailure();
  }

  /// A run that failed to start gets one notice in the chat carrying a Diagnose
  /// button. Deliberately not automatic: diagnosis re-patches the config and can
  /// relaunch training, so it stays one deliberate click rather than something
  /// that happens while nobody is watching.
  void _checkTrainingFailure() {
    final status = _state.trainingStatus;
    if (status == null || status['active'] == true) return;
    final message = status['message']?.toString() ?? '';
    if (!message.startsWith('failed')) return;
    if (message == _seenTrainingFailure) return;
    _seenTrainingFailure = message;
    messages.add(
      ChatMessage(
        kind: ChatKind.notice,
        text: 'Training did not start — $message',
        diagnoseError: message,
      ),
    );
    notifyListeners();
    _persistChat();
    _requestForceScroll();
  }

  /// Hand the failure to the agent and let it fix the config. Goes through the
  /// normal chat path, so the model has its whole toolbox (read the config,
  /// inspect the robot, patch) and every step stays visible in the transcript.
  Future<void> diagnoseTrainingFailure(String error) {
    return sendChat(
      'Training failed to start with this error:\n\n$error\n\n'
      'Diagnose the cause, then fix the environment config so training can '
      'start. Check the observations, rewards and terminations against what the '
      'training lane actually supports before proposing a patch, and tell me '
      'what you changed and why.',
    );
  }

  void _requestForceScroll() => forceScrollTick++;

  /// Whether this tool bubble should offer "Configure reward".
  ///
  /// `apply_behavior_goal` deliberately sets NO reward — it calibrates the joints,
  /// observations and fall floor, and the reward is a separate authoring step. That
  /// leaves the user at a dead end unless they know to ask for step two, so the
  /// applied bubble offers it.
  ///
  /// The gate is the config itself, not a flag: once any reward term is enabled the
  /// button disappears on its own, whether the agent authored it, the user did in
  /// the Rewards panel, or an older conversation already had.
  bool offersRewardStep(ChatMessage message) {
    if (message.kind != ChatKind.tool) return false;
    if (message.toolName != 'apply_behavior_goal') return false;
    if (message.toolOk != true || message.needsConfirmation) return false;
    return !_state.hasEnabledRewards;
  }

  /// Ask the agent for step two, in the words the tool's own `next_step` uses.
  Future<void> configureReward() {
    return sendChat(
      'Now author the reward for that goal: call patch_env_config with rewards[] '
      'entries that each carry params.expr, written against the state vocabulary '
      'and the measured numbers you just calibrated. Then tell me each term, its '
      'weight and what it is for, and report any audit warnings you get back.',
    );
  }

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
      bubble = ChatMessage(
        kind: ChatKind.agent,
        text: '',
        providerLabel: _pendingProviderLabel,
      )..streaming = true;
      messages.add(bubble);
    }
    if (_pendingProviderLabel.isNotEmpty) {
      bubble.providerLabel = _pendingProviderLabel;
    }
    bubble.text += text;
    bubble.thinking += think;
    notifyListeners();
  }

  /// Token counts for the round that just finished.
  ///
  /// The event arrives after that round's text, so the newest agent bubble is the
  /// one it describes. A round that produced only a tool call has no bubble yet —
  /// its cost is carried forward and added to the next reply, so nothing is lost
  /// and no bubble is invented just to hold a number.
  void _applyUsage(Map<String, dynamic> event) {
    _flushChunks(); // make sure this round's bubble exists before stamping it
    _pendingPromptTokens += (event['prompt_tokens'] as num?)?.toInt() ?? 0;
    _pendingCompletionTokens += (event['completion_tokens'] as num?)?.toInt() ?? 0;
    final tps = (event['tps'] as num?)?.toDouble();
    if (tps != null) _pendingTps = tps;
    final bubble = messages.isEmpty ? null : messages.last;
    if (bubble == null || bubble.kind != ChatKind.agent) return;
    bubble.promptTokens += _pendingPromptTokens;
    bubble.completionTokens += _pendingCompletionTokens;
    bubble.tps = _pendingTps ?? bubble.tps;
    _pendingPromptTokens = 0;
    _pendingCompletionTokens = 0;
    _pendingTps = null;
    notifyListeners();
  }

  int _pendingPromptTokens = 0;
  int _pendingCompletionTokens = 0;
  double? _pendingTps;

  void _applyRouteEvent(Map<String, dynamic> event) {
    final label = event['label']?.toString() ?? '';
    if (label.isEmpty) return;
    _pendingProviderLabel = label;
    if (messages.isNotEmpty &&
        messages.last.kind == ChatKind.agent &&
        messages.last.streaming) {
      messages.last.providerLabel = label;
      notifyListeners();
    }
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

  void clearChat() => _replaceMessages(const []);

  /// The transcript in the shape a project file stores it — the same subset
  /// that survives a restart, so reopening a project reads like relaunching.
  List<Map<String, dynamic>> chatJson() => [
    for (final m in messages)
      if (!m.streaming && m.kind != ChatKind.tool) m.toJson(),
  ];

  /// Adopt the conversation carried by a project file. `null` or empty (a file
  /// saved before chats were stored) starts a fresh one rather than leaving the
  /// previous project's turns in place.
  void restoreChat(List<dynamic>? stored) {
    _replaceMessages(stored ?? const []);
  }

  void _replaceMessages(List<dynamic> stored) {
    final loaded = [
      for (final item in stored)
        if (item is Map) ChatMessage.fromJson(item.cast<String, dynamic>()),
    ].whereType<ChatMessage>().toList();
    messages
      ..clear()
      ..addAll(loaded.isEmpty ? [welcomeMessage()] : loaded);
    // A project's transcript wins over whatever the startup read is about to
    // find in prefs, and unblocks _persistChat if it has not finished yet.
    _historyLoaded = true;
    notifyListeners();
    _persistChat();
  }

  String transcript() => messages.map((m) => m.asTranscript()).join('\n\n');

  ChatMessage? _pendingConfirmationTool() {
    for (final message in messages.reversed) {
      if (message.kind == ChatKind.tool && message.needsConfirmation) {
        return message;
      }
    }
    return null;
  }

  static bool _isApprovalReply(String text) {
    final normalized = text.trim().toLowerCase();
    return {
      'y',
      'yes',
      'yeah',
      'yep',
      'ok',
      'okay',
      'approve',
      'approved',
      'go ahead',
      'do it',
      'apply',
      'apply it',
      'proceed',
    }.contains(normalized);
  }

  static String _toolDoneText(Map<String, dynamic>? result, String? error) {
    if (error != null) return error;
    if (result?['change_set'] is Map || result?['config'] is Map) {
      return 'applied';
    }
    return 'done';
  }

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
      // A project opened while this was in flight already set the transcript.
      if (_historyLoaded) return;
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
    try {
      final prefs = await SharedPreferences.getInstance();
      await prefs.setString(_chatPrefsKey, jsonEncode(chatJson()));
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
            'Loaded `$path`. Describe in natural language what this robot should learn. I will inspect the robot, propose observations, actions, rewards and terminations, then ask before applying changes.',
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
      tool.toolResult = result;
      final error = result['error']?.toString();
      tool.toolOk = error == null;
      tool.text = _toolDoneText(result, error);
    } catch (e) {
      tool.toolOk = false;
      tool.text = '$e';
    }
    notifyListeners();
    _persistChat();
    _requestForceScroll();
  }

  Future<void> undoConfigChange(ChatMessage source) async {
    messages.add(
      ChatMessage(
        kind: ChatKind.tool,
        text: 'running...',
        toolName: 'undo_config_change',
      ),
    );
    notifyListeners();
    final tool = messages.last;
    final result = await _state.undoConfigChange();
    tool.toolResult = result;
    final error = result['error']?.toString();
    tool.toolOk = error == null;
    tool.text = _toolDoneText(result, error);
    source.toolResult = {...?source.toolResult, 'undo_used': error == null};
    notifyListeners();
    _persistChat();
  }

  Future<void> sendChat(String rawText) async {
    final text = rawText.trim();
    if (text.isEmpty || sending) return;
    final pendingTool = _pendingConfirmationTool();
    if (pendingTool != null && _isApprovalReply(text)) {
      messages.add(ChatMessage(kind: ChatKind.user, text: text));
      notifyListeners();
      _persistChat();
      _requestForceScroll();
      await confirmTool(pendingTool);
      return;
    }
    messages.add(ChatMessage(kind: ChatKind.user, text: text));
    _pendingProviderLabel = '';
    _pendingPromptTokens = 0;
    _pendingCompletionTokens = 0;
    _pendingTps = null;
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
          case 'route':
            _applyRouteEvent(event);
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
            }
            if (result is Map) {
              tool.toolResult = result.cast<String, dynamic>();
            }
            if (!needsConfirm) {
              tool.text = _toolDoneText(tool.toolResult, error);
            }
            notifyListeners();
          case 'usage':
            _applyUsage(event);
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
        messages.add(
          ChatMessage(
            kind: ChatKind.agent,
            text: 'No response.',
            providerLabel: _pendingProviderLabel,
          ),
        );
      }
      _pendingProviderLabel = '';
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
      _pendingProviderLabel = '';
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
