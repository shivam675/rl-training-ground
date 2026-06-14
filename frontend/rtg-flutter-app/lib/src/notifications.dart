import 'dart:async';
import 'dart:convert';

import 'package:flutter/foundation.dart';
import 'package:flutter_riverpod/legacy.dart';
import 'package:web_socket_channel/web_socket_channel.dart';

final notificationCenterProvider = ChangeNotifierProvider<NotificationCenter>((
  ref,
) {
  final center = NotificationCenter()..connect();
  ref.onDispose(center.shutdown);
  return center;
});

class AgentNotification {
  AgentNotification({
    required this.id,
    required this.title,
    required this.body,
    required this.severity,
    required this.category,
    required this.nextSteps,
    required this.time,
    required this.replay,
  });

  factory AgentNotification.fromJson(Map<String, dynamic> json) {
    return AgentNotification(
      id: (json['id'] as num?)?.toInt() ?? 0,
      title: json['title']?.toString() ?? 'Notification',
      body: json['body']?.toString() ?? '',
      severity: json['severity']?.toString() ?? 'info',
      category: json['category']?.toString() ?? 'general',
      nextSteps: [
        for (final step in (json['next_steps'] as List? ?? [])) step.toString(),
      ],
      time: DateTime.fromMillisecondsSinceEpoch(
        (((json['timestamp'] as num?)?.toDouble() ?? 0) * 1000).round(),
      ),
      replay: json['replay'] == true,
    );
  }

  final int id;
  final String title;
  final String body;
  final String severity;
  final String category;
  final List<String> nextSteps;
  final DateTime time;
  final bool replay;

  String asText() {
    final steps = nextSteps.isEmpty
        ? ''
        : '\nNext steps:\n${nextSteps.map((s) => '- $s').join('\n')}';
    return '$title\n$body$steps';
  }
}

/// Listens to /ws/agent_events and stores proactive agent notifications.
class NotificationCenter extends ChangeNotifier {
  final notifications = <AgentNotification>[];
  int unread = 0;
  bool connected = false;

  WebSocketChannel? _channel;
  StreamSubscription<dynamic>? _subscription;
  Timer? _reconnectTimer;
  bool _disposed = false;
  final _toastController = StreamController<AgentNotification>.broadcast();

  /// Live (non-replay) notifications, for snackbar toasts.
  Stream<AgentNotification> get toasts => _toastController.stream;

  void connect() {
    if (_disposed) return;
    _subscription?.cancel();
    try {
      _channel?.sink.close();
    } catch (_) {}
    _channel = null;
    try {
      _channel = WebSocketChannel.connect(
        Uri.parse('ws://127.0.0.1:8000/ws/agent_events'),
      );
    } catch (_) {
      _scheduleReconnect();
      return;
    }
    // Handle the `ready` rejection too, or a refused connection becomes an
    // unhandled exception instead of a quiet reconnect.
    _channel!.ready.catchError((Object _) => _handleDisconnect());
    _subscription = _channel!.stream.listen(
      (event) {
        if (_disposed || event is! String) return;
        Map<String, dynamic> data;
        try {
          data = jsonDecode(event) as Map<String, dynamic>;
        } catch (_) {
          return;
        }
        if (data['type'] != 'notification') return;
        connected = true;
        final notification = AgentNotification.fromJson(data);
        if (notifications.any((n) => n.id == notification.id)) return;
        notifications.insert(0, notification);
        if (notifications.length > 100) notifications.removeLast();
        if (!notification.replay) {
          unread += 1;
          _toastController.add(notification);
        }
        notifyListeners();
      },
      onError: (_) => _handleDisconnect(),
      onDone: _handleDisconnect,
      cancelOnError: true,
    );
  }

  void _handleDisconnect() {
    if (_disposed) return;
    connected = false;
    notifyListeners();
    _scheduleReconnect();
  }

  void _scheduleReconnect() {
    _reconnectTimer?.cancel();
    _reconnectTimer = Timer(const Duration(seconds: 3), () {
      if (!_disposed) connect();
    });
  }

  void markAllRead() {
    if (unread == 0) return;
    unread = 0;
    notifyListeners();
  }

  void clear() {
    notifications.clear();
    unread = 0;
    notifyListeners();
  }

  void shutdown() {
    _disposed = true;
    _reconnectTimer?.cancel();
    _subscription?.cancel();
    try {
      _channel?.sink.close();
    } catch (_) {}
    _toastController.close();
  }
}
