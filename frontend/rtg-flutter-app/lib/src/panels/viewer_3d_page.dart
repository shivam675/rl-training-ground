import 'dart:async';
import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:web_socket_channel/web_socket_channel.dart';

import '../app_state.dart';

/// The dedicated 3D page. The user is taken here when they start an Isaac
/// training live-view or an evaluation, and it's also reachable from the rail.
///
/// On Windows the live WebRTC stream can't be embedded in-app — NVIDIA's
/// browser client is a Docker web viewer supported only on Ubuntu/DGX, so live
/// view runs in NVIDIA's standalone desktop "Isaac Sim WebRTC Streaming Client"
/// (connects to 127.0.0.1:49100). The fully in-app 3D path is Record→video.
/// Modes here are driven entirely by backend status:
///
///  - LIVE (training live-view / non-recording eval) → a panel that launches
///    the desktop streaming client.
///  - RECORDING (recording eval, still headless) → "recording…" placeholder;
///    the mp4 replays here (via `/ws/simulation`) when it finishes or is
///    cancelled.
///  - VIDEO / IDLE → the recorded mp4 frames, or "Nothing in view".
class Viewer3DPage extends ConsumerWidget {
  const Viewer3DPage({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final state = ref.watch(appStateProvider);
    final training = state.trainingStatus ?? const {};
    final eval = state.evaluationStatus ?? const {};

    final liveTraining =
        training['active'] == true && training['stream_url'] != null;
    final liveEval = eval['active'] == true && eval['stream'] is Map;
    final recording = eval['active'] == true && eval['recording'] == true;

    if (liveTraining || liveEval) return const _LiveStreamPanel();
    if (recording) return const _RecordingPlaceholder();
    return const _IsaacVideoView();
  }
}

/// Live view: launch NVIDIA's desktop streaming client (the only live client on
/// Windows). Isaac's Kit process needs ~30-60s to boot before the stream is up;
/// the client retries on its own, so launching early is fine.
class _LiveStreamPanel extends ConsumerStatefulWidget {
  const _LiveStreamPanel();

  @override
  ConsumerState<_LiveStreamPanel> createState() => _LiveStreamPanelState();
}

class _LiveStreamPanelState extends ConsumerState<_LiveStreamPanel> {
  bool _launching = false;

  Future<void> _launch() async {
    setState(() => _launching = true);
    String? error;
    try {
      await ref.read(appStateProvider).api.postJson(
        '/isaac/stream_client/launch',
        {},
      );
    } catch (e) {
      error = e.toString();
    }
    if (!mounted) return;
    setState(() => _launching = false);
    ScaffoldMessenger.of(context)
      ..clearSnackBars()
      ..showSnackBar(
        SnackBar(
          content: Text(
            error == null
                ? 'Streaming client launched — connect it to 127.0.0.1.'
                : 'Could not launch the streaming client: $error',
          ),
        ),
      );
  }

  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;
    return Center(
      child: Padding(
        padding: const EdgeInsets.all(24),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(Icons.cast_connected, size: 48, color: scheme.primary),
            const SizedBox(height: 14),
            const Text(
              'Live view runs in the streaming client',
              style: TextStyle(fontSize: 16, fontWeight: FontWeight.w700),
            ),
            const SizedBox(height: 8),
            SizedBox(
              width: 460,
              child: Text(
                "Isaac Sim streams the live world over WebRTC to NVIDIA's "
                'desktop "Isaac Sim WebRTC Streaming Client" (it can\'t be '
                'embedded in-app on Windows). Launch it, connect to 127.0.0.1, '
                'and it shows the robot once Isaac finishes booting (~30–60s). '
                'For a fully in-app 3D view, use Evaluate → Record video.',
                textAlign: TextAlign.center,
                style: TextStyle(
                  fontSize: 12.5,
                  height: 1.45,
                  color: scheme.onSurface.withValues(alpha: 0.72),
                ),
              ),
            ),
            const SizedBox(height: 18),
            FilledButton.icon(
              onPressed: _launching ? null : _launch,
              icon: _launching
                  ? const SizedBox(
                      width: 16,
                      height: 16,
                      child: CircularProgressIndicator(strokeWidth: 2),
                    )
                  : const Icon(Icons.open_in_new, size: 18),
              label: Text(_launching ? 'Launching…' : 'Launch live viewer'),
            ),
          ],
        ),
      ),
    );
  }
}

/// Minimal viewer for a recorded-evaluation replay: shows the JPEG frames the
/// backend broadcasts over `/ws/simulation`, or "Nothing in view" when none are
/// flowing. Deliberately chrome-free — no camera controls, no live PyBullet.
class _IsaacVideoView extends StatefulWidget {
  const _IsaacVideoView();

  @override
  State<_IsaacVideoView> createState() => _IsaacVideoViewState();
}

class _IsaacVideoViewState extends State<_IsaacVideoView> {
  WebSocketChannel? _channel;
  StreamSubscription<dynamic>? _sub;
  Timer? _reconnect;
  Uint8List? _frame;
  bool _disposed = false;

  @override
  void initState() {
    super.initState();
    _connect();
  }

  void _connect() {
    _sub?.cancel();
    try {
      _channel?.sink.close();
    } catch (_) {}
    try {
      _channel = WebSocketChannel.connect(
        Uri.parse('ws://127.0.0.1:8000/ws/simulation'),
      );
    } catch (_) {
      _scheduleReconnect();
      return;
    }
    _sub = _channel!.stream.listen(
      (data) {
        // Binary = a video frame; JSON status/idle text is ignored — this view
        // only ever shows the recorded Isaac playback, never the live sim.
        if (data is List<int> && mounted) {
          setState(() => _frame = Uint8List.fromList(data));
        }
      },
      onError: (_) => _scheduleReconnect(),
      onDone: _scheduleReconnect,
    );
  }

  void _scheduleReconnect() {
    if (_disposed) return;
    _reconnect?.cancel();
    _reconnect = Timer(const Duration(seconds: 2), _connect);
  }

  @override
  void dispose() {
    _disposed = true;
    _reconnect?.cancel();
    _sub?.cancel();
    try {
      _channel?.sink.close();
    } catch (_) {}
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    if (_frame == null) return const _NothingInView();
    return ColoredBox(
      color: Colors.black,
      child: Center(
        child: Image.memory(_frame!, gaplessPlayback: true, fit: BoxFit.contain),
      ),
    );
  }
}

/// Idle state for the 3D page: no live stream, no recording, no video playing.
class _NothingInView extends StatelessWidget {
  const _NothingInView();

  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;
    return Center(
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(
            Icons.visibility_off_outlined,
            size: 46,
            color: scheme.onSurface.withValues(alpha: 0.4),
          ),
          const SizedBox(height: 14),
          const Text(
            'Nothing in view',
            style: TextStyle(fontSize: 16, fontWeight: FontWeight.w700),
          ),
          const SizedBox(height: 8),
          SizedBox(
            width: 420,
            child: Text(
              'Start an Isaac training with "Live view", or evaluate a run, to '
              'see the robot here.',
              textAlign: TextAlign.center,
              style: TextStyle(
                fontSize: 12.5,
                height: 1.4,
                color: scheme.onSurface.withValues(alpha: 0.65),
              ),
            ),
          ),
        ],
      ),
    );
  }
}

/// Shown while a recording evaluation runs headless — no live stream yet.
class _RecordingPlaceholder extends StatelessWidget {
  const _RecordingPlaceholder();

  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;
    return Center(
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(Icons.videocam_outlined, size: 48, color: scheme.primary),
          const SizedBox(height: 14),
          const Text(
            'Recording the policy…',
            style: TextStyle(fontSize: 16, fontWeight: FontWeight.w700),
          ),
          const SizedBox(height: 8),
          SizedBox(
            width: 420,
            child: Text(
              'The evaluation runs headless (no live stream). The video will '
              'play here automatically when it finishes — or when you cancel, '
              'showing what was captured up to that point.',
              textAlign: TextAlign.center,
              style: TextStyle(
                fontSize: 12.5,
                height: 1.4,
                color: scheme.onSurface.withValues(alpha: 0.7),
              ),
            ),
          ),
          const SizedBox(height: 18),
          const SizedBox(
            width: 22,
            height: 22,
            child: CircularProgressIndicator(strokeWidth: 2.5),
          ),
        ],
      ),
    );
  }
}
