import 'dart:async';
import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:media_kit/media_kit.dart';
import 'package:media_kit_video/media_kit_video.dart';
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
///  - RECORDING (recording eval, still headless) → "recording…" placeholder.
///  - PREPARING (recording stopped, Isaac still closing the mp4) → a spinner that
///    says so. Not cosmetic: `active` goes false the instant Stop is pressed but
///    the file needs seconds more, and a silent blank page in that window reads as
///    a failure.
///  - VIDEO → the recorded mp4 in a real player (play/pause/seek/save).
///  - BROADCAST / IDLE → relayed frames from the legacy SB3 evaluation lane, or
///    "Nothing in view".
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
    if (state.videoFinalizing) return const _PreparingVideo();
    final videoRun = state.videoRunName;
    if (videoRun != null) {
      // Keyed by run so switching recordings rebuilds the player with the new file
      // rather than leaving the old one loaded.
      return _RecordedVideoView(runName: videoRun, key: ValueKey(videoRun));
    }
    // A recording that finished with no file: say so with the lane's own message,
    // rather than the generic "Nothing in view" that hides the failure.
    if (eval['video_run'] != null && eval['video_ready'] == false) {
      return _RecordingFailed(message: '${eval['message'] ?? ''}');
    }
    return const _BroadcastFrameView();
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
      await ref
          .read(appStateProvider)
          .api
          .postJson('/isaac/stream_client/launch', {});
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

/// The recorded evaluation, in a real player.
///
/// Streamed straight from `GET /runs/<name>/video` (Range-served), so the player
/// owns the timeline: play/pause, seek, duration, volume. That is why the backend
/// no longer decodes the mp4 into JPEGs — a frame push has no timeline to offer,
/// and the first frame had to wait for a decode thread to spin up.
class _RecordedVideoView extends ConsumerStatefulWidget {
  const _RecordedVideoView({required this.runName, super.key});

  final String runName;

  @override
  ConsumerState<_RecordedVideoView> createState() => _RecordedVideoViewState();
}

class _RecordedVideoViewState extends ConsumerState<_RecordedVideoView> {
  late final Player _player = Player();
  late final VideoController _controller = VideoController(_player);
  bool _saving = false;

  @override
  void initState() {
    super.initState();
    final state = ref.read(appStateProvider);
    // A cancelled recording is a truncated fragmented mp4 — playable, but its
    // duration is unknown until enough of it is read, so don't loop on it.
    _player.open(Media(state.videoUrl(widget.runName)), play: true);
  }

  @override
  void dispose() {
    _player.dispose();
    super.dispose();
  }

  Future<void> _save() async {
    setState(() => _saving = true);
    String message;
    try {
      final path = await ref.read(appStateProvider).saveRunVideo(widget.runName);
      message = path == null ? 'Save cancelled.' : 'Saved to $path';
    } catch (e) {
      message = 'Could not save the recording: $e';
    }
    if (!mounted) return;
    setState(() => _saving = false);
    ScaffoldMessenger.of(context)
      ..clearSnackBars()
      ..showSnackBar(SnackBar(content: Text(message)));
  }

  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;
    return Column(
      children: [
        Expanded(
          child: Video(
            controller: _controller,
            // media_kit's desktop controls are the seekbar/play/pause/volume set
            // this page needs; no reason to hand-roll them.
            controls: AdaptiveVideoControls,
          ),
        ),
        Padding(
          padding: const EdgeInsets.fromLTRB(12, 6, 12, 10),
          child: Row(
            children: [
              Icon(Icons.movie_outlined, size: 16, color: scheme.primary),
              const SizedBox(width: 8),
              Expanded(
                child: Text(
                  'Recording · ${widget.runName}',
                  overflow: TextOverflow.ellipsis,
                  style: const TextStyle(fontSize: 12.5),
                ),
              ),
              OutlinedButton.icon(
                onPressed: _saving ? null : _save,
                icon: _saving
                    ? const SizedBox(
                        width: 14,
                        height: 14,
                        child: CircularProgressIndicator(strokeWidth: 2),
                      )
                    : const Icon(Icons.download_outlined, size: 16),
                label: Text(_saving ? 'Saving…' : 'Save a copy'),
              ),
              const SizedBox(width: 8),
              TextButton.icon(
                onPressed: () =>
                    ref.read(appStateProvider).showRunVideo(null),
                icon: const Icon(Icons.close, size: 16),
                label: const Text('Close'),
              ),
            ],
          ),
        ),
      ],
    );
  }
}

/// Frames relayed over `/ws/simulation` — today only the legacy SB3 evaluation
/// lane, which renders a live PyBullet world frame by frame. Isaac recordings go
/// through the player above instead. Deliberately chrome-free: there is no
/// timeline to control on a live broadcast.
class _BroadcastFrameView extends StatefulWidget {
  const _BroadcastFrameView();

  @override
  State<_BroadcastFrameView> createState() => _BroadcastFrameViewState();
}

class _BroadcastFrameViewState extends State<_BroadcastFrameView> {
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
        // Binary = a rendered frame; JSON status/idle text is ignored. Frames
        // only arrive while a worker is broadcasting, so an idle socket shows
        // "Nothing in view" rather than the live PyBullet world.
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
        child: Image.memory(
          _frame!,
          gaplessPlayback: true,
          fit: BoxFit.contain,
        ),
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

/// The gap between "Stop" and a playable file. Isaac exits on the STOP sentinel and
/// closes the mp4 on its way out, which takes a few seconds — and the evaluation is
/// already reported inactive by then. Saying so is the whole point of this state:
/// before it, the page went blank and looked broken for exactly that long.
class _PreparingVideo extends StatelessWidget {
  const _PreparingVideo();

  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;
    return Center(
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          const SizedBox(
            width: 26,
            height: 26,
            child: CircularProgressIndicator(strokeWidth: 2.5),
          ),
          const SizedBox(height: 16),
          const Text(
            'Preparing the video…',
            style: TextStyle(fontSize: 16, fontWeight: FontWeight.w700),
          ),
          const SizedBox(height: 8),
          SizedBox(
            width: 420,
            child: Text(
              'Isaac is finishing the recording file. It starts playing here on its '
              'own — usually a few seconds.',
              textAlign: TextAlign.center,
              style: TextStyle(
                fontSize: 12.5,
                height: 1.4,
                color: scheme.onSurface.withValues(alpha: 0.7),
              ),
            ),
          ),
        ],
      ),
    );
  }
}

/// The recording ended without producing a file — usually the Isaac child died
/// during Kit boot, before capture began.
class _RecordingFailed extends StatelessWidget {
  const _RecordingFailed({required this.message});

  final String message;

  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;
    return Center(
      child: Padding(
        padding: const EdgeInsets.all(24),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(Icons.videocam_off_outlined, size: 46, color: scheme.error),
            const SizedBox(height: 14),
            const Text(
              'No video was recorded',
              style: TextStyle(fontSize: 16, fontWeight: FontWeight.w700),
            ),
            const SizedBox(height: 8),
            SizedBox(
              width: 460,
              child: Text(
                message.isEmpty
                    ? 'The evaluation ended before any frames were captured. The '
                          'run log has the details.'
                    : message,
                textAlign: TextAlign.center,
                style: TextStyle(
                  fontSize: 12.5,
                  height: 1.4,
                  color: scheme.onSurface.withValues(alpha: 0.7),
                ),
              ),
            ),
          ],
        ),
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
              'The evaluation runs headless (no live stream). The video plays '
              'here automatically when it finishes — or when you cancel, showing '
              'what was captured up to that point. It stays with the run, so you '
              'can replay it any time from the Evaluate tab.',
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
