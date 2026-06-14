import 'dart:async';
import 'dart:convert';
import 'dart:typed_data';
import 'dart:ui' as ui;

import 'package:flutter/gestures.dart';
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:web_socket_channel/web_socket_channel.dart';

import '../app_state.dart';
import '../widgets/common.dart';

class SimulationViewport extends ConsumerStatefulWidget {
  const SimulationViewport({super.key});

  @override
  ConsumerState<SimulationViewport> createState() => _SimulationViewportState();
}

class _SimulationViewportState extends ConsumerState<SimulationViewport> {
  WebSocketChannel? channel;
  StreamSubscription<dynamic>? subscription;
  Uint8List? frame;
  ui.Image? rawFrame;
  bool decodeBusy = false;
  Timer? reconnectTimer;
  String status = 'connecting';
  String? playbackLabel; // set while evaluation playback owns the viewport
  bool connected = false;
  bool trainingActive = false;
  bool paused = false;
  Offset? lastPosition;
  int dragButtons = 0;
  Size? lastSize;
  bool disposed = false;

  @override
  void initState() {
    super.initState();
    connect();
  }

  void connect() {
    subscription?.cancel();
    try {
      channel?.sink.close();
    } catch (_) {}
    channel = null;
    setState(() {
      status = 'connecting';
      connected = false;
    });
    try {
      channel = WebSocketChannel.connect(
        Uri.parse('ws://127.0.0.1:8000/ws/simulation'),
      );
    } catch (_) {
      setState(() {
        status = 'disconnected';
        connected = false;
      });
      scheduleReconnect();
      return;
    }
    // Connection failures surface via `ready`; left unhandled they become an
    // unhandled-exception crash instead of flowing into the stream's onError.
    channel!.ready.catchError((Object _) {
      if (mounted && !disposed) {
        setState(() {
          status = 'disconnected';
          connected = false;
        });
        scheduleReconnect();
      }
    });
    subscription = channel!.stream.listen(
      (event) {
        if (!mounted || disposed) return;
        if (event is Uint8List) {
          handleFrameBytes(event);
        } else if (event is List<int>) {
          handleFrameBytes(Uint8List.fromList(event));
        } else if (event is String) {
          final data = jsonDecode(event) as Map<String, dynamic>;
          setState(() {
            trainingActive = data['training'] == true;
            status = trainingActive
                ? 'Training…'
                : '${data['renderer'] ?? 'stream'} · ${data['fps'] ?? 0} fps';
            playbackLabel = data['mode']?.toString();
            connected = true;
          });
        }
      },
      onError: (_) {
        if (mounted && !disposed) {
          setState(() {
            status = 'disconnected';
            connected = false;
          });
          scheduleReconnect();
        }
      },
      onDone: () {
        if (mounted && !disposed) {
          setState(() {
            status = 'disconnected';
            connected = false;
          });
          scheduleReconnect();
        }
      },
      cancelOnError: true,
    );
  }

  void handleFrameBytes(Uint8List bytes) {
    connected = true;
    // A frame means rendering resumed — training (which pauses live render) is
    // no longer holding the viewport.
    trainingActive = false;
    if (_isRawFrame(bytes)) {
      decodeRawFrame(bytes);
      return;
    }
    // Swap first, dispose after: disposing an image still referenced by the
    // current RawImage crashes the raster thread mid-paint.
    final old = rawFrame;
    setState(() {
      rawFrame = null;
      frame = bytes;
    });
    if (old != null) {
      WidgetsBinding.instance.addPostFrameCallback((_) => old.dispose());
    }
  }

  bool _isRawFrame(Uint8List bytes) {
    return bytes.length > 12 &&
        bytes[0] == 0x52 &&
        bytes[1] == 0x54 &&
        bytes[2] == 0x47 &&
        bytes[3] == 0x46;
  }

  void decodeRawFrame(Uint8List bytes) {
    if (decodeBusy) return;
    decodeBusy = true;
    final data = ByteData.sublistView(bytes);
    final width = data.getUint32(4, Endian.little);
    final height = data.getUint32(8, Endian.little);
    final pixels = bytes.sublist(12);
    ui.decodeImageFromPixels(pixels, width, height, ui.PixelFormat.rgba8888, (
      image,
    ) {
      decodeBusy = false;
      if (!mounted || disposed) {
        image.dispose();
        return;
      }
      final old = rawFrame;
      setState(() {
        rawFrame = image;
        frame = null;
      });
      if (old != null) {
        WidgetsBinding.instance.addPostFrameCallback((_) => old.dispose());
      }
    });
  }

  void scheduleReconnect() {
    reconnectTimer?.cancel();
    reconnectTimer = Timer(const Duration(seconds: 2), () {
      if (!disposed && mounted) connect();
    });
  }

  void send(Map<String, dynamic> data) {
    // Writing to a closed sink throws a StateError that would otherwise
    // crash the app when the user interacts mid-disconnect.
    if (channel == null || !connected && status == 'disconnected') return;
    try {
      channel?.sink.add(jsonEncode(data));
    } catch (_) {
      // Sink already closed; the reconnect timer will restore the stream.
    }
  }

  @override
  void dispose() {
    disposed = true;
    reconnectTimer?.cancel();
    subscription?.cancel();
    try {
      channel?.sink.close();
    } catch (_) {}
    rawFrame?.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return LayoutBuilder(
      builder: (context, constraints) {
        final size = Size(constraints.maxWidth, constraints.maxHeight);
        final dpr = MediaQuery.devicePixelRatioOf(context);
        final streamScale = ref.watch(appStateProvider).streamResolutionScale;
        final pixelSize = Size(
          size.width * dpr * streamScale,
          size.height * dpr * streamScale,
        );
        final changedSize =
            lastSize == null ||
            (lastSize!.width - pixelSize.width).abs() > 8 ||
            (lastSize!.height - pixelSize.height).abs() > 8;
        if (changedSize) {
          lastSize = pixelSize;
          WidgetsBinding.instance.addPostFrameCallback((_) {
            send({
              'cmd': 'resize',
              'width': pixelSize.width.round(),
              'height': pixelSize.height.round(),
            });
          });
        }
        return Listener(
          onPointerDown: (event) {
            lastPosition = event.localPosition;
            dragButtons = event.buttons;
          },
          onPointerMove: (event) {
            final last = lastPosition;
            if (last == null) return;
            final delta = event.localPosition - last;
            lastPosition = event.localPosition;
            final pan =
                (dragButtons & kSecondaryMouseButton) != 0 ||
                (dragButtons & kMiddleMouseButton) != 0;
            send({
              'cmd': pan ? 'pan' : 'orbit',
              'dx': delta.dx,
              'dy': delta.dy,
            });
          },
          onPointerSignal: (event) {
            if (event is PointerScrollEvent) {
              send({'cmd': 'zoom', 'notches': -event.scrollDelta.dy / 120.0});
            }
          },
          child: MouseRegion(
            cursor: SystemMouseCursors.move,
            child: Stack(
              fit: StackFit.expand,
              children: [
                ColoredBox(
                  color: Colors.black,
                  child: trainingActive
                      ? const _TrainingOverlay()
                      : rawFrame != null
                      ? RawImage(image: rawFrame, fit: BoxFit.contain)
                      : frame == null
                      ? _WaitingOverlay(status: status, onRetry: connect)
                      : Image.memory(
                          frame!,
                          gaplessPlayback: true,
                          fit: BoxFit.contain,
                          errorBuilder: (context, error, stackTrace) {
                            return EmptyState(
                              icon: Icons.broken_image_outlined,
                              title: 'Frame decode failed',
                              subtitle:
                                  '${frame!.length} bytes · header ${frame!.take(8).toList()}',
                            );
                          },
                        ),
                ),
                Positioned(
                  left: 12,
                  top: 12,
                  child: _GlassPill(
                    child: Row(
                      mainAxisSize: MainAxisSize.min,
                      children: [
                        if (playbackLabel != null) ...[
                          const Icon(
                            Icons.smart_display_outlined,
                            size: 13,
                            color: Color(0xffffc857),
                          ),
                          const SizedBox(width: 7),
                          Text(
                            playbackLabel!,
                            style: const TextStyle(
                              fontSize: 12,
                              color: Color(0xffffc857),
                              fontWeight: FontWeight.w600,
                            ),
                          ),
                        ] else ...[
                          Container(
                            width: 7,
                            height: 7,
                            decoration: BoxDecoration(
                              shape: BoxShape.circle,
                              color: connected
                                  ? const Color(0xff5fe089)
                                  : const Color(0xffff6f64),
                            ),
                          ),
                          const SizedBox(width: 7),
                          Text(status, style: const TextStyle(fontSize: 12)),
                        ],
                      ],
                    ),
                  ),
                ),
                if (!trainingActive)
                  Positioned(
                    right: 10,
                    top: 10,
                    child: _GlassPill(
                      padding: const EdgeInsets.all(3),
                      child: Row(
                        mainAxisSize: MainAxisSize.min,
                        children: [
                          IconButton(
                            tooltip: paused ? 'Resume' : 'Pause',
                            visualDensity: VisualDensity.compact,
                            onPressed: () {
                              send({'cmd': 'pause'});
                              setState(() => paused = !paused);
                            },
                            icon: Icon(
                              paused
                                  ? Icons.play_arrow_rounded
                                  : Icons.pause_rounded,
                              size: 20,
                            ),
                          ),
                          IconButton(
                            tooltip: 'Step one frame',
                            visualDensity: VisualDensity.compact,
                            onPressed: () => send({'cmd': 'step'}),
                            icon: const Icon(Icons.skip_next_rounded, size: 20),
                          ),
                          IconButton(
                            tooltip: 'Reset simulation',
                            visualDensity: VisualDensity.compact,
                            onPressed: () => send({'cmd': 'reset'}),
                            icon: const Icon(
                              Icons.restart_alt_rounded,
                              size: 20,
                            ),
                          ),
                        ],
                      ),
                    ),
                  ),
                if (!trainingActive)
                  Positioned(
                    left: 12,
                    bottom: 10,
                    child: _GlassPill(
                      child: Text(
                        'Drag: orbit · Right/middle drag: pan · Scroll: zoom',
                        style: TextStyle(
                          fontSize: 11,
                          color: Colors.white.withValues(alpha: 0.75),
                        ),
                      ),
                    ),
                  ),
              ],
            ),
          ),
        );
      },
    );
  }
}

class _WaitingOverlay extends StatelessWidget {
  const _WaitingOverlay({required this.status, required this.onRetry});

  final String status;
  final VoidCallback onRetry;

  @override
  Widget build(BuildContext context) {
    final disconnected = status == 'disconnected';
    return EmptyState(
      icon: disconnected ? Icons.cloud_off_outlined : Icons.sensors,
      title: disconnected
          ? 'Stream disconnected'
          : 'Waiting for PyBullet frame stream…',
      subtitle: disconnected
          ? 'The backend simulation stream is unreachable. Retrying automatically every 2 seconds.'
          : 'Connecting to ws://127.0.0.1:8000/ws/simulation',
      action: disconnected
          ? OutlinedButton.icon(
              onPressed: onRetry,
              icon: const Icon(Icons.refresh, size: 16),
              label: const Text('Retry now'),
            )
          : null,
    );
  }
}

class _TrainingOverlay extends StatelessWidget {
  const _TrainingOverlay();

  @override
  Widget build(BuildContext context) {
    return Center(
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          const SizedBox(
            width: 30,
            height: 30,
            child: CircularProgressIndicator(
              strokeWidth: 2.5,
              color: Colors.white,
            ),
          ),
          const SizedBox(height: 16),
          const Text(
            'Training in progress',
            style: TextStyle(
              color: Colors.white,
              fontSize: 15,
              fontWeight: FontWeight.w700,
            ),
          ),
          const SizedBox(height: 6),
          ConstrainedBox(
            constraints: const BoxConstraints(maxWidth: 320),
            child: Text(
              'Live view is paused so the simulator runs at full speed. '
              'Watch the reward curve in the Training panel.',
              textAlign: TextAlign.center,
              style: TextStyle(
                color: Colors.white.withValues(alpha: 0.7),
                fontSize: 12.5,
                height: 1.4,
              ),
            ),
          ),
        ],
      ),
    );
  }
}

class _GlassPill extends StatelessWidget {
  const _GlassPill({
    required this.child,
    this.padding = const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
  });

  final Widget child;
  final EdgeInsets padding;

  @override
  Widget build(BuildContext context) {
    return DecoratedBox(
      decoration: BoxDecoration(
        color: Colors.black.withValues(alpha: 0.55),
        borderRadius: BorderRadius.circular(8),
        border: Border.all(color: Colors.white.withValues(alpha: 0.12)),
      ),
      child: Padding(padding: padding, child: child),
    );
  }
}
