import 'dart:ui' show AppExitResponse;

import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:google_fonts/google_fonts.dart';

import 'src/backend_launcher.dart';
import 'src/home.dart';
import 'src/theme/app_theme.dart';
import 'src/theme/theme_controller.dart';

final backendLauncher = BackendLauncher();

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  // Fonts are bundled in google_fonts/ — never fetch from the network.
  GoogleFonts.config.allowRuntimeFetching = false;
  // Last-resort guard: a stray async exception (e.g. a refused socket) must
  // never take down the whole app. Returning true marks it handled.
  PlatformDispatcher.instance.onError = (error, stack) {
    debugPrint('Unhandled async error: $error');
    return true;
  };
  // Spawn the backend if it isn't already running; UI shows reconnecting
  // states in the meantime, so don't block startup on it.
  backendLauncher.ensureRunning();
  runApp(const ProviderScope(child: EasyRtgApp()));
}

class EasyRtgApp extends ConsumerStatefulWidget {
  const EasyRtgApp({super.key});

  @override
  ConsumerState<EasyRtgApp> createState() => _EasyRtgAppState();
}

class _EasyRtgAppState extends ConsumerState<EasyRtgApp> {
  late final AppLifecycleListener _lifecycle;

  @override
  void initState() {
    super.initState();
    _lifecycle = AppLifecycleListener(
      onExitRequested: () async {
        // Stop the backend only if this app started it.
        backendLauncher.shutdown();
        return AppExitResponse.exit;
      },
    );
  }

  @override
  void dispose() {
    _lifecycle.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final theme = ref.watch(themeControllerProvider);
    return MaterialApp(
      title: 'EasyRTG',
      debugShowCheckedModeBanner: false,
      themeMode: theme.mode,
      theme: buildAppTheme(
        brightness: Brightness.light,
        seed: theme.accent.seed,
      ),
      darkTheme: buildAppTheme(
        brightness: Brightness.dark,
        seed: theme.accent.seed,
      ),
      home: const EasyRtgHome(),
    );
  }
}
