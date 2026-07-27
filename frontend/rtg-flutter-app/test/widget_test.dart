import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import 'package:rtg_flutter_app/main.dart';
import 'package:rtg_flutter_app/src/api_client.dart';
import 'package:rtg_flutter_app/src/app_state.dart';
import 'package:rtg_flutter_app/src/panels/reward_panel.dart';

class EmptyApi extends BackendApi {
  @override
  Future<Map<String, dynamic>> getJson(String path) async => {'ok': true};

  @override
  Future<Map<String, dynamic>> postJson(
    String path,
    Map<String, dynamic> body,
  ) async => {'ok': true};
}

class TelemetryApi extends EmptyApi {
  final requests = <String>[];
  int poll = 0;

  @override
  Future<Map<String, dynamic>> getJson(String path) async {
    requests.add(path);
    if (path == '/training/status') return {'active': true};
    if (path.startsWith('/training/telemetry')) {
      poll += 1;
      final start = poll == 1 ? 0 : 2001;
      final count = poll == 1 ? 2001 : 1;
      return {
        'total': start + count,
        'points': [
          for (var i = start; i < start + count; i++) {'timestep': i},
        ],
      };
    }
    return {'ok': true};
  }
}

void main() {
  testWidgets('telemetry keeps a bounded window without resetting its cursor', (
    tester,
  ) async {
    final api = TelemetryApi();
    final state = AppState(api)..trainingStatus = {'active': true};

    await tester.pump(const Duration(seconds: 2));
    await tester.pump();
    expect(state.telemetry.length, 2000);

    await tester.pump(const Duration(seconds: 2));
    await tester.pump();
    expect(api.requests, contains('/training/telemetry?since=2001'));
    expect(state.telemetry.length, 2000);
    expect(state.telemetry.last['timestep'], 2001);
    state.dispose();
  });

  testWidgets('EasyRTG shell renders', (WidgetTester tester) async {
    // EasyRTG is a desktop app; the default 800x600 test window is below its
    // usable width once the co-pilot dock is docked. Use a realistic size.
    tester.view.physicalSize = const Size(1600, 1000);
    tester.view.devicePixelRatio = 1.0;
    addTearDown(tester.view.resetPhysicalSize);
    addTearDown(tester.view.resetDevicePixelRatio);

    await tester.pumpWidget(const ProviderScope(child: EasyRtgApp()));

    expect(find.text('EasyRTG'), findsWidgets);
    // Home now leads with the assistant as the main canvas (no 3D viewport),
    // beside the Setup & Status + Training sidebar.
    expect(find.text('Assistant'), findsWidgets);
    expect(find.text('Setup & Status'), findsOneWidget);
  });

  testWidgets('RewardPanel renders structured reward test output', (
    tester,
  ) async {
    final state = AppState(EmptyApi())
      ..observations = {'reward_components': const []}
      ..envConfig = {'rewards': const []}
      ..message = 'Connected to backend.'
      ..lastRewardResult = {
        'reward': 0.5,
        'formula': '0.5*stay_alive',
        'warnings': const ['Check reward scale.'],
        'terms': const [
          {'key': 'stay_alive', 'raw': 1.0, 'weight': 0.5, 'value': 0.5},
        ],
      };

    await tester.pumpWidget(
      ProviderScope(
        overrides: [appStateProvider.overrideWith((ref) => state)],
        child: const MaterialApp(home: Scaffold(body: RewardPanel())),
      ),
    );

    expect(find.text('Reward 0.5000'), findsOneWidget);
    expect(find.text('0.5*stay_alive'), findsOneWidget);
    expect(find.text('stay_alive'), findsOneWidget);
    expect(find.textContaining('Check reward scale.'), findsOneWidget);
  });
}
