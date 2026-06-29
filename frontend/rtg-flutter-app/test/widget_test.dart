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

void main() {
  testWidgets('EasyRTG shell renders', (WidgetTester tester) async {
    // EasyRTG is a desktop app; the default 800x600 test window is below its
    // usable width once the co-pilot dock is docked. Use a realistic size.
    tester.view.physicalSize = const Size(1600, 1000);
    tester.view.devicePixelRatio = 1.0;
    addTearDown(tester.view.resetPhysicalSize);
    addTearDown(tester.view.resetDevicePixelRatio);

    await tester.pumpWidget(const ProviderScope(child: EasyRtgApp()));

    expect(find.text('EasyRTG'), findsWidgets);
    expect(find.text('3D Simulation'), findsOneWidget);
    expect(find.text('Setup & Status'), findsOneWidget);
  });

  testWidgets('RewardPanel renders structured reward test output', (
    tester,
  ) async {
    final state = AppState(EmptyApi())
      ..observations = {
        'reward_components': const [],
      }
      ..envConfig = {
        'rewards': const [],
      }
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

