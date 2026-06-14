import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:flutter_test/flutter_test.dart';

import 'package:rtg_flutter_app/src/api_client.dart';
import 'package:rtg_flutter_app/src/app_state.dart';
import 'package:rtg_flutter_app/src/agent/chat_view.dart';

/// Emits a fast burst of chunks the way Ollama does, plus a tool round.
class BurstApi extends BackendApi {
  @override
  Future<Map<String, dynamic>> getJson(String path) async => {'ok': true};

  @override
  Future<Map<String, dynamic>> postJson(
    String path,
    Map<String, dynamic> body,
  ) async => {'ok': true};

  @override
  Stream<Map<String, dynamic>> streamPostJson(
    String path,
    Map<String, dynamic> body,
  ) async* {
    yield {
      'type': 'tool_call',
      'name': 'get_robot_info',
      'args': {'detail': true},
    };
    yield {
      'type': 'tool_result',
      'name': 'get_robot_info',
      'result': {'name': 'r2d2'},
    };
    for (var i = 0; i < 400; i++) {
      yield {'type': 'chunk', 'text': 'token$i '};
      if (i % 50 == 0) {
        await Future<void>.delayed(const Duration(milliseconds: 10));
      }
    }
    yield {'type': 'done'};
  }
}

void main() {
  testWidgets('chat survives a rapid streaming burst with tool events', (
    tester,
  ) async {
    await tester.pumpWidget(
      ProviderScope(
        overrides: [
          appStateProvider.overrideWith((ref) => AppState(BurstApi())),
        ],
        child: const MaterialApp(home: Scaffold(body: AssistantChat())),
      ),
    );

    await tester.enterText(find.byType(TextField), 'What can you do');
    await tester.tap(find.byTooltip('Send (Enter)'));

    // Tool chip appears at the start of the stream, while still in view.
    await tester.pump(const Duration(milliseconds: 20));
    await tester.pump(const Duration(milliseconds: 20));
    expect(find.textContaining('get_robot_info'), findsOneWidget);

    // Advance through the stream in small steps, as real frames would.
    // (No pumpAndSettle: the focused TextField cursor blinks forever.)
    for (var i = 0; i < 40; i++) {
      await tester.pump(const Duration(milliseconds: 40));
    }

    expect(tester.takeException(), isNull);
    // The finished reply renders as markdown (rich text), fully streamed,
    // and the list auto-followed to the bottom so the tail is built.
    expect(find.textContaining('token399', findRichText: true), findsOneWidget);
  });
}
