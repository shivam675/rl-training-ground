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

class FinalChangesApi extends BackendApi {
  @override
  Future<Map<String, dynamic>> getJson(String path) async => {'ok': true};

  @override
  Future<Map<String, dynamic>> postJson(
    String path,
    Map<String, dynamic> body,
  ) async {
    if (path == '/env/config/undo') {
      return {
        'ok': true,
        'config': const {},
        'problems': const [],
        'warnings': const [],
        'revision': 2,
        'vector_sizes': const {
          'observation_vector_size': 0,
          'action_vector_size': 0,
        },
        'change_set': {
          'undoable': true,
          'summary': ['Disabled observation base_position.'],
        },
      };
    }
    return {'ok': true};
  }

  @override
  Stream<Map<String, dynamic>> streamPostJson(
    String path,
    Map<String, dynamic> body,
  ) async* {
    yield {'type': 'tool_call', 'name': 'patch_env_config', 'args': const {}};
    yield {
      'type': 'tool_result',
      'name': 'patch_env_config',
      'result': {
        'ok': true,
        'config': const {},
        'problems': const [],
        'warnings': const [],
        'revision': 1,
        'vector_sizes': const {
          'observation_vector_size': 3,
          'action_vector_size': 1,
        },
        'change_set': {
          'undoable': true,
          'summary': ['Enabled observation base_position.'],
        },
      },
    };
    yield {'type': 'done'};
  }
}

class RoutedApi extends BackendApi {
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
    yield {'type': 'route', 'provider': 'openai', 'label': 'NVIDIA'};
    yield {'type': 'chunk', 'text': 'I will draft the objective first.'};
    yield {'type': 'done'};
  }
}

class PendingConfirmationApi extends BackendApi {
  bool approved = false;
  Map<String, dynamic>? approvalBody;
  int chatRequests = 0;

  @override
  Future<Map<String, dynamic>> getJson(String path) async => {'ok': true};

  @override
  Future<Map<String, dynamic>> postJson(
    String path,
    Map<String, dynamic> body,
  ) async {
    if (path == '/agents/execute_tool') {
      approved = true;
      approvalBody = body;
      return {
        'tool': body['name'],
        'result': {
          'ok': true,
          'config': const {},
          'problems': const [],
          'warnings': const [],
          'revision': 1,
          'vector_sizes': const {
            'observation_vector_size': 455,
            'action_vector_size': 12,
          },
          'change_set': {
            'undoable': true,
            'summary': ['Applied walking goal.'],
          },
        },
      };
    }
    return {'ok': true};
  }

  @override
  Stream<Map<String, dynamic>> streamPostJson(
    String path,
    Map<String, dynamic> body,
  ) async* {
    chatRequests += 1;
    yield {'type': 'route', 'provider': 'openai', 'label': 'NVIDIA'};
    yield {
      'type': 'tool_call',
      'name': 'apply_behavior_goal',
      'args': {'goal': 'walk straight'},
    };
    yield {
      'type': 'tool_result',
      'name': 'apply_behavior_goal',
      'result': {
        'requires_confirmation': true,
        'tool': 'apply_behavior_goal',
        'args': {'goal': 'walk straight'},
      },
    };
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

  testWidgets('chat shows final changes and undo for config patches', (
    tester,
  ) async {
    await tester.pumpWidget(
      ProviderScope(
        overrides: [
          appStateProvider.overrideWith((ref) => AppState(FinalChangesApi())),
        ],
        child: const MaterialApp(home: Scaffold(body: AssistantChat())),
      ),
    );

    await tester.enterText(find.byType(TextField), 'Configure the robot');
    await tester.tap(find.byTooltip('Send (Enter)'));
    await tester.pump(const Duration(milliseconds: 50));

    expect(find.text('Final changes'), findsOneWidget);
    expect(find.text('Enabled observation base_position.'), findsOneWidget);
    expect(find.text('Configuration valid'), findsOneWidget);
    expect(find.text('Undo'), findsOneWidget);

    await tester.tap(find.text('Undo'));
    await tester.pump(const Duration(milliseconds: 50));
    expect(find.text('Disabled observation base_position.'), findsOneWidget);
  });

  testWidgets('chat shows the provider route tag', (tester) async {
    await tester.pumpWidget(
      ProviderScope(
        overrides: [
          appStateProvider.overrideWith((ref) => AppState(RoutedApi())),
        ],
        child: const MaterialApp(home: Scaffold(body: AssistantChat())),
      ),
    );

    await tester.enterText(find.byType(TextField), 'make this robot walk');
    await tester.tap(find.byTooltip('Send (Enter)'));
    await tester.pump(const Duration(milliseconds: 80));

    expect(find.text('NVIDIA'), findsOneWidget);
  });

  testWidgets('chat exposes an approve button for pending tool confirmations', (
    tester,
  ) async {
    final api = PendingConfirmationApi();
    await tester.pumpWidget(
      ProviderScope(
        overrides: [appStateProvider.overrideWith((ref) => AppState(api))],
        child: const MaterialApp(home: Scaffold(body: AssistantChat())),
      ),
    );

    await tester.enterText(find.byType(TextField), 'make this robot walk');
    await tester.tap(find.byTooltip('Send (Enter)'));
    await tester.pump(const Duration(milliseconds: 80));

    expect(find.text('Approve & Apply'), findsOneWidget);
    await tester.tap(find.text('Approve & Apply'));
    await tester.pump(const Duration(milliseconds: 80));

    expect(api.approved, isTrue);
    expect(api.approvalBody?['name'], 'apply_behavior_goal');
    expect(api.approvalBody?['args'], {'goal': 'walk straight'});
    expect(find.text('Applied walking goal.'), findsOneWidget);
  });

  testWidgets(
    'typing yes approves the pending tool instead of chatting again',
    (tester) async {
      final api = PendingConfirmationApi();
      await tester.pumpWidget(
        ProviderScope(
          overrides: [appStateProvider.overrideWith((ref) => AppState(api))],
          child: const MaterialApp(home: Scaffold(body: AssistantChat())),
        ),
      );

      await tester.enterText(find.byType(TextField), 'make this robot walk');
      await tester.tap(find.byTooltip('Send (Enter)'));
      await tester.pump(const Duration(milliseconds: 80));

      await tester.enterText(find.byType(TextField), 'yes');
      await tester.tap(find.byTooltip('Send (Enter)'));
      await tester.pump(const Duration(milliseconds: 80));

      expect(api.chatRequests, 1);
      expect(api.approved, isTrue);
      expect(api.approvalBody?['name'], 'apply_behavior_goal');
      expect(find.text('Applied walking goal.'), findsOneWidget);
    },
  );

  testWidgets('loaded robot prompt asks for a generic natural language goal', (
    tester,
  ) async {
    final state = AppState(RoutedApi());
    await tester.pumpWidget(
      ProviderScope(
        overrides: [appStateProvider.overrideWith((ref) => state)],
        child: const MaterialApp(home: Scaffold(body: AssistantChat())),
      ),
    );

    state.lastLoadedRobotPath = r'D:\robots\new_robot.urdf';
    state.robotLoadRevision += 1;
    state.notifyListeners();
    await tester.pump(const Duration(milliseconds: 20));

    expect(find.textContaining('natural language'), findsOneWidget);
    expect(find.textContaining('sit Japanese-style'), findsNothing);
  });
}
