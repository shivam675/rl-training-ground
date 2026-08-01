import 'dart:io';

import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:google_fonts/google_fonts.dart';

import 'package:rtg_flutter_app/main.dart';
import 'package:rtg_flutter_app/src/api_client.dart';
import 'package:rtg_flutter_app/src/app_state.dart';
import 'package:rtg_flutter_app/src/panels/home_panel.dart';
import 'package:rtg_flutter_app/src/panels/robot_panel.dart';
import 'package:rtg_flutter_app/src/panels/reward_panel.dart';
import 'package:rtg_flutter_app/src/panels/training_panel.dart';
import 'package:rtg_flutter_app/src/theme/app_theme.dart';
import 'package:rtg_flutter_app/src/widgets/common.dart';

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
  // buildAppTheme() reaches GoogleFonts, which reaches the asset bundle, which
  // needs a binding. The testWidgets below happen to initialise one at
  // declaration time — don't rely on that staying true.
  TestWidgetsFlutterBinding.ensureInitialized();

  // main.dart sets GoogleFonts.config.allowRuntimeFetching = false, so a font
  // the bundle doesn't carry is NOT fetched and NOT an error — it degrades to
  // the platform default behind a debugPrint nobody reads. The lookup matches
  // on an exact '<Family>-<Weight>' filename suffix (google_fonts_variant.dart
  // maps w600 -> 'SemiBold'), so a single typo silently un-brands the app.
  // These two tests are the only thing standing between that and a release.
  test('every weight the app renders has a bundled asset', () {
    const required = [
      'Geist-Regular', // w400, body
      'Geist-Medium', // w500
      'Geist-SemiBold', // w600
      'Geist-Bold', // w700, the heaviest weight any widget asks for
      'GeistMono-Regular',
      'GeistMono-Bold',
    ];
    for (final name in required) {
      expect(
        File('google_fonts/$name.ttf').existsSync(),
        isTrue,
        reason:
            'google_fonts/$name.ttf is missing — google_fonts will fall back '
            'to the platform default without failing.',
      );
    }
  });

  test('the theme and monoStyle resolve to Geist', () {
    final theme = buildAppTheme(brightness: Brightness.light);
    expect(theme.textTheme.bodyMedium?.fontFamily, contains('Geist'));
    expect(GoogleFonts.geistMono().fontFamily, contains('GeistMono'));
    // Ink-only accent (D4): primary is a near-black neutral, never a hue. If
    // someone reintroduces a tinted accent, the semantic four stop being the
    // only saturated colour in the app and audit verdicts lose their punch.
    final primary = theme.colorScheme.primary;
    expect(
      primary.computeLuminance(),
      lessThan(0.05),
      reason: 'not near-black',
    );
    // Ink is a slightly cool near-black and sits at 0.25; any real accent hue
    // is far higher (the rejected teal is 0.84, cobalt 0.81). 0.5 splits them
    // with margin rather than sitting on ink's exact value.
    expect(
      HSVColor.fromColor(primary).saturation,
      lessThan(0.5),
      reason: 'primary picked up a hue',
    );
  });

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

  testWidgets('Home survives a robot loaded with a long validator problem', (
    tester,
  ) async {
    // The state the app is actually in most of the time, and the one the
    // shell-renders test never reaches: a robot is loaded, so the sidebar shows
    // real content, and the config has a problem whose text is a full compiler
    // rejection. That combination overflowed the Training panel by 250px in the
    // running app while every test here stayed green — the panel has a fixed
    // height, so anything that can grow has to scroll.
    const longProblem =
        'Reward expression rejected — rotation_reward: Subscript is not allowed '
        'in an expression. Use arithmetic, comparisons, and the built-in '
        'functions only. Per-element state is reduced with mean(), sum(), max() '
        'or norm() — never indexed — and a single component has its own name '
        '(yaw_rate not base_angular_velocity[2], base_vz not '
        'base_linear_velocity[2]).';
    final state = AppState(EmptyApi())
      ..robotInfo = {'name': 'go1', 'joint_count': 45, 'path': 'go1.urdf'}
      ..envConfig = {
        'observations': const [
          {'key': 'base_position', 'enabled': true},
        ],
        'actions': const [
          {'joint_name': 'FR_hip_joint', 'enabled': true},
        ],
        'rewards': const [
          {'key': 'rotation_reward', 'enabled': true},
        ],
      }
      ..configProblems = [longProblem];

    // Not the 1600x1000 of the shell test: 1280x720 is what main.cpp opens.
    tester.view.physicalSize = const Size(1280, 720);
    tester.view.devicePixelRatio = 1.0;
    addTearDown(tester.view.resetPhysicalSize);
    addTearDown(tester.view.resetDevicePixelRatio);

    await tester.pumpWidget(
      ProviderScope(
        overrides: [appStateProvider.overrideWith((ref) => state)],
        child: MaterialApp(
          theme: buildAppTheme(brightness: Brightness.light),
          home: const Scaffold(body: HomePanel(urdfPath: 'go1.urdf')),
        ),
      ),
    );
    await tester.pump(const Duration(milliseconds: 400));

    // A RenderFlex overflow is reported as an exception, not a failed finder.
    expect(tester.takeException(), isNull, reason: 'Home overflowed');
    // The blocker line must stay short enough to render; the full text belongs
    // in the scrollable problems list, not spliced into "Next: …".
    expect(find.textContaining('fix 1 configuration problem'), findsWidgets);
    // No state.dispose() — the ChangeNotifierProvider owns it and disposing
    // here double-disposes.
  });

  testWidgets('Robot Setup keeps reference detail collapsed until asked', (
    tester,
  ) async {
    // The page used to render all 45 joint rows, the world physics knobs and
    // the URDF warning list the moment a robot loaded. Everything that is
    // reference rather than action now sits behind a disclosure.
    final state = AppState(EmptyApi())
      ..robotInfo = {
        'name': 'go1',
        'joint_count': 45,
        'path': 'go1.urdf',
        'warnings': const ['link FR_foot has near-zero mass'],
        'joints': [
          for (var i = 0; i < 45; i++)
            {
              'index': i,
              'name': 'joint_$i',
              'type': i.isEven ? 'revolute' : 'fixed',
              'lower_limit': -1.0,
              'upper_limit': 1.0,
            },
        ],
      };

    tester.view.physicalSize = const Size(1280, 900);
    tester.view.devicePixelRatio = 1.0;
    addTearDown(tester.view.resetPhysicalSize);
    addTearDown(tester.view.resetDevicePixelRatio);

    await tester.pumpWidget(
      ProviderScope(
        overrides: [appStateProvider.overrideWith((ref) => state)],
        child: MaterialApp(
          theme: buildAppTheme(brightness: Brightness.light),
          home: const Scaffold(body: RobotPanel()),
        ),
      ),
    );

    // Identity is up front...
    expect(find.text('go1'), findsWidgets);
    expect(find.text('45'), findsWidgets);
    // ...but none of the reference material is.
    expect(find.text('joint_0'), findsNothing);
    expect(find.text('Gravity Z (m/s²)'), findsNothing);
    expect(find.textContaining('near-zero mass'), findsNothing);

    await tester.tap(find.text('Joints'));
    await tester.pumpAndSettle();
    expect(find.text('joint_0'), findsOneWidget);
    expect(tester.takeException(), isNull);
  });

  // The four layout regressions the 2026-08-01 pass fixed. Each was invisible
  // to `flutter analyze` and to every assertion above — they render fine, just
  // in the wrong place — so they are pinned by geometry, not by finding text.
  testWidgets('a section header parks its action against the right margin', (
    tester,
  ) async {
    // SectionHeader paired `Flexible(flex: 1)` with `Spacer(flex: 1)`. Both
    // claimed one flex unit, so the Spacer only ever got HALF the free space
    // and the loose Flexible's unclaimed half was dropped past the trailing
    // widget — parking Test reward / the runs refresh / the vector-size chip
    // around 65% of the row instead of flush right.
    final state = AppState(EmptyApi())..envConfig = {'rewards': const []};

    tester.view.physicalSize = const Size(1280, 900);
    tester.view.devicePixelRatio = 1.0;
    addTearDown(tester.view.resetPhysicalSize);
    addTearDown(tester.view.resetDevicePixelRatio);

    await tester.pumpWidget(
      ProviderScope(
        overrides: [appStateProvider.overrideWith((ref) => state)],
        child: MaterialApp(
          theme: buildAppTheme(brightness: Brightness.light),
          home: const Scaffold(body: RewardPanel()),
        ),
      ),
    );

    final header = tester.getRect(find.byType(SectionHeader).first);
    final action = tester.getRect(find.byType(FilledButton).first);
    expect(
      header.right - action.right,
      lessThan(1.0),
      reason:
          'the header action should end at the header\'s right edge; it ended '
          '${(header.right - action.right).toStringAsFixed(0)}px short, which '
          'is the Flexible-plus-Spacer bug back again',
    );
  });

  testWidgets('the three robot identity tiles stay on one line', (
    tester,
  ) async {
    // Three StatChips want 327px in a Wrap and the identity column offers ~308,
    // so MOVABLE dropped to a second row. They are equal thirds of a Row now.
    final state = AppState(EmptyApi())
      ..robotInfo = {
        'name': 'go1',
        'joint_count': 45,
        'path': 'go1.urdf',
        'joints': [
          for (var i = 0; i < 45; i++)
            {
              'index': i,
              'name': 'joint_$i',
              'type': i < 12 ? 'revolute' : 'fixed',
              'lower_limit': -1.0,
              'upper_limit': 1.0,
            },
        ],
      };

    tester.view.physicalSize = const Size(1280, 900);
    tester.view.devicePixelRatio = 1.0;
    addTearDown(tester.view.resetPhysicalSize);
    addTearDown(tester.view.resetDevicePixelRatio);

    await tester.pumpWidget(
      ProviderScope(
        overrides: [appStateProvider.overrideWith((ref) => state)],
        child: MaterialApp(
          theme: buildAppTheme(brightness: Brightness.light),
          home: const Scaffold(body: RobotPanel()),
        ),
      ),
    );

    final tiles = tester
        .widgetList<StatChip>(find.byType(StatChip))
        .toList(growable: false);
    expect(tiles.length, 3);
    final tops = <double>{
      for (var i = 0; i < 3; i++)
        tester.getTopLeft(find.byType(StatChip).at(i)).dy,
    };
    expect(tops.length, 1, reason: 'a tile wrapped onto a second line');
    // Movable counts the 12 revolute joints, not all 45.
    expect(find.text('12'), findsOneWidget);
  });

  testWidgets('World puts its field left, its controls right, with a gap', (
    tester,
  ) async {
    // Two fixes at once: the disclosure had bottom-only childrenPadding, so its
    // content butted against the title that opened it; and the field sat
    // *between* two toggles in one Wrap, so nothing lined up.
    final state = AppState(EmptyApi())
      ..robotInfo = {'name': 'go1', 'joint_count': 0, 'joints': const []};

    tester.view.physicalSize = const Size(1280, 900);
    tester.view.devicePixelRatio = 1.0;
    addTearDown(tester.view.resetPhysicalSize);
    addTearDown(tester.view.resetDevicePixelRatio);

    await tester.pumpWidget(
      ProviderScope(
        overrides: [appStateProvider.overrideWith((ref) => state)],
        child: MaterialApp(
          theme: buildAppTheme(brightness: Brightness.light),
          home: const Scaffold(body: RobotPanel()),
        ),
      ),
    );

    await tester.tap(find.text('World'));
    await tester.pumpAndSettle();
    expect(tester.takeException(), isNull);

    final title = tester.getRect(find.text('World'));
    final field = tester.getRect(find.byType(TextField).first);
    final reset = tester.getRect(find.text('Reset world'));

    expect(
      field.top - title.bottom,
      greaterThan(8.0),
      reason: 'the expanded form needs air under the title that opened it',
    );
    expect(
      field.right,
      lessThan(reset.left),
      reason: 'the input belongs left of every button and toggle',
    );
    // Every floating label must render inside the filled box. With an outline
    // border it straddled the top edge and got clipped in half.
    final label = tester.getRect(find.text('Gravity Z (m/s²)'));
    expect(
      label.top,
      greaterThanOrEqualTo(field.top),
      reason: 'the label is hanging outside the field and will be clipped',
    );
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

  testWidgets('RewardPanel edits the expression and shows the audit verdict', (
    tester,
  ) async {
    final api = RecordingApi();
    final state = AppState(api)
      ..observations = {'reward_components': const []}
      ..envConfig = {
        'rewards': const [
          {
            'key': 'flight_phase',
            'enabled': true,
            'weight': 0.5,
            'params': {'expr': '1 - max(contact)'},
          },
        ],
      }
      // Audit output from the last patch, keyed by the term it is about.
      ..configWarnings = ['flight_phase: penalty dominates the reward budget.'];

    await tester.pumpWidget(
      ProviderScope(
        overrides: [appStateProvider.overrideWith((ref) => state)],
        child: const MaterialApp(home: Scaffold(body: RewardPanel())),
      ),
    );

    // The expression IS the term, so it is visible without expanding, and the
    // audit verdict sits next to the term it concerns rather than in a list.
    expect(find.text('1 - max(contact)'), findsOneWidget);
    expect(
      find.text('flight_phase: penalty dominates the reward budget.'),
      findsOneWidget,
    );

    await tester.tap(find.byTooltip('Edit expression'));
    await tester.pumpAndSettle();
    await tester.enterText(
      find.widgetWithText(TextField, '1 - max(contact)'),
      'mean(air_time)',
    );
    await tester.tap(find.text('Save expression'));
    await tester.pumpAndSettle();

    expect(api.patches, hasLength(1));
    final rewards = (api.patches.single['patch'] as Map)['rewards'] as List;
    expect((rewards.single as Map)['key'], 'flight_phase');
    expect(
      ((rewards.single as Map)['params'] as Map)['expr'],
      'mean(air_time)',
    );
  });

  testWidgets('Training controls all sit on one baseline', (tester) async {
    // "Parallel envs" carried a helperText, which reserved a line under that one
    // field and pushed its box up relative to every sibling in the same Wrap.
    final state = AppState(EmptyApi());

    tester.view.physicalSize = const Size(1600, 1000);
    tester.view.devicePixelRatio = 1.0;
    addTearDown(tester.view.resetPhysicalSize);
    addTearDown(tester.view.resetDevicePixelRatio);

    await tester.pumpWidget(
      ProviderScope(
        overrides: [appStateProvider.overrideWith((ref) => state)],
        child: MaterialApp(
          theme: buildAppTheme(brightness: Brightness.light),
          home: const Scaffold(body: TrainingPanel()),
        ),
      ),
    );
    await tester.pump(const Duration(milliseconds: 50));

    final envs = tester.getRect(
      find.ancestor(
        of: find.text('Parallel envs'),
        matching: find.byType(TextField),
      ),
    );
    final arch = tester.getRect(
      find.ancestor(
        of: find.text('MLP hidden layers'),
        matching: find.byType(TextField),
      ),
    );
    expect(
      envs.top,
      moreOrLessEquals(arch.top, epsilon: 1.0),
      reason: 'the envs field must share a top edge with its neighbours',
    );
    expect(envs.height, moreOrLessEquals(arch.height, epsilon: 1.0));
  });

  testWidgets('a running trainer reports elapsed, ETA and episode length', (
    tester,
  ) async {
    final state = AppState(EmptyApi());

    tester.view.physicalSize = const Size(1600, 1400);
    tester.view.devicePixelRatio = 1.0;
    addTearDown(tester.view.resetPhysicalSize);
    addTearDown(tester.view.resetDevicePixelRatio);

    await tester.pumpWidget(
      ProviderScope(
        overrides: [appStateProvider.overrideWith((ref) => state)],
        child: MaterialApp(
          theme: buildAppTheme(brightness: Brightness.light),
          home: const Scaffold(body: TrainingPanel()),
        ),
      ),
    );

    // 60s in, a quarter done -> ~180s left at the run's own average rate.
    state.trainingStatus = {
      'active': true,
      'message': 'training',
      'timestep': 250000,
      'total_timesteps': 1000000,
      'episode_length': 411,
      'started_at':
          DateTime.now().subtract(const Duration(seconds: 60)).millisecondsSinceEpoch /
          1000,
    };
    state.notifyListeners();
    await tester.pump(const Duration(milliseconds: 50));

    expect(find.text('1m 0s'), findsOneWidget); // elapsed
    expect(find.text('3m 0s'), findsOneWidget); // finishes in
    expect(find.text('411'), findsOneWidget); // episode length

    // Idle: elapsed would otherwise keep counting up from a start with no end.
    state.trainingStatus = {'active': false, 'message': 'complete'};
    state.notifyListeners();
    await tester.pump(const Duration(milliseconds: 50));
    expect(find.text('Elapsed'), findsNothing);
    expect(find.text('Finishes in'), findsNothing);
  });
}

/// Captures config patches so an edit can be asserted on what it SENDS, not on
/// what the widget happens to redraw.
class RecordingApi extends EmptyApi {
  final patches = <Map<String, dynamic>>[];

  @override
  Future<Map<String, dynamic>> postJson(
    String path,
    Map<String, dynamic> body,
  ) async {
    if (path == '/env/config/patch') patches.add(body);
    return {'ok': true};
  }

}
