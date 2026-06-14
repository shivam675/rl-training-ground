import 'dart:async';
import 'dart:convert';
import 'dart:io';

import 'package:flutter/foundation.dart';
import 'package:flutter_riverpod/legacy.dart';

import 'api_client.dart';

final appStateProvider = ChangeNotifierProvider<AppState>((ref) {
  return AppState(BackendApi())..refreshAll();
});

class AppState extends ChangeNotifier {
  AppState(this.api) {
    _telemetryTimer = Timer.periodic(
      const Duration(seconds: 2),
      (_) => _pollTelemetry(),
    );
    // The active LLM provider (Ollama or an OpenAI-compatible endpoint) can be
    // reachable or not independently of this app's backend; poll it on a slow
    // cadence so the chat connection dot reflects the real provider, not just
    // "backend process up".
    _healthTimer = Timer.periodic(
      const Duration(seconds: 8),
      (_) => refreshAgentHealth(),
    );
    refreshAgentHealth();
  }

  final BackendApi api;
  Timer? _telemetryTimer;
  Timer? _healthTimer;
  final telemetry = <Map<String, dynamic>>[];
  bool _pollingTelemetry = false;
  bool _pollingHealth = false;

  @override
  void dispose() {
    _telemetryTimer?.cancel();
    _healthTimer?.cancel();
    super.dispose();
  }

  Map<String, dynamic>? agentHealth;

  /// True only when this backend AND the active LLM provider both answer.
  bool get agentConnected =>
      health?['ok'] == true && agentHealth?['reachable'] == true;

  /// Human-readable agent connection state for the chat header tooltip.
  String get agentConnectionDetail {
    if (health?['ok'] != true) return 'Backend offline';
    if (agentHealth == null) return 'Checking provider…';
    final provider = agentHealth?['provider']?.toString() ?? 'provider';
    if (agentHealth?['reachable'] != true) {
      final detail = agentHealth?['detail']?.toString();
      return 'Provider unreachable${detail != null && detail.isNotEmpty ? ' — $detail' : ''}';
    }
    final model = agentHealth?['model']?.toString() ?? 'model';
    if (agentHealth?['model_available'] == false) {
      return 'Connected ($provider), but model "$model" not found';
    }
    return 'Connected · $model';
  }

  Future<void> refreshAgentHealth() async {
    if (_pollingHealth) return;
    _pollingHealth = true;
    try {
      agentHealth = await api.getJson('/agent/health');
      notifyListeners();
    } catch (_) {
      agentHealth = {'ok': false, 'reachable': false};
      notifyListeners();
    } finally {
      _pollingHealth = false;
    }
  }

  // --- Projects (.rtg) ------------------------------------------------------
  // A project bundles the loaded robot + observation/action/reward config.
  String? currentProjectPath;

  String? get currentProjectName {
    final path = currentProjectPath;
    if (path == null) return null;
    final name = path.split(RegExp(r'[\\/]')).last;
    return name.isEmpty ? null : name;
  }

  /// Project name without the `.rtg` extension — what we send to the backend
  /// to label runs and stamp into exported project files.
  String? get _projectLabel {
    final name = currentProjectName;
    if (name == null) return null;
    return name.endsWith('.rtg') ? name.substring(0, name.length - 4) : name;
  }

  /// Stable id of the currently loaded project, surfaced by the backend in the
  /// env config. Runs are tagged with this so the Evaluation tab can show only
  /// the open project's models.
  String? get currentProjectId => envConfig?['project_id']?.toString();

  /// When false (default) the Evaluation tab lists only the open project's
  /// runs; when true it shows every run with a project label.
  bool runsShowAll = false;

  void setRunsShowAll(bool value) {
    if (runsShowAll == value) return;
    runsShowAll = value;
    loadRuns();
  }

  /// Fresh workspace: unload the robot and clear the saved config server-side.
  Future<void> newProject() async {
    await guard(() async {
      await api.postJson('/project/new', {});
      currentProjectPath = null;
      telemetry.clear();
      await refreshAll();
      message = 'New project — load a URDF to begin.';
    });
  }

  /// The current project as a portable payload ready to write to a .rtg file.
  Future<Map<String, dynamic>> exportProject() {
    final label = _projectLabel;
    final query = label == null
        ? ''
        : '?name=${Uri.encodeQueryComponent(label)}';
    return api.getJson('/project/export$query');
  }

  /// Apply a project payload read from a .rtg file (loads its URDF + config).
  Future<List<String>> openProjectConfig(
    Map<String, dynamic> config,
    String path,
  ) async {
    final problems = <String>[];
    final label = path
        .split(RegExp(r'[\\/]'))
        .last
        .replaceAll(RegExp(r'\.rtg$'), '');
    await guard(() async {
      final res = await api.postJson('/project/open', {
        'config': config,
        'name': label,
      });
      currentProjectPath = path;
      problems.addAll(
        (res['problems'] as List? ?? []).map((p) => p.toString()),
      );
      await refreshAll();
      robotLoadRevision += 1;
      lastLoadedRobotPath =
          robotInfo?['source_path']?.toString() ??
          robotInfo?['path']?.toString();
      message = problems.isEmpty
          ? 'Opened ${currentProjectName ?? 'project'}.'
          : 'Opened ${currentProjectName ?? 'project'} with problems: ${problems.join('; ')}';
    });
    return problems;
  }

  /// Write the current project as pretty .rtg JSON to [path] and remember it.
  Future<void> saveProjectToPath(String path) async {
    await guard(() async {
      // Set the path first so exportProject() can derive the project name and
      // the backend stamps it (plus a stable id) into the saved file.
      currentProjectPath = path;
      final payload = await exportProject();
      const encoder = JsonEncoder.withIndent('  ');
      await File(path).writeAsString(encoder.convert(payload));
      await loadEnvConfig(notify: false); // pick up the freshly-stamped id
      message = 'Saved ${currentProjectName ?? 'project'}.';
    });
  }

  Future<void> _pollTelemetry() async {
    final trainingActive = trainingStatus?['active'] == true;
    final evalActive = evaluationStatus?['active'] == true;
    final tuningActive = tuningStatus?['active'] == true;
    if (tuningActive && !_pollingTelemetry) {
      try {
        tuningStatus = await api.getJson('/tuning/status');
        notifyListeners();
      } catch (_) {}
    }
    if ((!trainingActive && !evalActive) || _pollingTelemetry) return;
    _pollingTelemetry = true;
    try {
      if (trainingActive) {
        final status = await api.getJson('/training/status');
        final res = await api.getJson(
          '/training/telemetry?since=${telemetry.length}',
        );
        final total = (res['total'] as num?)?.toInt() ?? 0;
        if (total < telemetry.length) telemetry.clear(); // new run started
        telemetry.addAll(
          (res['points'] as List? ?? []).cast<Map<String, dynamic>>(),
        );
        trainingStatus = status;
        // A finished run should appear in the runs list right away.
        if (status['active'] != true) await loadRuns(notify: false);
      }
      if (evalActive) {
        evaluationStatus = await api.getJson('/evaluation/status');
        if (evaluationStatus?['active'] != true) await loadRuns(notify: false);
      }
      notifyListeners();
    } catch (_) {
      // Backend unreachable; the regular refresh flow surfaces this.
    } finally {
      _pollingTelemetry = false;
    }
  }

  List<Map<String, dynamic>> runs = [];
  Map<String, dynamic>? evaluationStatus;
  Map<String, dynamic>? envConfig;
  List<String> configProblems = [];
  bool envConfigSaved = false;

  /// Effective observation/action dimensions for the ENABLED entries — the
  /// sizes the policy actually sees. The catalog endpoints (`/robot/...`)
  /// report the full space; the backend recomputes these from the saved config
  /// so the builders update live as sources are toggled.
  int obsVectorSize = 0;
  int actionVectorSize = 0;

  void _applyVectorSizes(dynamic sizes) {
    if (sizes is! Map) return;
    obsVectorSize =
        (sizes['observation_vector_size'] as num?)?.toInt() ?? obsVectorSize;
    actionVectorSize =
        (sizes['action_vector_size'] as num?)?.toInt() ?? actionVectorSize;
  }

  Map<String, dynamic>? advisor;
  Map<String, dynamic>? tuningStatus;

  /// Result of the last "Test reward" run ({reward, formula, terms}). The
  /// Reward Builder renders this directly instead of sniffing the status string.
  Map<String, dynamic>? lastRewardResult;

  Future<void> loadAdvisor({bool notify = true}) async {
    try {
      advisor = await api.getJson('/training/advisor');
      if (notify) notifyListeners();
    } catch (_) {}
  }

  Future<void> startTuning({
    String algorithm = 'PPO',
    int nTrials = 8,
    int timestepsPerTrial = 2000,
  }) async {
    await guard(() async {
      await api.postJson('/tuning/start', {
        'algorithm': algorithm,
        'n_trials': nTrials,
        'timesteps_per_trial': timestepsPerTrial,
      });
      tuningStatus = await api.getJson('/tuning/status');
      message = 'Tuning started: $nTrials trials of $algorithm.';
    });
  }

  Future<void> stopTuning() async {
    await guard(() async {
      await api.postJson('/tuning/stop', {});
      tuningStatus = await api.getJson('/tuning/status');
      message = 'Tuning stop requested.';
    });
  }

  Future<void> loadEnvConfig({bool notify = true}) async {
    try {
      final res = await api.getJson('/env/config');
      envConfig = res['config'] as Map<String, dynamic>?;
      envConfigSaved = res['saved'] == true;
      configProblems = [
        for (final p in (res['problems'] as List? ?? [])) p.toString(),
      ];
      _applyVectorSizes(res['vector_sizes']);
      if (notify) notifyListeners();
    } catch (_) {}
  }

  Future<void> patchConfig(Map<String, dynamic> patch) async {
    await guard(() async {
      final res = await api.postJson('/env/config/patch', patch);
      envConfig = res['config'] as Map<String, dynamic>?;
      envConfigSaved = true;
      configProblems = [
        for (final p in (res['problems'] as List? ?? [])) p.toString(),
      ];
      _applyVectorSizes(res['vector_sizes']);
      message = configProblems.isEmpty
          ? 'Environment config updated.'
          : 'Config updated with problems: ${configProblems.join('; ')}';
    });
  }

  Future<Map<String, dynamic>> validateCustomReward(String code) {
    return api.postJson('/reward/validate_custom', {'code': code});
  }

  Future<void> loadRuns({bool notify = true}) async {
    try {
      // Scope to the open project unless the user asked to see everything (or
      // no project is loaded yet, in which case we show all).
      final pid = currentProjectId;
      final scoped = !runsShowAll && pid != null && pid.isNotEmpty;
      final query = scoped
          ? '?project_id=${Uri.encodeQueryComponent(pid)}'
          : '';
      final res = await api.getJson('/runs$query');
      runs = (res['runs'] as List? ?? []).cast<Map<String, dynamic>>();
      if (notify) notifyListeners();
    } catch (_) {}
  }

  Future<void> startEvaluation(
    String runName, {
    int episodes = 3,
    bool deterministic = true,
  }) async {
    await guard(() async {
      await api.postJson('/evaluation/start', {
        'run_name': runName,
        'episodes': episodes,
        'deterministic': deterministic,
      });
      evaluationStatus = await api.getJson('/evaluation/status');
      message = 'Evaluating $runName ($episodes episodes).';
    });
  }

  Future<void> stopEvaluation() async {
    await guard(() async {
      await api.postJson('/evaluation/stop', {});
      evaluationStatus = await api.getJson('/evaluation/status');
      message = 'Stopping evaluation…';
    });
  }

  Future<Map<String, dynamic>?> fetchRunDetails(String runName) async {
    try {
      return await api.getJson('/runs/$runName');
    } catch (_) {
      return null;
    }
  }

  /// Tail of the backend log file (the real terminal output), for the Logs tab.
  Future<List<String>> fetchBackendLogs({int lines = 500}) async {
    try {
      final res = await api.getJson('/logs/backend?lines=$lines');
      return [
        for (final line in (res['lines'] as List? ?? [])) line.toString(),
      ];
    } catch (_) {
      return const [];
    }
  }

  Future<String?> exportRun(String runName) async {
    try {
      final res = await api.postJson('/runs/$runName/export', {});
      return res['path']?.toString();
    } catch (e) {
      message = e.toString();
      notifyListeners();
      return null;
    }
  }

  /// Permanently delete a run (its whole directory) and refresh the list.
  Future<bool> deleteRun(String runName) async {
    try {
      await api.postJson('/runs/$runName/delete', {});
      await loadRuns(notify: false);
      message = 'Deleted run $runName.';
      notifyListeners();
      return true;
    } catch (e) {
      message = e.toString();
      notifyListeners();
      return false;
    }
  }

  Map<String, dynamic>? health;
  Map<String, dynamic>? robotInfo;
  Map<String, dynamic>? observations;
  Map<String, dynamic>? actions;
  Map<String, dynamic>? trainingStatus;
  // Full multi-provider agent settings: {active_provider, ollama:{…}, openai:{…}}.
  Map<String, dynamic>? agentSettings;
  Map<String, dynamic>? appPreferences;
  String message = 'Backend not checked yet.';
  bool busy = false;
  double streamResolutionScale = 1.0;
  bool showInspectorOnDashboard = true;
  String agentAutonomy = 'act'; // 'act' = run tools freely, 'ask' = confirm
  int robotLoadRevision = 0;
  String? lastLoadedRobotPath;

  // Persisted training controls — shared by the Training tab and the Home
  // training card so the chosen algorithm / timesteps / tuned params survive
  // navigation (previously they snapped back to PPO when leaving the page).
  String trainingAlgorithm = 'PPO';
  int trainingTimesteps = 10000;
  Map<String, dynamic> trainingParams = const {
    'learning_rate': 0.0003,
    'batch_size': 64,
    'gamma': 0.99,
    'n_steps': 256,
  };

  void setTrainingAlgorithm(String value) {
    if (trainingAlgorithm == value) return;
    trainingAlgorithm = value;
    notifyListeners();
  }

  void setTrainingTimesteps(int value) {
    trainingTimesteps = value;
  }

  void setTrainingParams(Map<String, dynamic> value) {
    trainingParams = value;
    notifyListeners();
  }

  Future<void> setAgentAutonomy(String value) async {
    agentAutonomy = value == 'ask' ? 'ask' : 'act';
    notifyListeners();
    await savePreferences();
  }

  Future<Map<String, dynamic>> executeAgentTool(
    String name,
    Map<String, dynamic> args,
  ) async {
    final res = await api.postJson('/agents/execute_tool', {
      'name': name,
      'args': args,
    });
    unawaited(refreshAll());
    return (res['result'] as Map?)?.cast<String, dynamic>() ?? {};
  }

  Future<Map<String, dynamic>> checkModelCapabilities() {
    return api.getJson('/agent/capabilities');
  }

  void setStreamResolutionScale(double value) {
    streamResolutionScale = value.clamp(0.5, 1.5).toDouble();
    message =
        'Viewport resolution scale set to ${streamResolutionScale.toStringAsFixed(2)}x.';
    notifyListeners();
  }

  Future<void> refreshAll() async {
    await guard(() async {
      health = await api.getJson('/health');
      robotInfo = await api.getJson('/robot/info');
      observations = await api.getJson('/robot/observations');
      actions = await api.getJson('/robot/actions');
      trainingStatus = await api.getJson('/training/status');
      agentSettings = await api.getJson('/agent/providers');
      appPreferences = await api.getJson('/app/preferences');
      await loadRuns(notify: false);
      await loadEnvConfig(notify: false);
      await loadAdvisor(notify: false);
      streamResolutionScale =
          ((appPreferences?['stream_resolution_scale'] as num?)?.toDouble() ??
                  streamResolutionScale)
              .clamp(0.5, 1.5)
              .toDouble();
      showInspectorOnDashboard =
          appPreferences?['show_inspector_on_dashboard'] as bool? ??
          showInspectorOnDashboard;
      agentAutonomy =
          appPreferences?['agent_autonomy']?.toString() ?? agentAutonomy;
      message = 'Connected to backend.';
    });
  }

  Future<void> loadUrdf({
    required String path,
    required List<double> basePosition,
    required bool fixedBase,
    required bool addPlane,
  }) async {
    await guard(() async {
      await api.postJson('/simulation/load_urdf', {
        'path': path,
        'base_position': basePosition,
        'fixed_base': fixedBase,
        'add_plane': addPlane,
      });
      robotInfo = await api.getJson('/robot/info');
      observations = await api.getJson('/robot/observations');
      actions = await api.getJson('/robot/actions');
      await loadEnvConfig(notify: false);
      robotLoadRevision += 1;
      lastLoadedRobotPath =
          robotInfo?['source_path']?.toString() ??
          robotInfo?['path']?.toString() ??
          path;
      message = 'Loaded URDF.';
    });
  }

  /// Per-link mass/inertia/collision health of the loaded robot, surfaced by
  /// /robot/info as {ok, error_count, warning_count, summary}. Drives the
  /// Auto-fix banner in Robot Setup.
  Map<String, dynamic>? get robotDynamics =>
      (robotInfo?['dynamics'] as Map?)?.cast<String, dynamic>();

  bool get robotDynamicsHasIssues {
    final d = robotDynamics;
    if (d == null) return false;
    return ((d['error_count'] as num?) ?? 0) > 0 ||
        ((d['warning_count'] as num?) ?? 0) > 0;
  }

  /// Clamp implausible masses and rebuild degenerate inertia tensors in place.
  Future<void> fixRobotDynamics() async {
    await guard(() async {
      final res = await api.postJson('/robot/fix_dynamics', {});
      robotInfo = await api.getJson('/robot/info');
      message = res['summary']?.toString() ?? 'Repaired robot dynamics.';
    });
  }

  Future<void> resetSimulation() async {
    await guard(() async {
      await api.postJson('/simulation/reset', {'reload_current_urdf': true});
      message = 'Simulation reset.';
    });
  }

  Future<void> setGravity(double z) async {
    await guard(() async {
      await api.postJson('/simulation/set_gravity', {
        'gravity': [0.0, 0.0, z],
      });
      message = 'Gravity updated.';
    });
  }

  Future<void> zeroAction() async {
    final count = (actions?['action_vector_size'] as num?)?.toInt() ?? 0;
    await guard(() async {
      await api.postJson('/robot/action_test', {
        'values': List.filled(count, 0.0),
        'mode': 'position',
      });
      message = 'Applied zero action.';
    });
  }

  Future<void> randomAction() async {
    final count = (actions?['action_vector_size'] as num?)?.toInt() ?? 0;
    final values = List.generate(count, (i) => i.isEven ? 0.15 : -0.15);
    await guard(() async {
      await api.postJson('/robot/action_test', {
        'values': values,
        'mode': 'position',
      });
      message = 'Applied safe random action.';
    });
  }

  Future<void> testReward() async {
    final components = (envConfig?['rewards'] as List? ?? []).toList();
    await guard(() async {
      final result = await api.postJson('/reward/test', {
        'components': components,
      });
      lastRewardResult = result;
      message = 'Reward ${result['reward']} | ${result['formula']}';
    });
  }

  Future<void> saveEnvConfig(String? urdfPath) async {
    await guard(() async {
      // The backend config service derives the config from the loaded robot.
      await api.postJson('/env/save_config', {});
      await loadEnvConfig(notify: false);
      message = 'Saved current environment config.';
    });
  }

  Future<void> startTraining(
    String? urdfPath, {
    String algorithm = 'PPO',
    int totalTimesteps = 10000,
    double learningRate = 0.0003,
    Map<String, dynamic> hyperparams = const {},
  }) async {
    await guard(() async {
      final blockers = trainingBlockers();
      if (blockers.isNotEmpty) {
        message = 'Training locked: ${blockers.first}';
        return;
      }
      final params = <String, dynamic>{
        'learning_rate': learningRate,
        'batch_size': 64,
        'gamma': 0.99,
        'n_steps': 256,
        ...hyperparams,
      };
      // No config payload: the backend builds and validates it server-side.
      await api.postJson('/training/start', {
        'algorithm': algorithm,
        'total_timesteps': totalTimesteps,
        'learning_rate': params['learning_rate'],
        'batch_size': params['batch_size'],
        'gamma': params['gamma'],
        'n_steps': params['n_steps'],
        if (params['ent_coef'] != null) 'ent_coef': params['ent_coef'],
        if (params['clip_range'] != null) 'clip_range': params['clip_range'],
        if (params['tau'] != null) 'tau': params['tau'],
        if (params['buffer_size'] != null) 'buffer_size': params['buffer_size'],
        if (params['train_freq'] != null) 'train_freq': params['train_freq'],
        if (params['net_arch'] != null) 'net_arch': params['net_arch'],
        'policy_type': 'MlpPolicy',
        'checkpoint_every': totalTimesteps >= 5000 ? totalTimesteps ~/ 5 : 0,
      });
      telemetry.clear();
      trainingStatus = await api.getJson('/training/status');
      message = 'Started $algorithm training.';
    });
  }

  Future<void> stopTraining() async {
    await guard(() async {
      await api.postJson('/training/stop', {});
      trainingStatus = await api.getJson('/training/status');
      message = 'Stop requested.';
    });
  }

  bool get hasRobot =>
      robotInfo?['loaded'] == true || robotInfo?['name'] != null;

  bool get hasEnabledObservations =>
      ((envConfig?['observations'] as List?) ?? []).any(
        (item) => item is Map && item['enabled'] == true,
      );

  bool get hasEnabledActions => ((envConfig?['actions'] as List?) ?? []).any(
    (item) => item is Map && item['enabled'] == true,
  );

  bool get hasEnabledRewards => ((envConfig?['rewards'] as List?) ?? []).any(
    (item) => item is Map && item['enabled'] == true,
  );

  List<String> trainingBlockers() {
    final blockers = <String>[];
    if (!hasRobot) blockers.add('load a robot');
    if (!envConfigSaved) blockers.add('save the environment setup');
    if (!hasEnabledObservations) blockers.add('enable observations');
    if (!hasEnabledActions) blockers.add('enable actions');
    if (!hasEnabledRewards) blockers.add('set a reward');
    blockers.addAll(configProblems);
    return blockers;
  }

  bool get canStartTraining => trainingBlockers().isEmpty;

  bool get trainingActive => trainingStatus?['active'] == true;
  bool get tuningActive => tuningStatus?['active'] == true;
  bool get evaluationActive => evaluationStatus?['active'] == true;

  /// True while any heavy backend job runs. Training, tuning and evaluation all
  /// drive the single shared PyBullet world, so only one runs at a time — the
  /// start controls disable while one is active.
  bool get anyJobActive => trainingActive || tuningActive || evaluationActive;

  /// The job currently blocking a new training start (training itself excluded),
  /// for a clear "locked because X" message.
  String? get blockingJobLabel =>
      tuningActive ? 'Tuning' : (evaluationActive ? 'Evaluation' : null);

  Future<void> savePreferences() async {
    await guard(() async {
      await api.postJson('/app/preferences', {
        'stream_resolution_scale': streamResolutionScale,
        'show_inspector_on_dashboard': showInspectorOnDashboard,
        'agent_autonomy': agentAutonomy,
      });
      appPreferences = await api.getJson('/app/preferences');
      message = 'Saved app preferences.';
    });
  }

  /// Persist the full multi-provider settings ({active_provider, ollama, openai})
  /// plus app preferences, then re-check the active provider's connection.
  Future<void> saveAgentSettings(Map<String, dynamic> providers) async {
    await guard(() async {
      await api.postJson('/agent/providers', providers);
      await api.postJson('/app/preferences', {
        'stream_resolution_scale': streamResolutionScale,
        'show_inspector_on_dashboard': showInspectorOnDashboard,
        'agent_autonomy': agentAutonomy,
      });
      agentSettings = await api.getJson('/agent/providers');
      appPreferences = await api.getJson('/app/preferences');
      message = 'Saved agent settings.';
    });
    await refreshAgentHealth();
  }

  Future<String> chat(String text) async {
    final result = await api.postJson('/agents/chat', {
      'agent': 'helper',
      'message': text,
      'context': const {},
    });
    return result['reply']?.toString() ?? '';
  }

  /// Streams agent events: chunk, tool_call, tool_result, notice, done.
  Stream<Map<String, dynamic>> chatEvents(
    String text, {
    String agent = 'helper',
    List<Map<String, String>> history = const [],
  }) async* {
    // Deliberately send NO client context. The backend assembles a small,
    // curated app-state summary itself; dumping the full robotInfo (40 links,
    // joints, meshes) here made the model summarize that blob instead of acting.
    await for (final event in api.streamPostJson('/agents/chat/stream', {
      'agent': agent,
      'message': text,
      'context': const <String, dynamic>{},
      'history': history,
    })) {
      if (event['type'] == 'error') {
        throw Exception(event['detail'] ?? 'Agent stream failed.');
      }
      yield event;
      // Tools may have changed app state (robot loaded, training started).
      if (event['type'] == 'done') {
        unawaited(refreshAll());
      }
    }
  }

  Future<void> guard(Future<void> Function() fn) async {
    busy = true;
    notifyListeners();
    try {
      await fn();
    } catch (e) {
      message = e.toString();
    } finally {
      busy = false;
      notifyListeners();
    }
  }
}
