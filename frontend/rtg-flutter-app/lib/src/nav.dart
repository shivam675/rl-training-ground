import 'package:flutter/material.dart';
import 'package:flutter_riverpod/legacy.dart';

/// The destinations reachable from the navigation rail, in display order. The
/// rail groups them as Home · BUILD · TRAIN · tools; [label]/[icon] drive both
/// the rail and the header breadcrumb.
enum AppPage {
  home('Home', Icons.space_dashboard_outlined),
  robot('Robot Setup', Icons.precision_manufacturing),
  obsAction('Obs / Action', Icons.schema),
  rewards('Rewards', Icons.functions),
  training('Training', Icons.school),
  evaluation('Evaluation', Icons.fact_check),
  // Entered programmatically during Isaac training live-view / evaluation — not
  // in the rail. Shows the WebRTC live stream or a recorded-eval video.
  viewer('3D View', Icons.view_in_ar),
  settings('Settings', Icons.settings),
  logs('Logs', Icons.terminal);

  const AppPage(this.label, this.icon);

  final String label;
  final IconData icon;
}

/// Selected page in the navigation rail; panels can jump the user elsewhere
/// (e.g. starting an evaluation switches to the 3D View page).
final navIndexProvider = StateProvider<AppPage>((ref) => AppPage.home);
