import 'package:flutter_test/flutter_test.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import 'package:rtg_flutter_app/main.dart';

void main() {
  testWidgets('EasyRTG shell renders', (WidgetTester tester) async {
    await tester.pumpWidget(const ProviderScope(child: EasyRtgApp()));

    expect(find.text('EasyRTG'), findsWidgets);
    expect(find.text('3D Simulation'), findsOneWidget);
    expect(find.text('Robot Inspector'), findsOneWidget);
  });
}

