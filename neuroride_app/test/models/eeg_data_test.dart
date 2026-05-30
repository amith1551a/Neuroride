import 'package:flutter_test/flutter_test.dart';
import 'package:neuroride_app/models/eeg_data.dart';

EegData _sample({int signalStrength = 0}) => EegData(
      delta: 5.0, theta: 8.0, lowAlpha: 3.0, highAlpha: 2.0,
      lowBeta: 3.0, highBeta: 4.0, lowGamma: 2.0, midGamma: 1.0,
      attention: 48, meditation: 57,
      signalStrength: signalStrength,
      timestamp: DateTime(2026),
    );

Map<String, dynamic> _identityScaler() => {
      'mean':          List<double>.filled(14, 0.0),
      'scale':         List<double>.filled(14, 1.0),
      'feature_names': List.generate(14, (i) => 'f$i'),
    };

void main() {
  group('EegData.toFeatureVector', () {
    test('returns exactly 14 elements', () {
      expect(_sample().toFeatureVector(_identityScaler()).length, equals(14));
    });

    test('delta is first element with identity scaler', () {
      final v = _sample().toFeatureVector(_identityScaler());
      expect(v[0], closeTo(5.0, 0.001));
    });

    test('attention is element [8] with identity scaler', () {
      final v = _sample().toFeatureVector(_identityScaler());
      expect(v[8], closeTo(48.0, 0.001));
    });

    test('meditation is element [9] with identity scaler', () {
      final v = _sample().toFeatureVector(_identityScaler());
      expect(v[9], closeTo(57.0, 0.001));
    });
  });

  group('EegData.normalizeRaw', () {
    test('100 000 maps to 100.0', () =>
        expect(EegData.normalizeRaw(100000), closeTo(100.0, 0.001)));
    test('50 000 maps to 50.0', () =>
        expect(EegData.normalizeRaw(50000), closeTo(50.0, 0.001)));
    test('0 maps to 0.0', () =>
        expect(EegData.normalizeRaw(0), closeTo(0.0, 0.001)));
    test('above max is clamped to 100.0', () =>
        expect(EegData.normalizeRaw(999999), closeTo(100.0, 0.001)));
  });

  group('EegData.hasGoodSignal', () {
    test('true when signalStrength == 0', () =>
        expect(_sample(signalStrength: 0).hasGoodSignal, isTrue));
    test('false when signalStrength == 200', () =>
        expect(_sample(signalStrength: 200).hasGoodSignal, isFalse));
  });
}
