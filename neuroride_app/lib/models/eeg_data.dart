import 'dart:convert';
import 'package:flutter/services.dart';

/// One EEG reading decoded from a ThinkGear packet.
///
/// Wave values (delta…midGamma) are already normalised to 0–100
/// by the ThinkGear parser before being stored here.
class EegData {
  final double delta;
  final double theta;
  final double lowAlpha;
  final double highAlpha;
  final double lowBeta;
  final double highBeta;
  final double lowGamma;
  final double midGamma;
  final int attention;     // eSense 0–100
  final int meditation;    // eSense 0–100
  final int signalStrength; // 0 = good, 200 = off-head
  final DateTime timestamp;

  const EegData({
    required this.delta,
    required this.theta,
    required this.lowAlpha,
    required this.highAlpha,
    required this.lowBeta,
    required this.highBeta,
    required this.lowGamma,
    required this.midGamma,
    required this.attention,
    required this.meditation,
    required this.signalStrength,
    required this.timestamp,
  });

  /// True when the headband has good skin contact.
  bool get hasGoodSignal => signalStrength == 0;

  /// Normalises a raw TGAM1 24-bit band value (0–16 777 215) to 0–100.
  /// 100 000 is a representative maximum for typical indoor EEG readings.
  static double normalizeRaw(double raw) =>
      (raw / 100000.0 * 100.0).clamp(0.0, 100.0);

  /// Builds the 14-element feature vector expected by the TFLite model.
  ///
  /// Feature order (must match training data):
  ///   [0]  Delta Waves
  ///   [1]  Theta Waves
  ///   [2]  Low Alpha Waves
  ///   [3]  High Alpha Waves
  ///   [4]  Low Beta Waves
  ///   [5]  High Beta Waves
  ///   [6]  Low Gamma Waves
  ///   [7]  High Gamma Waves
  ///   [8]  Attention
  ///   [9]  Meditation
  ///   [10] Theta_Beta_Ratio
  ///   [11] Alpha_Beta_Ratio
  ///   [12] Delta_AB_Ratio
  ///   [13] Total_Power
  List<double> toFeatureVector(Map<String, dynamic> scalerParams) {
    final mean  = List<double>.from(scalerParams['mean']  as List);
    final scale = List<double>.from(scalerParams['scale'] as List);

    final thetaBeta  = theta / (lowBeta + highBeta + 0.001);
    final alphaBeta  = (lowAlpha + highAlpha) / (lowBeta + highBeta + 0.001);
    final deltaAB    = delta / (lowAlpha + highAlpha + 0.001);
    final totalPower = delta + theta + lowAlpha + highAlpha +
        lowBeta + highBeta + lowGamma + midGamma;

    final raw = [
      delta,    theta,    lowAlpha,  highAlpha,
      lowBeta,  highBeta, lowGamma,  midGamma,
      attention.toDouble(), meditation.toDouble(),
      thetaBeta, alphaBeta, deltaAB, totalPower,
    ];

    // Apply StandardScaler: z = (x - mean) / scale
    return List.generate(14, (i) => (raw[i] - mean[i]) / scale[i]);
  }
}

/// Loads and caches the scaler parameters from assets/scaler_params.json.
class ScalerParams {
  static Map<String, dynamic>? _cache;

  static Future<Map<String, dynamic>> load() async {
    if (_cache != null) return _cache!;
    final raw = await rootBundle.loadString('assets/scaler_params.json');
    _cache = json.decode(raw) as Map<String, dynamic>;
    return _cache!;
  }
}
