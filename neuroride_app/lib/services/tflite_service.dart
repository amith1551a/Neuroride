import 'package:tflite_flutter/tflite_flutter.dart';
import '../models/eeg_data.dart';

const double _safeThreshold   = 0.35;
const double _dangerThreshold = 0.50;

/// Drowsiness level derived from the model probability.
enum DrowsinessLevel { safe, warning, danger }

/// Result of one inference call.
class PredictionResult {
  final double         probability;
  final DrowsinessLevel level;

  const PredictionResult({required this.probability, required this.level});

  bool   get isDanger => level == DrowsinessLevel.danger;
  bool   get isSafe   => level == DrowsinessLevel.safe;

  /// Human-readable label with emoji.
  String get label {
    switch (level) {
      case DrowsinessLevel.safe:    return '✅ SAFE';
      case DrowsinessLevel.warning: return '⚠️ WARNING';
      case DrowsinessLevel.danger:  return '🚨 DANGER';
    }
  }

  /// 0–1 progress value for a LinearProgressIndicator.
  double get progressValue => probability;
}

/// Loads `assets/neuroride_model.tflite` and runs per-reading inference.
class TfliteService {
  Interpreter? _interp;

  bool get isLoaded => _interp != null;

  /// Load the TFLite model from assets. Call once during app init.
  Future<void> load() async {
    _interp = await Interpreter.fromAsset('assets/neuroride_model.tflite');
  }

  /// Predict drowsiness for one [EegData] reading.
  ///
  /// Returns [PredictionResult] with level SAFE if model not yet loaded
  /// or if [data] has poor signal.
  PredictionResult predict(EegData data, Map<String, dynamic> scalerParams) {
    if (_interp == null || !data.hasGoodSignal) {
      return const PredictionResult(probability: 0, level: DrowsinessLevel.safe);
    }

    final features = data.toFeatureVector(scalerParams);
    // TFLite model expects shape [1, 14] input, outputs [1, 1] probability
    final input  = [features];
    final output = [[0.0]];

    _interp!.run(input, output);

    final prob = output[0][0].clamp(0.0, 1.0);
    final DrowsinessLevel lvl;
    if (prob < _safeThreshold) {
      lvl = DrowsinessLevel.safe;
    } else if (prob < _dangerThreshold) {
      lvl = DrowsinessLevel.warning;
    } else {
      lvl = DrowsinessLevel.danger;
    }

    return PredictionResult(probability: prob, level: lvl);
  }

  void dispose() {
    _interp?.close();
    _interp = null;
  }
}
