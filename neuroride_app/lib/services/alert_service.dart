import 'package:audioplayers/audioplayers.dart';
import 'package:vibration/vibration.dart';

/// Triggers audio + vibration alarm when drowsiness is detected.
class AlertService {
  final _player = AudioPlayer();
  bool _active = false;

  bool get isActive => _active;

  /// Fire the alarm: 3 vibration pulses + alert sound.
  Future<void> triggerAlarm() async {
    if (_active) return;
    _active = true;

    // Vibrate: 3 strong pulses of 600 ms with 200 ms gaps
    if (await Vibration.hasVibrator()) {
      Vibration.vibrate(
        pattern:     [0, 600, 200, 600, 200, 600],
        intensities: [0, 255,   0, 255,   0, 255],
      );
    }

    // Play alert sound (fails silently if asset is missing/corrupt)
    try {
      await _player.play(AssetSource('alert_sound.mp3'));
    } catch (_) {
      // Vibration is the primary alert — audio is optional
    }
  }

  /// Stop alarm. Call when user taps the alarm banner.
  Future<void> stopAlarm() async {
    _active = false;
    await _player.stop();
    Vibration.cancel();
  }

  void dispose() {
    _player.dispose();
    Vibration.cancel();
  }
}
