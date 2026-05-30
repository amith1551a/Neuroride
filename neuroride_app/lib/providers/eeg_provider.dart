import 'dart:async';
import 'package:flutter/foundation.dart';
import 'package:flutter_bluetooth_serial/flutter_bluetooth_serial.dart';
import '../models/eeg_data.dart';
import '../services/bluetooth_service.dart';
import '../services/thinkgear_parser.dart';
import '../services/tflite_service.dart';
import '../services/alert_service.dart';

/// Number of consecutive DANGER predictions before firing the alarm.
const int _alarmThreshold = 3;

/// Maximum data points kept in chart history (60 seconds at ~1 Hz).
const int _maxHistory = 60;

/// Central state manager for the NeuroRide app.
///
/// Wires BluetoothService → ThinkGearParser → TfliteService → AlertService
/// and exposes streams + state for the UI via ChangeNotifier.
class EegProvider extends ChangeNotifier {
  final BluetoothService _bt;
  final ThinkGearParser  _parser;
  final TfliteService    _tflite;
  final AlertService     _alerts;

  Map<String, dynamic>? _scalerParams;
  StreamSubscription<Uint8List>? _dataSub;
  StreamSubscription<BtState>?   _stateSub;

  EegData?          _lastReading;
  PredictionResult? _lastPrediction;
  BtState           _btState = BtState.disconnected;
  int               _consecutiveDanger = 0;
  bool              _alarmActive = false;

  final _attentionHistory   = <double>[];
  final _meditationHistory  = <double>[];
  final _probabilityHistory = <double>[];

  EegProvider({
    required BluetoothService bluetooth,
    required ThinkGearParser  parser,
    required TfliteService    tflite,
    required AlertService     alertService,
  })  : _bt      = bluetooth,
        _parser  = parser,
        _tflite  = tflite,
        _alerts  = alertService;

  // ── Public getters ───────────────────────────────────────────────────────────
  EegData?          get lastReading        => _lastReading;
  PredictionResult? get lastPrediction     => _lastPrediction;
  BtState           get connectionState    => _btState;
  bool              get isConnected        => _btState == BtState.connected;
  bool              get alarmActive        => _alarmActive;
  List<double>      get attentionHistory   => List.unmodifiable(_attentionHistory);
  List<double>      get meditationHistory  => List.unmodifiable(_meditationHistory);
  List<double>      get probabilityHistory => List.unmodifiable(_probabilityHistory);

  // ── Initialisation ───────────────────────────────────────────────────────────
  Future<void> initialize(Map<String, dynamic> scalerParams) async {
    _scalerParams = scalerParams;
    await _tflite.load();

    _stateSub = _bt.stateStream.listen((state) {
      _btState = state;
      notifyListeners();
    });
  }

  // ── Bluetooth ────────────────────────────────────────────────────────────────
  Future<List<BluetoothDevice>> getPairedDevices() =>
      _bt.getPairedDevices();

  Future<void> connect(String address) async {
    await _bt.connect(address);
    _dataSub = _bt.dataStream.listen(_onData);
  }

  Future<void> disconnect() async {
    await _dataSub?.cancel();
    _dataSub = null;
    await _bt.disconnect();
    _parser.reset();
  }

  // ── Data pipeline ─────────────────────────────────────────────────────────────
  void _onData(Uint8List bytes) {
    final reading = _parser.addBytes(bytes);
    if (reading == null) return;

    _lastReading = reading;

    // Skip inference if signal is poor or model not ready
    if (!reading.hasGoodSignal || _scalerParams == null || !_tflite.isLoaded) {
      notifyListeners();
      return;
    }

    final pred = _tflite.predict(reading, _scalerParams!);
    _lastPrediction = pred;

    _push(_attentionHistory,   reading.attention.toDouble());
    _push(_meditationHistory,  reading.meditation.toDouble());
    _push(_probabilityHistory, pred.probability);

    // Alarm logic: N consecutive DANGER readings
    if (pred.isDanger) {
      _consecutiveDanger++;
      if (_consecutiveDanger >= _alarmThreshold && !_alarmActive) {
        _alarmActive = true;
        _alerts.triggerAlarm();
      }
    } else {
      _consecutiveDanger = 0;
    }

    notifyListeners();
  }

  void _push(List<double> list, double value) {
    list.add(value);
    if (list.length > _maxHistory) list.removeAt(0);
  }

  // ── Alarm control ─────────────────────────────────────────────────────────────
  Future<void> dismissAlarm() async {
    _alarmActive       = false;
    _consecutiveDanger = 0;
    await _alerts.stopAlarm();
    notifyListeners();
  }

  @override
  void dispose() {
    _dataSub?.cancel();
    _stateSub?.cancel();
    _bt.dispose();
    _tflite.dispose();
    _alerts.dispose();
    super.dispose();
  }
}
