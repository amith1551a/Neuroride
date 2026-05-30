import 'dart:async';
import 'dart:typed_data';
import 'package:flutter_bluetooth_serial/flutter_bluetooth_serial.dart';

/// Connection states for the MindLink headband.
enum BtState { disconnected, connecting, connected, error }

/// Manages a Bluetooth Classic SPP connection to the FT&S MindLink headband.
///
/// Usage:
/// ```dart
/// final bt = BluetoothService();
/// bt.stateStream.listen((s) => print('State: $s'));
/// bt.dataStream.listen((bytes) => parser.addBytes(bytes));
/// await bt.connect('00:81:F9:XX:XX:XX');
/// ```
class BluetoothService {
  BluetoothConnection? _conn;

  final _stateCtrl = StreamController<BtState>.broadcast();
  final _dataCtrl  = StreamController<Uint8List>.broadcast();

  /// Stream of connection state changes.
  Stream<BtState>   get stateStream => _stateCtrl.stream;

  /// Stream of raw byte chunks from the headband.
  Stream<Uint8List> get dataStream  => _dataCtrl.stream;

  /// True when a Bluetooth connection is active.
  bool get isConnected => _conn?.isConnected ?? false;

  /// Returns all Bluetooth Classic devices already paired with this phone.
  ///
  /// The MindLink must be paired in Android Settings before it appears here.
  Future<List<BluetoothDevice>> getPairedDevices() =>
      FlutterBluetoothSerial.instance.getBondedDevices();

  /// Opens a Bluetooth Classic SPP connection to [address].
  ///
  /// Emits [BtState.connecting] then [BtState.connected] on success,
  /// or [BtState.error] on failure.
  Future<void> connect(String address) async {
    _stateCtrl.add(BtState.connecting);
    try {
      _conn = await BluetoothConnection.toAddress(address)
          .timeout(const Duration(seconds: 12));
      _stateCtrl.add(BtState.connected);

      _conn!.input!.listen(
        (Uint8List data) => _dataCtrl.add(data),
        onDone:  () => _stateCtrl.add(BtState.disconnected),
        onError: (_) => _stateCtrl.add(BtState.error),
        cancelOnError: false,
      );
    } catch (_) {
      _stateCtrl.add(BtState.error);
      rethrow;
    }
  }

  /// Closes the connection and emits [BtState.disconnected].
  Future<void> disconnect() async {
    await _conn?.close();
    _conn = null;
    if (!_stateCtrl.isClosed) _stateCtrl.add(BtState.disconnected);
  }

  /// Releases all resources. Call in the owning widget's dispose().
  void dispose() {
    disconnect();
    _stateCtrl.close();
    _dataCtrl.close();
  }
}
