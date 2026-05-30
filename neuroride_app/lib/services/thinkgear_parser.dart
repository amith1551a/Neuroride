import 'dart:typed_data';
import '../models/eeg_data.dart';

/// Decodes a streaming byte buffer of NeuroSky TGAM1 ThinkGear packets.
///
/// Feed raw bytes from Bluetooth with [addBytes]. Returns a decoded [EegData]
/// when a complete, valid packet containing EEG power bands is found.
///
/// Usage:
/// ```dart
/// final parser = ThinkGearParser();
/// bluetoothStream.listen((bytes) {
///   final reading = parser.addBytes(bytes);
///   if (reading != null) processReading(reading);
/// });
/// ```
class ThinkGearParser {
  final _buf = <int>[];

  /// Feed a chunk of raw Bluetooth bytes into the parser.
  /// Returns [EegData] when a complete valid packet is decoded, otherwise null.
  EegData? addBytes(Uint8List bytes) {
    _buf.addAll(bytes);
    return _tryParse();
  }

  EegData? _tryParse() {
    while (_buf.length >= 4) {
      // Step 1: Locate sync header 0xAA 0xAA
      if (_buf[0] != 0xAA || _buf[1] != 0xAA) {
        _buf.removeAt(0);
        continue;
      }

      final payloadLen = _buf[2];

      // Sanity check per ThinkGear spec
      if (payloadLen > 169) {
        _buf.removeAt(0);
        continue;
      }

      // Full packet = 2 sync + 1 length + payload + 1 checksum
      final totalLen = 2 + 1 + payloadLen + 1;
      if (_buf.length < totalLen) break; // Wait for more bytes

      // Verify checksum: ones-complement of sum of payload bytes
      int sum = 0;
      for (var i = 3; i < 3 + payloadLen; i++) {
        sum += _buf[i];
      }
      final expectedChecksum = (~sum) & 0xFF;

      if (expectedChecksum != _buf[2 + 1 + payloadLen]) {
        // Bad checksum — discard first byte and retry
        _buf.removeAt(0);
        continue;
      }

      // Extract payload and consume the packet from buffer
      final payload = List<int>.from(_buf.sublist(3, 3 + payloadLen));
      _buf.removeRange(0, totalLen);

      final result = _decodePayload(payload);
      if (result != null) return result;
    }
    return null;
  }

  EegData? _decodePayload(List<int> payload) {
    var i = 0;
    var signalStrength = 0;
    var attention = 0;
    var meditation = 0;
    double delta = 0, theta = 0, lowAlpha = 0, highAlpha = 0;
    double lowBeta = 0, highBeta = 0, lowGamma = 0, midGamma = 0;
    var hasWaves = false;

    while (i < payload.length) {
      final code = payload[i++];

      if (code == 0x02) {
        // Signal quality: 1 byte
        if (i < payload.length) signalStrength = payload[i++];
      } else if (code == 0x04) {
        // eSense Attention: 1 byte
        if (i < payload.length) attention = payload[i++];
      } else if (code == 0x05) {
        // eSense Meditation: 1 byte
        if (i < payload.length) meditation = payload[i++];
      } else if (code == 0x83) {
        // EEG power bands: length byte (should be 24) + 24 data bytes
        if (i >= payload.length) break;
        final len = payload[i++];
        if (len == 24 && i + 24 <= payload.length) {
          delta     = _readUint24(payload, i);
          theta     = _readUint24(payload, i + 3);
          lowAlpha  = _readUint24(payload, i + 6);
          highAlpha = _readUint24(payload, i + 9);
          lowBeta   = _readUint24(payload, i + 12);
          highBeta  = _readUint24(payload, i + 15);
          lowGamma  = _readUint24(payload, i + 18);
          midGamma  = _readUint24(payload, i + 21);
          hasWaves  = true;
          i += 24;
        } else {
          i += len; // Skip unexpected length
        }
      } else if (code >= 0x80) {
        // Other multi-byte extended code — skip
        if (i < payload.length) {
          i += payload[i] + 1;
        }
      } else {
        // Single-byte code — skip the value byte
        if (i < payload.length) i++;
      }
    }

    // Only return a reading if we decoded EEG bands
    if (!hasWaves) return null;

    return EegData(
      // Normalise raw TGAM1 24-bit values (0–16 777 215) to 0–100
      delta:          EegData.normalizeRaw(delta),
      theta:          EegData.normalizeRaw(theta),
      lowAlpha:       EegData.normalizeRaw(lowAlpha),
      highAlpha:      EegData.normalizeRaw(highAlpha),
      lowBeta:        EegData.normalizeRaw(lowBeta),
      highBeta:       EegData.normalizeRaw(highBeta),
      lowGamma:       EegData.normalizeRaw(lowGamma),
      midGamma:       EegData.normalizeRaw(midGamma),
      attention:      attention,
      meditation:     meditation,
      signalStrength: signalStrength,
      timestamp:      DateTime.now(),
    );
  }

  /// Reads a 3-byte big-endian unsigned integer from [data] at [offset].
  double _readUint24(List<int> data, int offset) =>
      ((data[offset] << 16) | (data[offset + 1] << 8) | data[offset + 2])
          .toDouble();

  /// Clears the internal byte buffer (call on disconnect).
  void reset() => _buf.clear();
}
