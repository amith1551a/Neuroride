import 'dart:typed_data';
import 'package:flutter_test/flutter_test.dart';
import 'package:neuroride_app/services/thinkgear_parser.dart';

/// Builds a syntactically valid ThinkGear packet.
Uint8List buildPacket({
  int signal = 0,
  int attention = 60,
  int meditation = 50,
  List<int>? bands, // 24 raw bytes for 8 × 3-byte band values
}) {
  final payload = <int>[
    0x02, signal,
    0x04, attention,
    0x05, meditation,
    0x83, 24,
    ...(bands ?? List<int>.filled(24, 0)),
  ];
  int cs = payload.fold(0, (a, b) => a + b);
  cs = (~cs) & 0xFF;
  return Uint8List.fromList([0xAA, 0xAA, payload.length, ...payload, cs]);
}

/// Encodes an integer as 3 big-endian bytes.
List<int> u24(int v) => [(v >> 16) & 0xFF, (v >> 8) & 0xFF, v & 0xFF];

void main() {
  group('ThinkGearParser', () {
    test('parses valid packet — attention and meditation', () {
      final result = ThinkGearParser().addBytes(
          buildPacket(attention: 75, meditation: 40, signal: 0));
      expect(result, isNotNull);
      expect(result!.attention, equals(75));
      expect(result.meditation, equals(40));
      expect(result.signalStrength, equals(0));
    });

    test('returns null for incomplete packet', () {
      final parser = ThinkGearParser();
      final full = buildPacket();
      final result = parser.addBytes(Uint8List.fromList(full.sublist(0, 5)));
      expect(result, isNull);
    });

    test('completes after second chunk', () {
      final parser = ThinkGearParser();
      final full = buildPacket(attention: 55);
      parser.addBytes(Uint8List.fromList(full.sublist(0, 5)));
      final result = parser.addBytes(Uint8List.fromList(full.sublist(5)));
      expect(result, isNotNull);
      expect(result!.attention, equals(55));
    });

    test('rejects packet with bad checksum', () {
      final corrupt = buildPacket().toList();
      corrupt[corrupt.length - 1] ^= 0xFF; // flip bits
      final result = ThinkGearParser()
          .addBytes(Uint8List.fromList(corrupt));
      expect(result, isNull);
    });

    test('delta = 100 000 normalises to 100.0', () {
      final bands = [...u24(100000), ...List<int>.filled(21, 0)];
      final result = ThinkGearParser().addBytes(buildPacket(bands: bands));
      expect(result, isNotNull);
      expect(result!.delta, closeTo(100.0, 0.01));
    });

    test('theta = 50 000 normalises to 50.0', () {
      final bands = [...u24(0), ...u24(50000), ...List<int>.filled(18, 0)];
      final result = ThinkGearParser().addBytes(buildPacket(bands: bands));
      expect(result, isNotNull);
      expect(result!.theta, closeTo(50.0, 0.01));
    });

    test('reset clears buffer — leftover bytes ignored', () {
      final parser = ThinkGearParser();
      final full = buildPacket(attention: 88);
      parser.addBytes(Uint8List.fromList(full.sublist(0, 5)));
      parser.reset();
      // After reset the tail bytes have no header — no valid packet
      final result = parser.addBytes(Uint8List.fromList(full.sublist(5)));
      expect(result, isNull);
    });

    test('signal quality stored correctly', () {
      final result = ThinkGearParser()
          .addBytes(buildPacket(signal: 200, attention: 0, meditation: 0));
      expect(result, isNotNull);
      expect(result!.signalStrength, equals(200));
      expect(result.hasGoodSignal, isFalse);
    });
  });
}
