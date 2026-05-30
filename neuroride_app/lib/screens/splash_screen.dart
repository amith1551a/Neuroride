import 'package:flutter/material.dart';
import 'package:flutter_bluetooth_serial/flutter_bluetooth_serial.dart';
import 'package:permission_handler/permission_handler.dart';
import 'device_scan_screen.dart';

/// First screen: requests permissions then navigates to DeviceScanScreen.
class SplashScreen extends StatefulWidget {
  const SplashScreen({super.key});

  @override
  State<SplashScreen> createState() => _SplashScreenState();
}

class _SplashScreenState extends State<SplashScreen> {
  String _status = 'Starting…';

  @override
  void initState() {
    super.initState();
    _init();
  }

  Future<void> _init() async {
    setState(() => _status = 'Requesting permissions…');
    await _requestPermissions();
    if (!mounted) return;

    setState(() => _status = 'Checking Bluetooth…');
    await _ensureBluetoothOn();
    if (!mounted) return;

    setState(() => _status = 'Ready!');
    await Future.delayed(const Duration(milliseconds: 400));
    if (!mounted) return;

    Navigator.of(context).pushReplacement(
      MaterialPageRoute(builder: (_) => const DeviceScanScreen()),
    );
  }

  Future<void> _requestPermissions() async {
    await [
      Permission.bluetoothConnect,
      Permission.bluetoothScan,
      Permission.location,
    ].request();
  }

  Future<void> _ensureBluetoothOn() async {
    final state = await FlutterBluetoothSerial.instance.state;
    if (state == BluetoothState.STATE_OFF) {
      setState(() => _status = 'Enabling Bluetooth…');
      await FlutterBluetoothSerial.instance.requestEnable();
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFF0D1117),
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            const Icon(Icons.memory, size: 80, color: Color(0xFF00E5FF)),
            const SizedBox(height: 20),
            const Text(
              'NeuroRide',
              style: TextStyle(
                fontSize: 34,
                fontWeight: FontWeight.bold,
                color: Colors.white,
                letterSpacing: 3,
              ),
            ),
            const SizedBox(height: 6),
            const Text(
              'EEG Rider Safety System',
              style: TextStyle(fontSize: 14, color: Colors.white54),
            ),
            const SizedBox(height: 52),
            const CircularProgressIndicator(color: Color(0xFF00E5FF)),
            const SizedBox(height: 20),
            Text(
              _status,
              style: const TextStyle(color: Colors.white60, fontSize: 14),
            ),
          ],
        ),
      ),
    );
  }
}
