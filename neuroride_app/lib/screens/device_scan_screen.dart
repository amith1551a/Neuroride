import 'package:flutter/material.dart';
import 'package:flutter_bluetooth_serial/flutter_bluetooth_serial.dart';
import 'package:provider/provider.dart';
import '../providers/eeg_provider.dart';
import 'dashboard_screen.dart';

/// Shows all Bluetooth Classic devices paired with this phone.
/// User taps one to connect and navigate to DashboardScreen.
class DeviceScanScreen extends StatefulWidget {
  const DeviceScanScreen({super.key});

  @override
  State<DeviceScanScreen> createState() => _DeviceScanScreenState();
}

class _DeviceScanScreenState extends State<DeviceScanScreen> {
  List<BluetoothDevice> _devices = [];
  bool    _loading       = true;
  String? _connectingTo;

  @override
  void initState() {
    super.initState();
    _loadDevices();
  }

  Future<void> _loadDevices() async {
    setState(() => _loading = true);
    final devices = await context.read<EegProvider>().getPairedDevices();
    if (!mounted) return;
    setState(() {
      _devices = devices;
      _loading = false;
    });
  }

  Future<void> _connect(BluetoothDevice device) async {
    setState(() => _connectingTo = device.address);
    try {
      await context.read<EegProvider>().connect(device.address);
      if (!mounted) return;
      Navigator.of(context).pushReplacement(
        MaterialPageRoute(builder: (_) => const DashboardScreen()),
      );
    } catch (e) {
      if (!mounted) return;
      setState(() => _connectingTo = null);
      ScaffoldMessenger.of(context).showSnackBar(SnackBar(
        content: Text('Failed to connect: $e'),
        backgroundColor: Colors.redAccent,
      ));
    }
  }

  bool _isMindLink(BluetoothDevice d) =>
      (d.name ?? '').toLowerCase().contains('mindlink') ||
      (d.name ?? '').toLowerCase().contains('neurosky') ||
      (d.name ?? '').toLowerCase().contains('mindwave');

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFF0D1117),
      appBar: AppBar(
        backgroundColor: const Color(0xFF161B22),
        title: const Text(
          'Select MindLink Headband',
          style: TextStyle(color: Colors.white),
        ),
        actions: [
          IconButton(
            icon: const Icon(Icons.refresh, color: Colors.white70),
            onPressed: _loadDevices,
          ),
        ],
      ),
      body: _loading
          ? const Center(
              child: CircularProgressIndicator(color: Color(0xFF00E5FF)),
            )
          : _devices.isEmpty
              ? const Center(
                  child: Padding(
                    padding: EdgeInsets.all(28),
                    child: Text(
                      'No paired devices found.\n\n'
                      'Go to Android Settings → Connected devices → '
                      'Pair new device, pair your FT&S MindLink headband, '
                      'then return here.',
                      textAlign: TextAlign.center,
                      style: TextStyle(
                        color: Colors.white54,
                        fontSize: 15,
                        height: 1.6,
                      ),
                    ),
                  ),
                )
              : ListView.separated(
                  itemCount: _devices.length,
                  separatorBuilder: (_, __) =>
                      const Divider(color: Colors.white12, height: 1),
                  itemBuilder: (_, i) {
                    final d = _devices[i];
                    final connecting = _connectingTo == d.address;
                    final highlight  = _isMindLink(d);

                    return ListTile(
                      tileColor: const Color(0xFF161B22),
                      leading: Icon(
                        Icons.bluetooth,
                        color: highlight
                            ? const Color(0xFF00E5FF)
                            : Colors.white38,
                      ),
                      title: Text(
                        d.name ?? 'Unknown Device',
                        style: TextStyle(
                          color: highlight ? Colors.white : Colors.white70,
                          fontWeight: highlight
                              ? FontWeight.bold
                              : FontWeight.normal,
                        ),
                      ),
                      subtitle: Text(
                        d.address,
                        style: const TextStyle(
                          color: Colors.white30,
                          fontSize: 11,
                        ),
                      ),
                      trailing: connecting
                          ? const SizedBox(
                              width: 20,
                              height: 20,
                              child: CircularProgressIndicator(
                                strokeWidth: 2,
                                color: Color(0xFF00E5FF),
                              ),
                            )
                          : const Icon(
                              Icons.chevron_right,
                              color: Colors.white38,
                            ),
                      onTap: connecting ? null : () => _connect(d),
                    );
                  },
                ),
    );
  }
}
