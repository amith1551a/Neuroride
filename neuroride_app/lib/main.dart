import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'models/eeg_data.dart';
import 'providers/eeg_provider.dart';
import 'screens/splash_screen.dart';
import 'services/alert_service.dart';
import 'services/bluetooth_service.dart';
import 'services/thinkgear_parser.dart';
import 'services/tflite_service.dart';

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  runApp(const NeuroRideApp());
}

class NeuroRideApp extends StatelessWidget {
  const NeuroRideApp({super.key});

  @override
  Widget build(BuildContext context) {
    return FutureBuilder<Map<String, dynamic>>(
      future: ScalerParams.load(),
      builder: (context, snapshot) {
        // Show spinner while scaler params load from assets
        if (!snapshot.hasData) {
          return const MaterialApp(
            debugShowCheckedModeBanner: false,
            home: Scaffold(
              backgroundColor: Color(0xFF0D1117),
              body: Center(
                child: CircularProgressIndicator(color: Color(0xFF00E5FF)),
              ),
            ),
          );
        }

        return ChangeNotifierProvider(
          create: (_) {
            final provider = EegProvider(
              bluetooth:    BluetoothService(),
              parser:       ThinkGearParser(),
              tflite:       TfliteService(),
              alertService: AlertService(),
            );
            // Load TFLite model + start BT state listener asynchronously
            provider.initialize(snapshot.data!);
            return provider;
          },
          child: MaterialApp(
            title: 'NeuroRide',
            debugShowCheckedModeBanner: false,
            theme: ThemeData(
              brightness: Brightness.dark,
              colorScheme: const ColorScheme.dark(
                primary:   Color(0xFF00E5FF),
                secondary: Color(0xFF00E5FF),
              ),
              scaffoldBackgroundColor: const Color(0xFF0D1117),
              appBarTheme: const AppBarTheme(
                backgroundColor: Color(0xFF161B22),
                elevation: 0,
              ),
            ),
            home: const SplashScreen(),
          ),
        );
      },
    );
  }
}
