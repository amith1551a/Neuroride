import 'package:fl_chart/fl_chart.dart';
import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../providers/eeg_provider.dart';
import 'device_scan_screen.dart';

/// Main screen: live EEG chart, drowsiness gauge, alarm banner.
class DashboardScreen extends StatelessWidget {
  const DashboardScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return Consumer<EegProvider>(
      builder: (context, prov, _) => Scaffold(
        backgroundColor: const Color(0xFF0D1117),
        appBar: _buildAppBar(context, prov),
        body: SingleChildScrollView(
          padding: const EdgeInsets.all(16),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              if (prov.alarmActive) ...[
                _AlarmBanner(provider: prov),
                const SizedBox(height: 12),
              ],
              _DrowsinessGauge(provider: prov),
              const SizedBox(height: 12),
              _SignalRow(provider: prov),
              const SizedBox(height: 12),
              _AttentionChart(provider: prov),
              const SizedBox(height: 12),
              _EegValuesCard(provider: prov),
            ],
          ),
        ),
      ),
    );
  }

  AppBar _buildAppBar(BuildContext ctx, EegProvider prov) {
    return AppBar(
      backgroundColor: const Color(0xFF161B22),
      title: Row(
        children: [
          Icon(
            Icons.circle,
            size: 10,
            color: prov.isConnected ? Colors.greenAccent : Colors.redAccent,
          ),
          const SizedBox(width: 8),
          Text(
            prov.isConnected ? 'NeuroRide — Live' : 'Disconnected',
            style: const TextStyle(color: Colors.white, fontSize: 16),
          ),
        ],
      ),
      actions: [
        IconButton(
          icon: const Icon(Icons.bluetooth_disabled, color: Colors.white54),
          onPressed: () async {
            await prov.disconnect();
            if (!ctx.mounted) return;
            Navigator.of(ctx).pushReplacement(
              MaterialPageRoute(builder: (_) => const DeviceScanScreen()),
            );
          },
        ),
      ],
    );
  }
}

// ── Alarm banner ───────────────────────────────────────────────────────────────

class _AlarmBanner extends StatelessWidget {
  final EegProvider provider;
  const _AlarmBanner({required this.provider});

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: provider.dismissAlarm,
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 14),
        decoration: BoxDecoration(
          color: const Color(0xFF7B1C1C),
          borderRadius: BorderRadius.circular(12),
          border: Border.all(color: Colors.redAccent, width: 2),
        ),
        child: const Row(
          children: [
            Icon(Icons.warning_amber_rounded,
                color: Colors.orangeAccent, size: 32),
            SizedBox(width: 12),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    'DROWSINESS DETECTED',
                    style: TextStyle(
                      color: Colors.white,
                      fontWeight: FontWeight.bold,
                      fontSize: 17,
                    ),
                  ),
                  Text(
                    'Tap to dismiss  •  Pull over safely',
                    style: TextStyle(color: Colors.white70, fontSize: 12),
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }
}

// ── Drowsiness gauge ───────────────────────────────────────────────────────────

class _DrowsinessGauge extends StatelessWidget {
  final EegProvider provider;
  const _DrowsinessGauge({required this.provider});

  @override
  Widget build(BuildContext context) {
    final pred  = provider.lastPrediction;
    final prob  = pred?.probability ?? 0.0;
    final label = pred?.label ?? '— Waiting for signal —';

    final color = prob >= 0.50
        ? Colors.redAccent
        : prob >= 0.35
            ? Colors.orangeAccent
            : Colors.greenAccent;

    return Container(
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        color: const Color(0xFF161B22),
        borderRadius: BorderRadius.circular(16),
      ),
      child: Column(
        children: [
          Text(
            label,
            style: TextStyle(
              fontSize: 26,
              fontWeight: FontWeight.bold,
              color: color,
            ),
          ),
          const SizedBox(height: 14),
          ClipRRect(
            borderRadius: BorderRadius.circular(8),
            child: LinearProgressIndicator(
              value: prob,
              minHeight: 14,
              backgroundColor: Colors.white10,
              valueColor: AlwaysStoppedAnimation<Color>(color),
            ),
          ),
          const SizedBox(height: 8),
          Text(
            '${(prob * 100).toStringAsFixed(1)}% drowsy probability',
            style: const TextStyle(color: Colors.white38, fontSize: 12),
          ),
        ],
      ),
    );
  }
}

// ── Signal quality row ─────────────────────────────────────────────────────────

class _SignalRow extends StatelessWidget {
  final EegProvider provider;
  const _SignalRow({required this.provider});

  @override
  Widget build(BuildContext context) {
    final r = provider.lastReading;
    if (r == null) {
      return const Center(
        child: Text(
          'Waiting for EEG packets…',
          style: TextStyle(color: Colors.white24, fontSize: 13),
        ),
      );
    }
    final good = r.hasGoodSignal;
    return Row(
      mainAxisAlignment: MainAxisAlignment.center,
      children: [
        Icon(
          good ? Icons.wifi : Icons.wifi_off,
          size: 16,
          color: good ? Colors.greenAccent : Colors.orange,
        ),
        const SizedBox(width: 6),
        Text(
          good ? 'Good signal' : 'Poor signal — adjust headband',
          style: TextStyle(
            color: good ? Colors.greenAccent : Colors.orange,
            fontSize: 13,
          ),
        ),
      ],
    );
  }
}

// ── Attention chart ────────────────────────────────────────────────────────────

class _AttentionChart extends StatelessWidget {
  final EegProvider provider;
  const _AttentionChart({required this.provider});

  @override
  Widget build(BuildContext context) {
    final hist = provider.attentionHistory;

    return Container(
      height: 140,
      padding: const EdgeInsets.fromLTRB(12, 10, 12, 8),
      decoration: BoxDecoration(
        color: const Color(0xFF161B22),
        borderRadius: BorderRadius.circular(16),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Text(
            'Attention  (last 60 s)',
            style: TextStyle(color: Colors.white38, fontSize: 11),
          ),
          const SizedBox(height: 6),
          Expanded(
            child: hist.isEmpty
                ? const Center(
                    child: Text(
                      'Chart will appear after connecting',
                      style: TextStyle(color: Colors.white12, fontSize: 11),
                    ),
                  )
                : LineChart(
                    LineChartData(
                      minY: 0,
                      maxY: 100,
                      gridData: const FlGridData(show: false),
                      titlesData: const FlTitlesData(show: false),
                      borderData: FlBorderData(show: false),
                      lineBarsData: [
                        LineChartBarData(
                          spots: hist
                              .asMap()
                              .entries
                              .map((e) =>
                                  FlSpot(e.key.toDouble(), e.value))
                              .toList(),
                          isCurved: true,
                          color: const Color(0xFF00E5FF),
                          barWidth: 2,
                          dotData: const FlDotData(show: false),
                          belowBarData: BarAreaData(
                            show: true,
                            color:
                                const Color(0xFF00E5FF).withAlpha(20),
                          ),
                        ),
                      ],
                    ),
                  ),
          ),
        ],
      ),
    );
  }
}

// ── Raw EEG values card ────────────────────────────────────────────────────────

class _EegValuesCard extends StatelessWidget {
  final EegProvider provider;
  const _EegValuesCard({required this.provider});

  @override
  Widget build(BuildContext context) {
    final r = provider.lastReading;
    if (r == null) return const SizedBox.shrink();

    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: const Color(0xFF161B22),
        borderRadius: BorderRadius.circular(16),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Text(
            'Live EEG Values',
            style: TextStyle(
              color: Colors.white54,
              fontWeight: FontWeight.bold,
            ),
          ),
          const SizedBox(height: 12),
          Wrap(
            spacing: 8,
            runSpacing: 8,
            children: [
              _Chip('Attention',  r.attention.toDouble(),  Colors.cyanAccent),
              _Chip('Meditation', r.meditation.toDouble(), Colors.purpleAccent),
              _Chip('Delta',      r.delta,                 Colors.blueAccent),
              _Chip('Theta',      r.theta,                 Colors.tealAccent),
              _Chip('α Low',      r.lowAlpha,              Colors.greenAccent),
              _Chip('α High',     r.highAlpha,             Colors.lightGreenAccent),
              _Chip('β Low',      r.lowBeta,               Colors.orangeAccent),
              _Chip('β High',     r.highBeta,              Colors.deepOrangeAccent),
            ],
          ),
        ],
      ),
    );
  }
}

class _Chip extends StatelessWidget {
  final String label;
  final double value;
  final Color  color;
  const _Chip(this.label, this.value, this.color);

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
      decoration: BoxDecoration(
        color: color.withAlpha(20),
        borderRadius: BorderRadius.circular(8),
        border: Border.all(color: color.withAlpha(90)),
      ),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          Text(
            label,
            style: TextStyle(color: color.withAlpha(180), fontSize: 10),
          ),
          Text(
            value.toStringAsFixed(1),
            style: TextStyle(color: color, fontWeight: FontWeight.bold),
          ),
        ],
      ),
    );
  }
}
