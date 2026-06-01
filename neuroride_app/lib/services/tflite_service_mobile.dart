class TfliteService {
  bool _loaded = false;

  Future<void> loadModel() async {
    _loaded = true;
  }

  bool get isLoaded => _loaded;

  Future<List<double>> predict(List<double> input) async {
    // Web/PWA fallback: mock inference until TensorFlow.js or backend API is added.
    if (!_loaded) {
      await loadModel();
    }

    final avg = input.isEmpty
        ? 0.0
        : input.reduce((a, b) => a + b) / input.length;

    return [avg];
  }

  void close() {
    _loaded = false;
  }
}
