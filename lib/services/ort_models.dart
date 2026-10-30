import 'dart:typed_data';

import 'package:flutter_onnxruntime/flutter_onnxruntime.dart';

import '../core/pipeline.dart';
import 'model_manager.dart';

/// [FaceModels] backed by ONNX Runtime on the phone's CPU.
class OrtFaceModels implements FaceModels {
  final ModelManager manager;
  final OnnxRuntime _ort = OnnxRuntime();
  OrtSession? _detector;
  OrtSession? _recognizer;
  OrtSession? _swapper;
  OrtSession? _hyper;

  OrtFaceModels(this.manager);

  OrtSessionOptions get _options => OrtSessionOptions(
        intraOpNumThreads: 4,
        interOpNumThreads: 1,
        providers: [OrtProvider.CPU],
      );

  Future<OrtSession> _open(ModelSpec spec) => _ort.createSession(manager.pathOf(spec), options: _options);

  Future<OrtSession> get _det async => _detector ??= await _open(ModelManager.detector);
  Future<OrtSession> get _rec async => _recognizer ??= await _open(ModelManager.recognizer);
  Future<OrtSession> get _swp async => _swapper ??= await _open(ModelManager.swapper);
  Future<OrtSession> get _hyp async => _hyper ??= await _open(ModelManager.hyperswap);

  Future<void> warmUpDetector() async => _det;
  Future<void> warmUpSwapper(SwapModel model) async {
    if (model == SwapModel.hyperswap) {
      await _hyp;
    } else {
      await _swp;
    }
  }

  /// Frees the recognizer (only needed when choosing a new source face).
  Future<void> releaseRecognizer() async {
    final s = _recognizer;
    _recognizer = null;
    await s?.close();
  }

  Future<void> dispose() async {
    for (final s in [_detector, _recognizer, _swapper, _hyper]) {
      await s?.close();
    }
    _detector = _recognizer = _swapper = _hyper = null;
  }

  static List<double> _floats(List<dynamic> values) {
    if (values is List<double>) return values;
    return Float32List.fromList([for (final v in values) (v as num).toDouble()]);
  }

  Future<Map<String, List<double>>> _run(OrtSession session, Map<String, (Float32List, List<int>)> inputs) async {
    final tensors = <String, OrtValue>{};
    Map<String, OrtValue>? outputs;
    try {
      for (final e in inputs.entries) {
        tensors[e.key] = await OrtValue.fromList(e.value.$1, e.value.$2);
      }
      outputs = await session.run(tensors);
      final result = <String, List<double>>{};
      for (final e in outputs.entries) {
        result[e.key] = _floats(await e.value.asFlattenedList());
      }
      return result;
    } finally {
      for (final t in tensors.values) {
        await t.dispose();
      }
      if (outputs != null) {
        for (final t in outputs.values) {
          await t.dispose();
        }
      }
    }
  }

  @override
  Future<List<List<double>>> detect(Float32List input, int size) async {
    final s = await _det;
    final out = await _run(s, {
      s.inputNames.first: (input, [1, 3, size, size]),
    });
    return [
      for (final name in s.outputNames)
        out[name] ?? (throw StateError('Uscita $name mancante dal rilevatore')),
    ];
  }

  @override
  Future<List<double>> embed(Float32List input) async {
    final s = await _rec;
    final out = await _run(s, {
      s.inputNames.first: (input, [1, 3, 112, 112]),
    });
    return out[s.outputNames.first]!;
  }

  @override
  Future<List<double>> swap(Float32List target, Float32List latent) async {
    final s = await _swp;
    final out = await _run(s, {
      'target': (target, [1, 3, 128, 128]),
      'source': (latent, [1, latent.length]),
    });
    return out[s.outputNames.first]!;
  }

  @override
  Future<(List<double>, List<double>)> hyperSwap(Float32List target, Float32List source) async {
    final s = await _hyp;
    final out = await _run(s, {
      'target': (target, [1, 3, 256, 256]),
      'source': (source, [1, source.length]),
    });
    final image = out['output'] ?? out[s.outputNames.first]!;
    final mask = out['mask'] ?? (s.outputNames.length > 1 ? out[s.outputNames[1]]! : List<double>.filled(256 * 256, 1));
    return (image, mask);
  }
}
