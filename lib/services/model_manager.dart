import 'dart:async';
import 'dart:io';
import 'dart:typed_data';

import 'package:flutter/foundation.dart';
import 'package:path_provider/path_provider.dart';

import '../core/onnx_reader.dart';

class ModelSpec {
  final String label;
  final String fileName;
  final int bytes;

  const ModelSpec(this.label, this.fileName, this.bytes);

  String get url => '${ModelManager.baseUrl}$fileName';
}

class DownloadCancelled implements Exception {
  const DownloadCancelled();
}

/// Downloads and stores the three ONNX models (from FaceFusion's public
/// GitHub releases) in the app's private folder.
class ModelManager {
  static const baseUrl = 'https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/';
  static const detector = ModelSpec('Rilevamento volti (SCRFD)', 'scrfd_2.5g.onnx', 3295067);
  static const recognizer = ModelSpec('Riconoscimento volto (ArcFace)', 'arcface_w600k_r50.onnx', 174388474);
  static const swapper = ModelSpec('Scambio volti (inswapper 128)', 'inswapper_128.onnx', 555303150);
  static const all = [detector, recognizer, swapper];
  static const _emapFile = 'inswapper_emap.bin';
  static const _emapBytes = 512 * 512 * 4;

  final Directory dir;

  ModelManager(this.dir);

  static Future<ModelManager> create() async {
    final base = await getApplicationSupportDirectory();
    final dir = Directory('${base.path}/models');
    await dir.create(recursive: true);
    return ModelManager(dir);
  }

  String pathOf(ModelSpec spec) => '${dir.path}/${spec.fileName}';

  bool isReady(ModelSpec spec) {
    final f = File(pathOf(spec));
    return f.existsSync() && f.lengthSync() == spec.bytes;
  }

  bool get allReady => all.every(isReady);

  int get totalBytes => all.fold(0, (a, s) => a + s.bytes);

  int downloadedBytes() {
    var sum = 0;
    for (final s in all) {
      if (isReady(s)) {
        sum += s.bytes;
        continue;
      }
      final part = File('${pathOf(s)}.part');
      if (part.existsSync()) sum += part.lengthSync();
    }
    return sum;
  }

  /// Downloads every missing model, resuming partial files.
  /// [onProgress] gets (bytes done, total bytes).
  Future<void> downloadAll({
    required void Function(int done, int total) onProgress,
    required bool Function() isCancelled,
  }) async {
    final total = totalBytes;
    for (final spec in all) {
      if (isReady(spec)) continue;
      await _download(spec, isCancelled, (specDone) {
        var done = 0;
        for (final s in all) {
          if (s == spec) {
            done += specDone;
          } else if (isReady(s)) {
            done += s.bytes;
          } else {
            final part = File('${pathOf(s)}.part');
            if (part.existsSync()) done += part.lengthSync();
          }
        }
        onProgress(done, total);
      });
    }
    onProgress(total, total);
  }

  Future<void> _download(ModelSpec spec, bool Function() isCancelled, void Function(int) onBytes) async {
    final target = File(pathOf(spec));
    final part = File('${target.path}.part');
    var have = part.existsSync() ? part.lengthSync() : 0;
    if (have > spec.bytes) {
      part.deleteSync();
      have = 0;
    }
    if (have == spec.bytes) {
      await part.rename(target.path);
      return;
    }
    final client = HttpClient()..connectionTimeout = const Duration(seconds: 30);
    IOSink? sink;
    try {
      var uri = Uri.parse(spec.url);
      late HttpClientResponse response;
      var hops = 0;
      while (true) {
        final request = await client.getUrl(uri);
        request.followRedirects = false;
        if (have > 0) request.headers.set(HttpHeaders.rangeHeader, 'bytes=$have-');
        response = await request.close();
        if (response.statusCode >= 300 && response.statusCode < 400) {
          final location = response.headers.value(HttpHeaders.locationHeader);
          await response.drain<void>();
          if (location == null || ++hops > 8) {
            throw HttpException('Reindirizzamento non valido (${response.statusCode})');
          }
          uri = uri.resolve(location);
          continue;
        }
        break;
      }
      if (response.statusCode == 200) {
        have = 0;
      } else if (response.statusCode != 206) {
        await response.drain<void>();
        throw HttpException('Errore del server: ${response.statusCode}');
      }
      sink = part.openWrite(mode: have > 0 ? FileMode.append : FileMode.write);
      onBytes(have);
      var lastReport = DateTime.now();
      await for (final chunk in response) {
        if (isCancelled()) throw const DownloadCancelled();
        sink.add(chunk);
        have += chunk.length;
        final now = DateTime.now();
        if (now.difference(lastReport).inMilliseconds > 200) {
          lastReport = now;
          onBytes(have);
        }
      }
      await sink.flush();
      await sink.close();
      sink = null;
      final size = part.lengthSync();
      if (size != spec.bytes) {
        if (size > spec.bytes) part.deleteSync();
        throw HttpException('Download incompleto (${spec.fileName}: $size di ${spec.bytes} byte)');
      }
      await part.rename(target.path);
      onBytes(spec.bytes);
    } finally {
      try {
        await sink?.close();
      } catch (_) {}
      client.close(force: true);
    }
  }

  /// The inswapper "emap" matrix, extracted once from the model and cached.
  Future<Float32List> loadEmap() async {
    final cache = File('${dir.path}/$_emapFile');
    if (cache.existsSync() && cache.lengthSync() == _emapBytes) {
      final bytes = await cache.readAsBytes();
      return Float32List.fromList(bytes.buffer.asFloat32List(bytes.offsetInBytes, _emapBytes ~/ 4));
    }
    final modelPath = pathOf(swapper);
    final tensor = await compute(readLastInitializer, modelPath);
    if (tensor.dims.length != 2 || tensor.dims[0] != 512 || tensor.dims[1] != 512) {
      throw FormatException('Matrice emap inattesa: ${tensor.dims}');
    }
    await cache.writeAsBytes(tensor.data.buffer.asUint8List(), flush: true);
    return tensor.data;
  }

  Future<void> deleteAll() async {
    if (dir.existsSync()) await dir.delete(recursive: true);
    await dir.create(recursive: true);
  }
}
