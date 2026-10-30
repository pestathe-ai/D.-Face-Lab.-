import 'dart:io';
import 'dart:typed_data';

import 'package:ffmpeg_kit_flutter_new/ffmpeg_kit.dart';
import 'package:ffmpeg_kit_flutter_new/ffprobe_kit.dart';
import 'package:ffmpeg_kit_flutter_new/return_code.dart';
import 'package:flutter/foundation.dart';
import 'package:path_provider/path_provider.dart';

import '../core/codec.dart';
import '../core/pipeline.dart';
import '../core/rgb_image.dart';
import '../core/scrfd.dart';
import '../core/video_plan.dart';
import 'model_manager.dart';
import 'ort_models.dart';

class UserFacingError implements Exception {
  final String message;
  const UserFacingError(this.message);
  @override
  String toString() => message;
}

class ProcessingCancelled implements Exception {
  const ProcessingCancelled();
}

class SourceFace {
  final Float32List latent;
  final Uint8List preview;
  final int facesFound;
  const SourceFace(this.latent, this.preview, this.facesFound);
}

class SwapResult {
  final String path;
  final Uint8List preview;
  final bool isVideo;
  final int facesSwapped;
  final int framesTotal;
  final int framesWithFaces;
  final bool truncated;
  final double seconds;

  const SwapResult({
    required this.path,
    required this.preview,
    required this.isVideo,
    required this.facesSwapped,
    this.framesTotal = 1,
    this.framesWithFaces = 1,
    this.truncated = false,
    this.seconds = 0,
  });
}

typedef ProgressCallback = void Function(String stage, double? fraction);

RgbImage? _decodePath(String path) => decodeToRgb(File(path).readAsBytesSync());

Uint8List _previewPng((RgbImage, List<double>) args) => encodePng(cropPreview(args.$1, args.$2));

Uint8List _encodeJpg((RgbImage, int) args) => encodeJpeg(args.$1, quality: args.$2);

Uint8List _encodeJpgToFile((RgbImage, String, int) args) {
  final jpg = encodeJpeg(args.$1, quality: args.$3);
  File(args.$2).writeAsBytesSync(jpg, flush: true);
  return jpg;
}

Future<RgbImage> _decodeFile(String path) async {
  final image = await compute(_decodePath, path);
  if (image == null) {
    throw const UserFacingError('Formato immagine non supportato. Usa una foto JPG o PNG.');
  }
  return image;
}

/// Glue between the UI, the pixel pipeline, ONNX Runtime and FFmpeg.
class SwapEngine {
  static const double maxVideoSeconds = 10;

  final ModelManager manager;
  final OrtFaceModels _models;
  late final FacePipeline _pipeline = FacePipeline(_models);
  Float32List? _emap;

  SwapEngine(this.manager) : _models = OrtFaceModels(manager);

  Future<Directory> _workDir() async {
    final tmp = await getTemporaryDirectory();
    final dir = Directory('${tmp.path}/dfacelab');
    await dir.create(recursive: true);
    return dir;
  }

  Future<SourceFace> prepareSource(String path, ProgressCallback progress) async {
    progress('Leggo la foto…', null);
    final image = await _decodeFile(path);
    progress('Cerco il volto…', null);
    final faces = await _pipeline.detect(image);
    final face = largestFace(faces);
    if (face == null) {
      throw const UserFacingError('Nessun volto trovato in questa foto. Prova con un primo piano ben illuminato.');
    }
    progress('Analizzo il volto…', null);
    final embedding = await _pipeline.embedding(image, face);
    await _models.releaseRecognizer();
    progress('Preparo il modello di scambio…', null);
    _emap ??= await manager.loadEmap();
    final latent = FacePipeline.latentFrom(embedding, _emap!);
    final preview = await compute(_previewPng, (image, face.box));
    return SourceFace(latent, preview, faces.length);
  }

  Future<SwapResult> swapPhoto(String path, SourceFace source, FaceSelection selection, ProgressCallback progress) async {
    progress('Leggo la foto…', null);
    final image = await _decodeFile(path);
    progress('Carico il modello (la prima volta richiede qualche secondo)…', null);
    await _models.warmUpSwapper();
    progress('Sostituisco i volti…', null);
    final count = await _pipeline.swapImage(image, source.latent, selection);
    if (count == 0) {
      throw const UserFacingError('Nessun volto trovato nella foto di destinazione.');
    }
    progress('Salvo il risultato…', null);
    final dir = await _workDir();
    final out = '${dir.path}/dfacelab_${DateTime.now().millisecondsSinceEpoch}.jpg';
    final bytes = await compute(_encodeJpg, (image, 95));
    await File(out).writeAsBytes(bytes, flush: true);
    return SwapResult(path: out, preview: bytes, isVideo: false, facesSwapped: count);
  }

  Future<SwapResult> swapVideo(
    String path,
    SourceFace source,
    FaceSelection selection,
    ProgressCallback progress,
    bool Function() isCancelled,
  ) async {
    progress('Analizzo il video…', null);
    String? frameRate;
    double? duration;
    try {
      final probe = await FFprobeKit.getMediaInformation(path);
      final info = probe.getMediaInformation();
      duration = double.tryParse(info?.getDuration() ?? '');
      for (final s in info?.getStreams() ?? const []) {
        if (s.getType() == 'video') {
          frameRate = parseFrameRate(s.getAverageFrameRate()) != null ? s.getAverageFrameRate() : s.getRealFrameRate();
          break;
        }
      }
    } catch (_) {
      // Fall back to defaults below.
    }
    final plan = VideoPlan.from(frameRate: frameRate, durationSeconds: duration, maxSeconds: maxVideoSeconds);

    final work = await _workDir();
    final frames = Directory('${work.path}/frames');
    if (frames.existsSync()) await frames.delete(recursive: true);
    await frames.create(recursive: true);

    try {
      progress('Estraggo i fotogrammi…', null);
      final extract = await FFmpegKit.executeWithArguments(plan.extractArgs(path, frames.path));
      if (!ReturnCode.isSuccess(await extract.getReturnCode())) {
        throw UserFacingError('Non riesco a leggere questo video.\n${await _tail(extract.getAllLogsAsString())}');
      }
      final files = frames.listSync().whereType<File>().where((f) => f.path.endsWith('.jpg')).toList()
        ..sort((a, b) => a.path.compareTo(b.path));
      if (files.isEmpty) throw const UserFacingError('Il video non contiene fotogrammi leggibili.');

      progress('Carico il modello (la prima volta richiede qualche secondo)…', 0);
      await _models.warmUpSwapper();

      var withFaces = 0;
      var swapped = 0;
      Uint8List? preview;
      final middle = files.length ~/ 2;
      for (var i = 0; i < files.length; i++) {
        if (isCancelled()) throw const ProcessingCancelled();
        progress('Fotogramma ${i + 1} di ${files.length}', i / files.length);
        final framePath = files[i].path;
        final image = await _decodeFile(framePath);
        final count = await _pipeline.swapImage(image, source.latent, selection);
        if (count > 0) {
          withFaces++;
          swapped += count;
          final bytes = await compute(_encodeJpgToFile, (image, framePath, 92));
          if (preview == null || i <= middle) preview = bytes;
        } else if (preview == null && i >= middle) {
          preview = await files[i].readAsBytes();
        }
      }
      if (withFaces == 0) {
        throw const UserFacingError('Nessun volto trovato nel video.');
      }

      if (isCancelled()) throw const ProcessingCancelled();
      progress('Creo il video finale…', 1);
      final out = '${work.path}/dfacelab_${DateTime.now().millisecondsSinceEpoch}.mp4';
      final assemble = await FFmpegKit.executeWithArguments(plan.assembleArgs(frames.path, path, out));
      if (!ReturnCode.isSuccess(await assemble.getReturnCode())) {
        throw UserFacingError('Non riesco a creare il video.\n${await _tail(assemble.getAllLogsAsString())}');
      }
      return SwapResult(
        path: out,
        preview: preview ?? await files[middle].readAsBytes(),
        isVideo: true,
        facesSwapped: swapped,
        framesTotal: files.length,
        framesWithFaces: withFaces,
        truncated: plan.truncated,
        seconds: plan.seconds,
      );
    } finally {
      try {
        if (frames.existsSync()) await frames.delete(recursive: true);
      } catch (_) {}
    }
  }

  static Future<String> _tail(Future<String?> logs) async {
    final text = (await logs) ?? '';
    final lines = text.trim().split('\n');
    return lines.length > 4 ? lines.sublist(lines.length - 4).join('\n') : lines.join('\n');
  }

  Future<void> dispose() => _models.dispose();
}
