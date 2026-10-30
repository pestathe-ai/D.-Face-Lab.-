import 'dart:math' as math;
import 'dart:typed_data';

import 'affine.dart';
import 'mask.dart';
import 'paste.dart';
import 'rgb_image.dart';
import 'scrfd.dart';
import 'warp.dart';

/// The three neural networks, behind an interface so the pixel pipeline can
/// be tested without ONNX Runtime.
abstract class FaceModels {
  /// SCRFD detector: CHW tensor of [size]x[size] -> its nine outputs.
  Future<List<List<double>>> detect(Float32List input, int size);

  /// ArcFace: CHW 112x112 tensor -> 512-d embedding.
  Future<List<double>> embed(Float32List input);

  /// inswapper: CHW 128x128 target crop + 512-d latent -> CHW 128x128 (0..1).
  Future<List<double>> swap(Float32List target, Float32List latent);
}

/// Which faces of the target get replaced.
enum FaceSelection { all, largest }

class FacePipeline {
  static const int swapSize = 128;
  static const int embedSize = 112;

  final FaceModels models;
  final int detectSize;
  final double scoreThreshold;
  final Float32List _mask = boxMask(swapSize);

  FacePipeline(this.models, {this.detectSize = 640, this.scoreThreshold = 0.5});

  Future<List<DetectedFace>> detect(RgbImage image) async {
    final input = prepareDetectInput(image, detectSize);
    final outputs = await models.detect(input.tensor, detectSize);
    return decodeScrfd(outputs, input, scoreThreshold: scoreThreshold);
  }

  Float32List embedInput(RgbImage image, DetectedFace face) {
    final m = Affine.similarity(face.landmarks, FaceTemplates.scaled(FaceTemplates.arcface112v2, embedSize));
    final crop = warpCrop(image, m, embedSize);
    return toChw(crop, embedSize, scale: 1 / 127.5, offset: -1);
  }

  Future<List<double>> embedding(RgbImage image, DetectedFace face) => models.embed(embedInput(image, face));

  /// inswapper source vector: embedding x emap, divided by the embedding norm
  /// (as FaceFusion does). [emap] is row-major [dim x dim].
  static Float32List latentFrom(List<double> embedding, List<double> emap) {
    final dim = embedding.length;
    if (emap.length != dim * dim) throw ArgumentError('emap must be $dim x $dim');
    var norm = 0.0;
    for (final v in embedding) {
      norm += v * v;
    }
    norm = math.sqrt(norm);
    if (norm == 0) throw ArgumentError('Empty face embedding');
    final out = Float64List(dim);
    for (var i = 0; i < dim; i++) {
      final e = embedding[i];
      if (e == 0) continue;
      final row = i * dim;
      for (var j = 0; j < dim; j++) {
        out[j] += e * emap[row + j];
      }
    }
    return Float32List.fromList([for (final v in out) v / norm]);
  }

  /// Replaces one face of [image] in place.
  Future<void> swapFace(RgbImage image, DetectedFace face, Float32List latent) async {
    final m = Affine.similarity(face.landmarks, FaceTemplates.scaled(FaceTemplates.arcface128, swapSize));
    final crop = warpCrop(image, m, swapSize);
    final input = toChw(crop, swapSize, scale: 1 / 255.0, offset: 0);
    final output = await models.swap(input, latent);
    final hwc = chwUnitToHwc(output, swapSize);
    pasteBack(image, hwc, _mask, m, swapSize);
  }

  /// Detects faces in [image] and replaces the selected ones in place.
  /// Returns how many faces were replaced.
  Future<int> swapImage(RgbImage image, Float32List latent, FaceSelection selection) async {
    final faces = await detect(image);
    final chosen = selectFaces(faces, selection);
    for (final f in chosen) {
      await swapFace(image, f, latent);
    }
    return chosen.length;
  }
}

List<DetectedFace> selectFaces(List<DetectedFace> faces, FaceSelection selection) {
  if (selection == FaceSelection.all) return faces;
  final big = largestFace(faces);
  return big == null ? const [] : [big];
}
