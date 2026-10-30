import 'dart:math' as math;
import 'dart:typed_data';

import 'affine.dart';
import 'mask.dart';
import 'paste.dart';
import 'rgb_image.dart';
import 'scrfd.dart';
import 'warp.dart';

/// Which face-swap network is used.
enum SwapModel {
  /// FaceFusion HyperSwap 1c, 256 px: keeps more of the source identity.
  hyperswap,

  /// InsightFace inswapper, 128 px (optionally "pixel boosted" to 256).
  inswapper,
}

/// The neural networks, behind an interface so the pixel pipeline can be
/// tested without ONNX Runtime.
abstract class FaceModels {
  /// SCRFD detector: CHW tensor of [size]x[size] -> its nine outputs.
  Future<List<List<double>>> detect(Float32List input, int size);

  /// ArcFace: CHW 112x112 tensor -> 512-d embedding.
  Future<List<double>> embed(Float32List input);

  /// inswapper: CHW 128x128 target crop (0..1) + 512-d latent -> CHW 128x128 (0..1).
  Future<List<double>> swap(Float32List target, Float32List latent);

  /// HyperSwap: CHW 256x256 target crop (-1..1) + normalised embedding ->
  /// (CHW 256x256 image in -1..1, 256x256 face mask).
  Future<(List<double>, List<double>)> hyperSwap(Float32List target, Float32List source);
}

/// Which faces of the target get replaced.
enum FaceSelection { all, largest }

class FacePipeline {
  static const int inswapperSize = 128;
  static const int hyperswapSize = 256;
  static const int embedSize = 112;

  final FaceModels models;
  final SwapModel model;
  final int detectSize;
  final double scoreThreshold;
  final Map<int, Float32List> _masks = {};

  FacePipeline(this.models, {this.model = SwapModel.hyperswap, this.detectSize = 640, this.scoreThreshold = 0.5});

  Float32List _mask(int size) => _masks.putIfAbsent(size, () => boxMask(size));

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

  /// The vector the swap network expects for a source embedding: the
  /// normalised embedding for HyperSwap, embedding x emap for inswapper.
  static Float32List sourceVector(SwapModel model, List<double> embedding, {List<double>? emap}) {
    if (model == SwapModel.hyperswap) return normalize(embedding);
    if (emap == null) throw ArgumentError('inswapper needs the emap matrix');
    return latentFrom(embedding, emap);
  }

  /// Average of several embeddings (several photos of the same person).
  static List<double> averageEmbedding(List<List<double>> embeddings) {
    if (embeddings.isEmpty) throw ArgumentError('No embeddings');
    final out = Float64List(embeddings.first.length);
    for (final e in embeddings) {
      for (var i = 0; i < out.length; i++) {
        out[i] += e[i];
      }
    }
    return [for (final v in out) v / embeddings.length];
  }

  static Float32List normalize(List<double> v) {
    var n = 0.0;
    for (final x in v) {
      n += x * x;
    }
    n = math.sqrt(n);
    if (n == 0) throw ArgumentError('Empty face embedding');
    return Float32List.fromList([for (final x in v) x / n]);
  }

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

  /// Replaces one face of [image] in place. [boost] (inswapper only) runs the
  /// 128 px model on a 128*boost crop split into boost^2 interleaved tiles.
  Future<void> swapFace(RgbImage image, DetectedFace face, Float32List source, {int boost = 1}) async {
    if (model == SwapModel.hyperswap) {
      const size = hyperswapSize;
      final m = Affine.similarity(face.landmarks, FaceTemplates.scaled(FaceTemplates.arcface128, size));
      final crop = warpCrop(image, m, size);
      final input = toChw(crop, size, scale: 1 / 127.5, offset: -1);
      final (output, faceMask) = await models.hyperSwap(input, source);
      final hwc = chwSignedToHwc(output, size);
      final box = _mask(size);
      final mask = Float32List(size * size);
      for (var i = 0; i < mask.length; i++) {
        var f = faceMask[i];
        if (f < 0) f = 0;
        if (f > 1) f = 1;
        mask[i] = math.min(box[i], f);
      }
      pasteBack(image, hwc, mask, m, size);
      return;
    }
    final size = inswapperSize * boost;
    final m = Affine.similarity(face.landmarks, FaceTemplates.scaled(FaceTemplates.arcface128, size));
    final crop = warpCrop(image, m, size);
    final tiles = implodePixelBoost(crop, inswapperSize, boost);
    final outTiles = <Float32List>[];
    for (final tile in tiles) {
      final input = toChw(tile, inswapperSize, scale: 1 / 255.0, offset: 0);
      final output = await models.swap(input, source);
      outTiles.add(chwUnitToHwc(output, inswapperSize));
    }
    final hwc = explodePixelBoost(outTiles, inswapperSize, boost);
    pasteBack(image, hwc, _mask(size), m, size);
  }

  /// Detects faces in [image] and replaces the selected ones in place.
  /// Returns how many faces were replaced.
  Future<int> swapImage(RgbImage image, Float32List source, FaceSelection selection, {int boost = 1}) async {
    final faces = await detect(image);
    final chosen = selectFaces(faces, selection);
    for (final f in chosen) {
      await swapFace(image, f, source, boost: boost);
    }
    return chosen.length;
  }
}

List<DetectedFace> selectFaces(List<DetectedFace> faces, FaceSelection selection) {
  if (selection == FaceSelection.all) return faces;
  final big = largestFace(faces);
  return big == null ? const [] : [big];
}
