import 'dart:math' as math;
import 'dart:typed_data';

import 'rgb_image.dart';

/// A detected face: bounding box (x1, y1, x2, y2), confidence and five
/// landmarks as flat x,y pairs (eyes, nose, mouth corners).
class DetectedFace {
  final List<double> box;
  final double score;
  final List<double> landmarks;

  const DetectedFace(this.box, this.score, this.landmarks);

  double get width => box[2] - box[0];
  double get height => box[3] - box[1];
  double get area => math.max(0, width) * math.max(0, height);
}

class DetectInput {
  final Float32List tensor;
  final int size;
  final double ratioWidth;
  final double ratioHeight;

  const DetectInput(this.tensor, this.size, this.ratioWidth, this.ratioHeight);
}

/// Builds the SCRFD input: the image shrunk to fit [size]x[size] (never
/// enlarged), placed top-left, BGR planes normalised as (v - 127.5) / 128.
DetectInput prepareDetectInput(RgbImage image, int size) {
  final w = image.width, h = image.height;
  final scale = math.min(1.0, math.min(size / h, size / w));
  final nw = math.max(1, (w * scale).floor());
  final nh = math.max(1, (h * scale).floor());
  final plane = size * size;
  final tensor = Float32List(plane * 3)..fillRange(0, plane * 3, (0 - 127.5) / 128.0);
  final px = List<double>.filled(3, 0);
  final src = image.data;
  final resize = scale < 1;
  for (var y = 0; y < nh; y++) {
    final sy = resize ? (y + 0.5) * h / nh - 0.5 : y.toDouble();
    for (var x = 0; x < nw; x++) {
      double r, g, b;
      if (resize) {
        final sx = (x + 0.5) * w / nw - 0.5;
        sampleBilinear(src, w, h, 3, sx, sy, px);
        r = px[0];
        g = px[1];
        b = px[2];
      } else {
        final k = (y * w + x) * 3;
        r = src[k].toDouble();
        g = src[k + 1].toDouble();
        b = src[k + 2].toDouble();
      }
      final i = y * size + x;
      tensor[i] = (b - 127.5) / 128.0;
      tensor[plane + i] = (g - 127.5) / 128.0;
      tensor[2 * plane + i] = (r - 127.5) / 128.0;
    }
  }
  return DetectInput(tensor, size, w / nw, h / nh);
}

/// Decodes the nine SCRFD outputs (scores, boxes, landmarks for strides
/// 8/16/32) into faces in original image coordinates, best first.
List<DetectedFace> decodeScrfd(
  List<List<double>> outputs,
  DetectInput input, {
  double scoreThreshold = 0.5,
  double nmsThreshold = 0.4,
}) {
  if (outputs.length < 9) throw ArgumentError('SCRFD needs 9 outputs, got ${outputs.length}');
  const strides = [8, 16, 32];
  final rw = input.ratioWidth, rh = input.ratioHeight;
  final candidates = <DetectedFace>[];
  for (var s = 0; s < 3; s++) {
    final stride = strides[s];
    final scores = outputs[s];
    final boxes = outputs[s + 3];
    final kps = outputs[s + 6];
    final grid = input.size ~/ stride;
    for (var k = 0; k < scores.length; k++) {
      final score = scores[k];
      if (score < scoreThreshold) continue;
      final cell = k ~/ 2;
      final cx = ((cell % grid) * stride).toDouble();
      final cy = ((cell ~/ grid) * stride).toDouble();
      final box = [
        (cx - boxes[k * 4] * stride) * rw,
        (cy - boxes[k * 4 + 1] * stride) * rh,
        (cx + boxes[k * 4 + 2] * stride) * rw,
        (cy + boxes[k * 4 + 3] * stride) * rh,
      ];
      final points = <double>[];
      for (var j = 0; j < 5; j++) {
        points.add((cx + kps[k * 10 + 2 * j] * stride) * rw);
        points.add((cy + kps[k * 10 + 2 * j + 1] * stride) * rh);
      }
      candidates.add(DetectedFace(box, score.toDouble(), points));
    }
  }
  candidates.sort((a, b) => b.score.compareTo(a.score));
  final kept = <DetectedFace>[];
  for (final c in candidates) {
    var keep = true;
    for (final k in kept) {
      if (_iou(c.box, k.box) > nmsThreshold) {
        keep = false;
        break;
      }
    }
    if (keep) kept.add(c);
  }
  return kept;
}

double _iou(List<double> a, List<double> b) {
  final iw = math.max(0.0, math.min(a[2], b[2]) - math.max(a[0], b[0]));
  final ih = math.max(0.0, math.min(a[3], b[3]) - math.max(a[1], b[1]));
  final inter = iw * ih;
  final union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter;
  return union > 0 ? inter / union : 0;
}

/// The face with the largest bounding box, or null.
DetectedFace? largestFace(List<DetectedFace> faces) {
  DetectedFace? best;
  for (final f in faces) {
    if (best == null || f.area > best.area) best = f;
  }
  return best;
}
