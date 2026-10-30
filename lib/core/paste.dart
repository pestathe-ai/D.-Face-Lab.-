import 'dart:math' as math;

import 'affine.dart';
import 'rgb_image.dart';

/// Blends a processed [size]x[size] crop (interleaved RGB floats 0..255)
/// back into [image] in place. [toCrop] is the transform used to cut the
/// crop; [mask] (size*size, 0..1) controls the blend.
void pasteBack(RgbImage image, List<double> crop, List<double> mask, Affine toCrop, int size) {
  final inv = toCrop.inverse();
  final corners = [0.0, 0.0, size.toDouble(), 0.0, size.toDouble(), size.toDouble(), 0.0, size.toDouble()];
  var minX = double.infinity, minY = double.infinity;
  var maxX = -double.infinity, maxY = -double.infinity;
  for (var i = 0; i < 4; i++) {
    final x = inv.mapX(corners[2 * i], corners[2 * i + 1]);
    final y = inv.mapY(corners[2 * i], corners[2 * i + 1]);
    minX = math.min(minX, x);
    minY = math.min(minY, y);
    maxX = math.max(maxX, x);
    maxY = math.max(maxY, y);
  }
  final w = image.width, h = image.height;
  final x1 = minX.floor().clamp(0, w).toInt();
  final x2 = maxX.ceil().clamp(0, w).toInt();
  final y1 = minY.floor().clamp(0, h).toInt();
  final y2 = maxY.ceil().clamp(0, h).toInt();
  final limit = (size - 1).toDouble();
  final alpha = List<double>.filled(1, 0);
  final color = List<double>.filled(3, 0);
  final data = image.data;
  for (var y = y1; y < y2; y++) {
    for (var x = x1; x < x2; x++) {
      final u = toCrop.mapX(x.toDouble(), y.toDouble());
      final v = toCrop.mapY(x.toDouble(), y.toDouble());
      if (u < 0 || v < 0 || u > limit || v > limit) continue;
      sampleBilinear(mask, size, size, 1, u, v, alpha);
      final a = alpha[0];
      if (a <= 0) continue;
      sampleBilinear(crop, size, size, 3, u, v, color);
      final k = (y * w + x) * 3;
      for (var c = 0; c < 3; c++) {
        final blended = data[k + c] * (1 - a) + color[c] * a;
        data[k + c] = (blended + 0.5).floor().clamp(0, 255).toInt();
      }
    }
  }
}
