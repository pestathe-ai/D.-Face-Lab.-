import 'dart:typed_data';

import 'affine.dart';
import 'rgb_image.dart';

/// Cuts a [size]x[size] crop out of [image]; [toCrop] maps image -> crop
/// coordinates. Returns interleaved RGB floats in 0..255. When the crop
/// shrinks the face a lot, several sub-samples per pixel are averaged to
/// avoid aliasing.
Float32List warpCrop(RgbImage image, Affine toCrop, int size) {
  final inv = toCrop.inverse();
  var n = (1 / toCrop.scale).ceil();
  if (n < 1) n = 1;
  if (n > 4) n = 4;
  final out = Float32List(size * size * 3);
  final px = List<double>.filled(3, 0);
  final w = image.width, h = image.height, src = image.data;
  for (var v = 0; v < size; v++) {
    for (var u = 0; u < size; u++) {
      var r = 0.0, g = 0.0, b = 0.0;
      for (var j = 0; j < n; j++) {
        final vv = v + (j + 0.5) / n - 0.5;
        for (var i = 0; i < n; i++) {
          final uu = u + (i + 0.5) / n - 0.5;
          final x = inv.a * uu + inv.b * vv + inv.c;
          final y = inv.d * uu + inv.e * vv + inv.f;
          sampleBilinear(src, w, h, 3, x, y, px);
          r += px[0];
          g += px[1];
          b += px[2];
        }
      }
      final k = (v * size + u) * 3;
      final nn = n * n;
      out[k] = r / nn;
      out[k + 1] = g / nn;
      out[k + 2] = b / nn;
    }
  }
  return out;
}

/// Interleaved RGB (HWC) -> planar CHW tensor: value * scale + offset.
Float32List toChw(Float32List hwc, int size, {required double scale, required double offset}) {
  final plane = size * size;
  final out = Float32List(plane * 3);
  for (var i = 0; i < plane; i++) {
    out[i] = hwc[i * 3] * scale + offset;
    out[plane + i] = hwc[i * 3 + 1] * scale + offset;
    out[2 * plane + i] = hwc[i * 3 + 2] * scale + offset;
  }
  return out;
}

/// Planar CHW tensor in 0..1 -> interleaved RGB floats 0..255 (clipped).
Float32List chwUnitToHwc(List<double> chw, int size) {
  final plane = size * size;
  final out = Float32List(plane * 3);
  for (var i = 0; i < plane; i++) {
    for (var c = 0; c < 3; c++) {
      var v = chw[c * plane + i];
      if (v < 0) v = 0;
      if (v > 1) v = 1;
      out[i * 3 + c] = v * 255;
    }
  }
  return out;
}
