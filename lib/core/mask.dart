import 'dart:math' as math;
import 'dart:typed_data';

/// Soft-edged square mask (FaceFusion "box" mask): ones with a zero border,
/// then Gaussian-blurred. [blur] 0..1, FaceFusion default 0.3.
Float32List boxMask(int size, {double blur = 0.3}) {
  final amount = (size * 0.5 * blur).floor();
  final area = math.max(amount ~/ 2, 1);
  final mask = Float64List(size * size);
  for (var y = 0; y < size; y++) {
    for (var x = 0; x < size; x++) {
      final inside = y >= area && y < size - area && x >= area && x < size - area;
      mask[y * size + x] = inside ? 1 : 0;
    }
  }
  if (amount <= 0) return Float32List.fromList(mask);
  final sigma = amount * 0.25;
  final radius = (sigma * 3).ceil();
  final kernel = Float64List(2 * radius + 1);
  var sum = 0.0;
  for (var i = -radius; i <= radius; i++) {
    final v = math.exp(-(i * i) / (2 * sigma * sigma));
    kernel[i + radius] = v;
    sum += v;
  }
  for (var i = 0; i < kernel.length; i++) {
    kernel[i] /= sum;
  }
  int reflect(int i) {
    if (i < 0) return -i;
    if (i >= size) return 2 * size - 2 - i;
    return i;
  }

  final tmp = Float64List(size * size);
  for (var y = 0; y < size; y++) {
    for (var x = 0; x < size; x++) {
      var acc = 0.0;
      for (var k = -radius; k <= radius; k++) {
        acc += kernel[k + radius] * mask[y * size + reflect(x + k)];
      }
      tmp[y * size + x] = acc;
    }
  }
  final out = Float32List(size * size);
  for (var y = 0; y < size; y++) {
    for (var x = 0; x < size; x++) {
      var acc = 0.0;
      for (var k = -radius; k <= radius; k++) {
        acc += kernel[k + radius] * tmp[reflect(y + k) * size + x];
      }
      out[y * size + x] = acc;
    }
  }
  return out;
}
