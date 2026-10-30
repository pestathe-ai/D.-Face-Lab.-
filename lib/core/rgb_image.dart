import 'dart:typed_data';

/// 8-bit RGB image, pixels stored row by row as R, G, B.
class RgbImage {
  final int width;
  final int height;
  final Uint8List data;

  RgbImage(this.width, this.height, this.data) {
    if (data.length != width * height * 3) {
      throw ArgumentError('RGB buffer size ${data.length} != ${width}x${height}x3');
    }
  }

  factory RgbImage.blank(int width, int height) => RgbImage(width, height, Uint8List(width * height * 3));

  RgbImage copy() => RgbImage(width, height, Uint8List.fromList(data));
}

/// Bilinear sample of an interleaved image with clamped (replicated) borders.
/// [channels] values per pixel, writes the result into [out] at [outOffset].
void sampleBilinear(
  List<num> src,
  int width,
  int height,
  int channels,
  double x,
  double y,
  List<double> out,
) {
  if (x < 0) x = 0;
  if (y < 0) y = 0;
  final maxX = (width - 1).toDouble();
  final maxY = (height - 1).toDouble();
  if (x > maxX) x = maxX;
  if (y > maxY) y = maxY;
  final x0 = x.floor();
  final y0 = y.floor();
  final x1 = x0 + 1 < width ? x0 + 1 : width - 1;
  final y1 = y0 + 1 < height ? y0 + 1 : height - 1;
  final fx = x - x0;
  final fy = y - y0;
  final i00 = (y0 * width + x0) * channels;
  final i01 = (y0 * width + x1) * channels;
  final i10 = (y1 * width + x0) * channels;
  final i11 = (y1 * width + x1) * channels;
  for (var c = 0; c < channels; c++) {
    final top = src[i00 + c].toDouble() * (1 - fx) + src[i01 + c].toDouble() * fx;
    final bottom = src[i10 + c].toDouble() * (1 - fx) + src[i11 + c].toDouble() * fx;
    out[c] = top * (1 - fy) + bottom * fy;
  }
}
