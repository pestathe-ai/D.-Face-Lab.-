import 'dart:math' as math;

/// 2D affine transform: x' = a*x + b*y + c, y' = d*x + e*y + f.
class Affine {
  final double a, b, c, d, e, f;

  const Affine(this.a, this.b, this.c, this.d, this.e, this.f);

  double mapX(double x, double y) => a * x + b * y + c;
  double mapY(double x, double y) => d * x + e * y + f;

  double get determinant => a * e - b * d;

  /// Linear scale factor (output pixels per input pixel).
  double get scale => math.sqrt(determinant.abs());

  Affine inverse() {
    final det = determinant;
    if (det == 0) throw StateError('Affine transform is not invertible');
    return Affine(
      e / det,
      -b / det,
      (b * f - c * e) / det,
      -d / det,
      a / det,
      (c * d - a * f) / det,
    );
  }

  /// Least-squares similarity transform (rotation, uniform scale, translation)
  /// mapping [src] points onto [dst] points. Both lists are flat x,y pairs.
  static Affine similarity(List<double> src, List<double> dst) {
    final n = src.length ~/ 2;
    if (n < 2 || dst.length != src.length) {
      throw ArgumentError('Need at least two matching point pairs');
    }
    var msx = 0.0, msy = 0.0, mdx = 0.0, mdy = 0.0;
    for (var i = 0; i < n; i++) {
      msx += src[2 * i];
      msy += src[2 * i + 1];
      mdx += dst[2 * i];
      mdy += dst[2 * i + 1];
    }
    msx /= n;
    msy /= n;
    mdx /= n;
    mdy /= n;
    var variance = 0.0, sxx = 0.0, sxy = 0.0;
    for (var i = 0; i < n; i++) {
      final sx = src[2 * i] - msx;
      final sy = src[2 * i + 1] - msy;
      final dx = dst[2 * i] - mdx;
      final dy = dst[2 * i + 1] - mdy;
      variance += sx * sx + sy * sy;
      sxx += sx * dx + sy * dy;
      sxy += sx * dy - sy * dx;
    }
    if (variance == 0) throw ArgumentError('Degenerate source points');
    final aa = sxx / variance;
    final bb = sxy / variance;
    return Affine(
      aa,
      -bb,
      mdx - (aa * msx - bb * msy),
      bb,
      aa,
      mdy - (bb * msx + aa * msy),
    );
  }

  @override
  String toString() => 'Affine([$a, $b, $c], [$d, $e, $f])';
}

/// Five-point face templates (normalised 0..1), same as FaceFusion.
/// Order: left eye, right eye, nose, left mouth corner, right mouth corner
/// (left/right as seen in the image).
class FaceTemplates {
  static const List<double> arcface112v2 = [
    0.34191607, 0.46157411, //
    0.65653393, 0.45983393, //
    0.50022500, 0.64050536, //
    0.37097589, 0.82469196, //
    0.63151696, 0.82325089, //
  ];

  static const List<double> arcface128 = [
    0.36167656, 0.40387734, //
    0.63696719, 0.40235469, //
    0.50019687, 0.56044219, //
    0.38710391, 0.72160547, //
    0.61507734, 0.72034453, //
  ];

  static List<double> scaled(List<double> template, int size) => [for (final v in template) v * size];
}
