import 'dart:typed_data';

import 'package:image/image.dart' as img;

import 'rgb_image.dart';

/// Decodes JPEG/PNG/WebP/... bytes into an upright 8-bit RGB image
/// (EXIF rotation applied). Returns null if the format is not supported.
RgbImage? decodeToRgb(Uint8List bytes) {
  final decoded = img.decodeImage(bytes);
  if (decoded == null) return null;
  var im = img.bakeOrientation(decoded);
  if (im.format != img.Format.uint8 || im.hasPalette || im.numChannels != 3) {
    im = im.convert(format: img.Format.uint8, numChannels: 3);
  }
  final rgb = im.getBytes(order: img.ChannelOrder.rgb);
  return RgbImage(im.width, im.height, Uint8List.fromList(rgb));
}

img.Image _toImage(RgbImage image) => img.Image.fromBytes(
      width: image.width,
      height: image.height,
      bytes: image.data.buffer,
      bytesOffset: image.data.offsetInBytes,
      numChannels: 3,
      order: img.ChannelOrder.rgb,
    );

Uint8List encodeJpeg(RgbImage image, {int quality = 95}) => img.encodeJpg(_toImage(image), quality: quality);

Uint8List encodePng(RgbImage image) => img.encodePng(_toImage(image));

/// Crops a square-ish region around [box] (x1,y1,x2,y2) with some margin and
/// scales it to at most [maxSide] pixels, for previews.
RgbImage cropPreview(RgbImage image, List<double> box, {int maxSide = 256, double margin = 0.35}) {
  final bw = box[2] - box[0], bh = box[3] - box[1];
  final x1 = (box[0] - bw * margin).floor().clamp(0, image.width - 1).toInt();
  final y1 = (box[1] - bh * margin).floor().clamp(0, image.height - 1).toInt();
  final x2 = (box[2] + bw * margin).ceil().clamp(x1 + 1, image.width).toInt();
  final y2 = (box[3] + bh * margin).ceil().clamp(y1 + 1, image.height).toInt();
  final cw = x2 - x1, ch = y2 - y1;
  final scale = (maxSide / (cw > ch ? cw : ch)).clamp(0.0, 1.0);
  final ow = (cw * scale).round().clamp(1, cw).toInt();
  final oh = (ch * scale).round().clamp(1, ch).toInt();
  final out = RgbImage.blank(ow, oh);
  for (var y = 0; y < oh; y++) {
    final sy = y1 + (y * ch ~/ oh);
    for (var x = 0; x < ow; x++) {
      final sx = x1 + (x * cw ~/ ow);
      final s = (sy * image.width + sx) * 3;
      final d = (y * ow + x) * 3;
      out.data[d] = image.data[s];
      out.data[d + 1] = image.data[s + 1];
      out.data[d + 2] = image.data[s + 2];
    }
  }
  return out;
}
