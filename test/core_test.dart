import 'dart:convert';
import 'dart:io';

import 'package:d_face_lab/core/affine.dart';
import 'package:d_face_lab/core/codec.dart';
import 'package:d_face_lab/core/onnx_reader.dart';
import 'package:d_face_lab/core/rgb_image.dart';
import 'package:d_face_lab/core/video_plan.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  group('Affine', () {
    test('similarity recovers a known rotation + scale + shift', () {
      const t = Affine(0.8, -0.6, 10, 0.6, 0.8, -5); // scale 1, 36.87 deg
      final src = [0.0, 0.0, 10.0, 0.0, 0.0, 10.0, 7.0, 3.0, -4.0, 9.0];
      final dst = <double>[];
      for (var i = 0; i < 5; i++) {
        dst.add(t.mapX(src[2 * i], src[2 * i + 1]));
        dst.add(t.mapY(src[2 * i], src[2 * i + 1]));
      }
      final m = Affine.similarity(src, dst);
      expect(m.a, closeTo(0.8, 1e-9));
      expect(m.b, closeTo(-0.6, 1e-9));
      expect(m.c, closeTo(10, 1e-9));
      expect(m.d, closeTo(0.6, 1e-9));
      expect(m.f, closeTo(-5, 1e-9));
      final inv = m.inverse();
      final x = m.mapX(3, 4), y = m.mapY(3, 4);
      expect(inv.mapX(x, y), closeTo(3, 1e-9));
      expect(inv.mapY(x, y), closeTo(4, 1e-9));
      expect(m.scale, closeTo(1, 1e-9));
    });
  });

  group('ONNX initializer reader', () {
    final expected = [for (var i = 0; i < 12; i++) i * 0.5 - 1];
    for (final name in ['tiny_raw.onnx', 'tiny_floatdata.onnx']) {
      test('reads the last initializer from $name', () {
        final t = readLastInitializer('test/fixtures/$name');
        expect(t.dims, [4, 3]);
        expect(t.data.toList(), expected);
      });
    }
  });

  group('codec', () {
    test('JPEG round trip keeps size and colours', () {
      final im = RgbImage.blank(33, 17);
      for (var i = 0; i < im.data.length; i += 3) {
        im.data[i] = 200;
        im.data[i + 1] = 100;
        im.data[i + 2] = 50;
      }
      final back = decodeToRgb(encodeJpeg(im, quality: 95))!;
      expect(back.width, 33);
      expect(back.height, 17);
      expect((back.data[0] - 200).abs(), lessThan(6));
      expect((back.data[1] - 100).abs(), lessThan(6));
      expect((back.data[2] - 50).abs(), lessThan(6));
    });

    test('PNG fixture decodes as RGB', () {
      final im = decodeToRgb(File('test/fixtures/source.png').readAsBytesSync())!;
      expect(im.width, 112);
      expect(im.height, 112);
    });

    test('cropPreview stays inside the image', () {
      final im = RgbImage.blank(100, 80);
      final p = cropPreview(im, [-20, -20, 150, 120], maxSide: 64);
      expect(p.width, lessThanOrEqualTo(64));
      expect(p.height, lessThanOrEqualTo(64));
    });
  });

  group('VideoPlan', () {
    test('parses frame rates', () {
      expect(parseFrameRate('30000/1001'), closeTo(29.97, 0.01));
      expect(parseFrameRate('25'), 25);
      expect(parseFrameRate('0/0'), isNull);
      expect(parseFrameRate(''), isNull);
      expect(parseFrameRate(null), isNull);
      expect(parseFrameRate('abc'), isNull);
    });

    test('caps fps and duration', () {
      final p = VideoPlan.from(frameRate: '60/1', durationSeconds: 42, maxSeconds: 10);
      expect(p.fps, 30);
      expect(p.seconds, 10);
      expect(p.truncated, isTrue);
      final q = VideoPlan.from(frameRate: null, durationSeconds: 4.2, maxSeconds: 10);
      expect(q.fps, VideoPlan.fallbackFps);
      expect(q.seconds, 4.2);
      expect(q.truncated, isFalse);
    });

    test('writes FFmpeg commands for the CI check', () {
      final plan = VideoPlan.from(frameRate: '30000/1001', durationSeconds: 12, maxSeconds: 10);
      final extract = plan.extractArgs('ci/in.mp4', 'ci/frames');
      final assemble = plan.assembleArgs('ci/frames', 'ci/in.mp4', 'ci/out.mp4');
      expect(extract.last, 'ci/frames/%06d.jpg');
      expect(assemble.last, 'ci/out.mp4');
      final out = Platform.environment['DFL_FFMPEG_JSON'];
      if (out != null) {
        File(out).writeAsStringSync(jsonEncode({
          'extract': extract,
          'assemble': assemble,
          'fps': plan.fps,
          'seconds': plan.seconds,
        }));
      }
    });
  });
}
