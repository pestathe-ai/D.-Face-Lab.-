// Replays the real model inputs/outputs recorded by tool/fixtures/gen_fixtures.py
// (numpy twin of lib/core) and checks that the Dart pixel pipeline produces the
// same tensors and the same final image, without needing ONNX Runtime.
import 'dart:convert';
import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:d_face_lab/core/codec.dart';
import 'package:d_face_lab/core/mask.dart';
import 'package:d_face_lab/core/pipeline.dart';
import 'package:d_face_lab/core/rgb_image.dart';
import 'package:d_face_lab/core/scrfd.dart';
import 'package:flutter_test/flutter_test.dart';

class Fixtures {
  final Map<String, dynamic> manifest;
  final ByteData bin;

  Fixtures._(this.manifest, this.bin);

  factory Fixtures.load() {
    final manifest = jsonDecode(File('test/fixtures/replay.json').readAsStringSync()) as Map<String, dynamic>;
    final bytes = File('test/fixtures/replay.bin').readAsBytesSync();
    return Fixtures._(manifest, ByteData.sublistView(bytes));
  }

  Float32List floats(String name) {
    final entry = manifest[name] as Map<String, dynamic>;
    final offset = entry['offset'] as int;
    final length = entry['length'] as int;
    final out = Float32List(length);
    for (var i = 0; i < length; i++) {
      out[i] = bin.getFloat32(offset + i * 4, Endian.little);
    }
    return out;
  }
}

class Diff {
  final double max;
  final double mean;
  Diff(this.max, this.mean);
  @override
  String toString() => 'max=$max mean=$mean';
}

Diff diff(List<num> a, List<num> b) {
  expect(a.length, b.length, reason: 'length mismatch');
  var m = 0.0, s = 0.0;
  for (var i = 0; i < a.length; i++) {
    final d = (a[i] - b[i]).abs().toDouble();
    m = math.max(m, d);
    s += d;
  }
  return Diff(m, s / a.length);
}

class ReplayModels implements FaceModels {
  final Fixtures fx;
  final List<String> detectOrder = ['source', 'target'];
  int detectCalls = 0;
  int swapCalls = 0;
  final List<String> log = [];

  ReplayModels(this.fx);

  @override
  Future<List<List<double>>> detect(Float32List input, int size) async {
    final name = detectOrder[detectCalls++];
    final samples = fx.floats('$name.det.input.samples');
    final mine = [for (var i = 0; i < input.length; i += 997) input[i]];
    final d = diff(mine, samples);
    log.add('detect $name input $d');
    expect(d.max, lessThan(1e-3), reason: 'detector input ($name) differs: $d');
    var sum = 0.0;
    for (final v in input) {
      sum += v;
    }
    final expected = (fx.manifest['$name.det.input.sum'] as num).toDouble();
    expect((sum - expected).abs() / expected.abs(), lessThan(1e-5), reason: 'detector input sum ($name)');
    return [for (var i = 0; i < 9; i++) fx.floats('$name.det.out.$i')];
  }

  @override
  Future<List<double>> embed(Float32List input) async {
    final d = diff(input, fx.floats('embed.input'));
    log.add('embed input $d');
    expect(d.max, lessThan(1e-3), reason: 'ArcFace input differs: $d');
    return fx.floats('embed.out');
  }

  @override
  Future<List<double>> swap(Float32List target, Float32List latent) async {
    final k = swapCalls++;
    final d = diff(target, fx.floats('swap.$k.input'));
    log.add('swap $k input $d');
    expect(d.max, lessThan(0.02), reason: 'swapper input $k differs: $d');
    expect(d.mean, lessThan(1e-4), reason: 'swapper input $k differs: $d');
    expect(diff(latent, fx.floats('latent')).max, lessThan(1e-6));
    return fx.floats('swap.$k.out');
  }
}

void checkFaces(List<DetectedFace> faces, List<dynamic> expected) {
  expect(faces.length, expected.length);
  for (var i = 0; i < faces.length; i++) {
    final e = expected[i] as Map<String, dynamic>;
    expect(faces[i].score, closeTo((e['score'] as num).toDouble(), 1e-6));
    final box = (e['box'] as List).cast<num>();
    final kps = (e['kps'] as List).cast<num>();
    expect(diff(faces[i].box, box).max, lessThan(1e-3), reason: 'box $i');
    expect(diff(faces[i].landmarks, kps).max, lessThan(1e-3), reason: 'landmarks $i');
  }
}

RgbImage loadPng(String path) => decodeToRgb(File(path).readAsBytesSync())!;

void main() {
  test('box mask matches the reference', () {
    final fx = Fixtures.load();
    final d = diff(boxMask(128), fx.floats('mask'));
    expect(d.max, lessThan(1e-5), reason: '$d');
  });

  test('full pipeline reproduces the reference swap', () async {
    final fx = Fixtures.load();
    final models = ReplayModels(fx);
    final pipeline = FacePipeline(models, detectSize: 320);

    final source = loadPng('test/fixtures/source.png');
    final sourceFaces = await pipeline.detect(source);
    checkFaces(sourceFaces, fx.manifest['source.faces'] as List);
    final face = largestFace(sourceFaces)!;
    final embedding = await pipeline.embedding(source, face);
    expect(embedding.length, 512);

    final target = loadPng('test/fixtures/target.png');
    final faces = await pipeline.detect(target);
    checkFaces(faces, fx.manifest['target.faces'] as List);

    final latent = fx.floats('latent');
    for (final f in selectFaces(faces, FaceSelection.all)) {
      await pipeline.swapFace(target, f, latent);
    }
    expect(models.swapCalls, 4);

    final expected = loadPng('test/fixtures/expected.png');
    expect(target.width, expected.width);
    expect(target.height, expected.height);
    final d = diff(target.data, expected.data);
    // ignore: avoid_print
    print([...models.log, 'final image $d'].join('\n'));
    expect(d.max, lessThanOrEqualTo(3), reason: 'final image differs: $d');
    expect(d.mean, lessThan(0.01), reason: 'final image differs: $d');
  });

  test('selectFaces keeps only the largest face when asked', () {
    const small = DetectedFace([0, 0, 10, 10], 0.9, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
    const big = DetectedFace([0, 0, 50, 40], 0.6, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
    expect(selectFaces([small, big], FaceSelection.largest), [big]);
    expect(selectFaces([small, big], FaceSelection.all), [small, big]);
    expect(selectFaces([], FaceSelection.largest), isEmpty);
  });

  test('latentFrom multiplies by emap and divides by the norm', () {
    final emb = [3.0, 4.0];
    final emap = [1.0, 2.0, 3.0, 4.0]; // [[1,2],[3,4]]
    final l = FacePipeline.latentFrom(emb, emap);
    expect(l[0], closeTo((3 * 1 + 4 * 3) / 5, 1e-6));
    expect(l[1], closeTo((3 * 2 + 4 * 4) / 5, 1e-6));
  });
}
