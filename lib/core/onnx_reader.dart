import 'dart:io';
import 'dart:typed_data';

/// Float tensor read from an ONNX file.
class OnnxTensor {
  final List<int> dims;
  final Float32List data;

  const OnnxTensor(this.dims, this.data);
}

/// Reads the last initializer of an ONNX model's graph without loading the
/// whole file (the inswapper "emap" matrix lives there). Only the small
/// protobuf headers are parsed; large fields are skipped with seeks.
OnnxTensor readLastInitializer(String path) {
  final file = File(path).openSync();
  try {
    final r = _Reader(file);
    final end = file.lengthSync();
    int? graphStart, graphEnd;
    while (r.position < end) {
      final key = r.varint();
      final field = key >> 3, wire = key & 7;
      if (field == 7 && wire == 2) {
        final len = r.varint();
        graphStart = r.position;
        graphEnd = graphStart + len;
        r.seek(graphEnd);
      } else {
        r.skip(wire);
      }
    }
    if (graphStart == null) throw const FormatException('ONNX graph not found');
    r.seek(graphStart);
    int? initStart, initEnd;
    while (r.position < graphEnd!) {
      final key = r.varint();
      final field = key >> 3, wire = key & 7;
      if (field == 5 && wire == 2) {
        final len = r.varint();
        initStart = r.position;
        initEnd = initStart + len;
        r.seek(initEnd);
      } else {
        r.skip(wire);
      }
    }
    if (initStart == null) throw const FormatException('ONNX initializer not found');
    r.seek(initStart);
    final dims = <int>[];
    int? dataType;
    Uint8List? raw;
    Uint8List? floatData;
    while (r.position < initEnd!) {
      final key = r.varint();
      final field = key >> 3, wire = key & 7;
      if (field == 1 && wire == 0) {
        dims.add(r.varint());
      } else if (field == 1 && wire == 2) {
        final len = r.varint();
        final stop = r.position + len;
        while (r.position < stop) {
          dims.add(r.varint());
        }
      } else if (field == 2 && wire == 0) {
        dataType = r.varint();
      } else if (field == 9 && wire == 2) {
        raw = r.bytes(r.varint());
      } else if (field == 4 && wire == 2) {
        floatData = r.bytes(r.varint());
      } else {
        r.skip(wire);
      }
    }
    if (dataType != 1) throw FormatException('Initializer is not float32 (type $dataType)');
    final bytes = raw ?? floatData;
    if (bytes == null) throw const FormatException('Initializer has no inline data');
    final count = dims.fold<int>(1, (a, b) => a * b);
    if (bytes.length != count * 4) {
      throw FormatException('Initializer size ${bytes.length} != ${count * 4}');
    }
    final view = ByteData.sublistView(bytes);
    final data = Float32List(count);
    for (var i = 0; i < count; i++) {
      data[i] = view.getFloat32(i * 4, Endian.little);
    }
    return OnnxTensor(dims, data);
  } finally {
    file.closeSync();
  }
}

class _Reader {
  final RandomAccessFile file;
  final Uint8List _buf = Uint8List(1 << 16);
  int _bufStart = 0;
  int _bufLen = 0;
  int position = 0;

  _Reader(this.file);

  void seek(int pos) => position = pos;

  int byte() {
    if (position < _bufStart || position >= _bufStart + _bufLen) {
      file.setPositionSync(position);
      _bufStart = position;
      _bufLen = file.readIntoSync(_buf);
      if (_bufLen <= 0) throw const FormatException('Unexpected end of ONNX file');
    }
    return _buf[position++ - _bufStart];
  }

  int varint() {
    var result = 0;
    var shift = 0;
    while (true) {
      final b = byte();
      result |= (b & 0x7F) << shift;
      if (b < 0x80) return result;
      shift += 7;
      if (shift > 63) throw const FormatException('Bad varint');
    }
  }

  Uint8List bytes(int n) {
    file.setPositionSync(position);
    final out = file.readSync(n);
    if (out.length != n) throw const FormatException('Unexpected end of ONNX file');
    position += n;
    return out;
  }

  void skip(int wire) {
    switch (wire) {
      case 0:
        varint();
      case 1:
        position += 8;
      case 2:
        final n = varint();
        position += n;
      case 5:
        position += 4;
      default:
        throw FormatException('Unsupported protobuf wire type $wire');
    }
  }
}
