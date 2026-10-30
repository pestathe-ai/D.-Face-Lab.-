/// Builds the FFmpeg argument lists used for videos (pure Dart, testable).
class VideoPlan {
  /// Longest processed side in pixels.
  static const int maxSide = 1280;

  /// Highest frame rate kept; faster videos are resampled to this.
  static const double maxFps = 30;

  /// Used when the frame rate cannot be read.
  static const double fallbackFps = 25;

  final double fps;
  final double seconds;
  final bool truncated;

  const VideoPlan(this.fps, this.seconds, this.truncated);

  /// [frameRate] like "30000/1001" or "30"; [durationSeconds] may be null.
  factory VideoPlan.from({String? frameRate, double? durationSeconds, required double maxSeconds}) {
    var fps = parseFrameRate(frameRate) ?? fallbackFps;
    if (fps > maxFps) fps = maxFps;
    final duration = durationSeconds ?? maxSeconds;
    final truncated = duration > maxSeconds + 0.05;
    final seconds = truncated ? maxSeconds : duration;
    return VideoPlan(fps, seconds, truncated);
  }

  String get fpsText => fps.toStringAsFixed(3);

  int get estimatedFrames => (fps * seconds).ceil();

  List<String> extractArgs(String input, String framesDir) => [
        '-y',
        '-t',
        seconds.toStringAsFixed(3),
        '-i',
        input,
        '-an',
        '-vf',
        "fps=$fpsText,scale='if(gte(iw,ih),min($maxSide,iw),-2)':'if(gte(iw,ih),-2,min($maxSide,ih))'",
        '-q:v',
        '2',
        '$framesDir/%06d.jpg',
      ];

  List<String> assembleArgs(String framesDir, String input, String output) => [
        '-y',
        '-framerate',
        fpsText,
        '-i',
        '$framesDir/%06d.jpg',
        '-t',
        seconds.toStringAsFixed(3),
        '-i',
        input,
        '-map',
        '0:v:0',
        '-map',
        '1:a:0?',
        '-vf',
        'pad=ceil(iw/2)*2:ceil(ih/2)*2',
        '-c:v',
        'libx264',
        '-preset',
        'veryfast',
        '-crf',
        '18',
        '-pix_fmt',
        'yuv420p',
        '-c:a',
        'aac',
        '-b:a',
        '160k',
        '-shortest',
        '-movflags',
        '+faststart',
        output,
      ];
}

double? parseFrameRate(String? value) {
  if (value == null) return null;
  final v = value.trim();
  if (v.isEmpty) return null;
  double? result;
  if (v.contains('/')) {
    final parts = v.split('/');
    if (parts.length != 2) return null;
    final n = double.tryParse(parts[0]);
    final d = double.tryParse(parts[1]);
    if (n == null || d == null || d == 0) return null;
    result = n / d;
  } else {
    result = double.tryParse(v);
  }
  if (result == null || result.isNaN || result <= 0 || result > 1000) return null;
  return result;
}
