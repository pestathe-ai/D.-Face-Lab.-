"""Runs the app's real FFmpeg argument lists (written by a Dart test) on a
generated test video and checks the result. Used in CI."""
import json, os, subprocess, sys

cmds = json.load(open(sys.argv[1]))
os.makedirs('ci/frames', exist_ok=True)


def run(args):
    p = subprocess.run(['ffmpeg', '-v', 'error', *args], capture_output=True, text=True)
    if p.returncode != 0:
        print(p.stderr[-2000:])
        sys.exit('ffmpeg failed: %s' % ' '.join(args))


run(cmds['extract'])
frames = sorted(f for f in os.listdir('ci/frames') if f.endswith('.jpg'))
expected = round(cmds['fps'] * cmds['seconds'])
print('frames', len(frames), 'expected about', expected)
assert abs(len(frames) - expected) <= 2, 'unexpected frame count %d' % len(frames)
dims = subprocess.run(['ffprobe', '-v', 'error', '-show_entries', 'stream=width,height', '-of', 'csv=p=0',
                       'ci/frames/' + frames[0]], capture_output=True, text=True).stdout.strip()
print('frame size', dims)
w, h = map(int, dims.split(','))
assert max(w, h) <= 1280, dims
run(cmds['assemble'])
info = subprocess.run(['ffprobe', '-v', 'error', '-show_entries', 'stream=codec_type,codec_name,duration',
                       '-of', 'csv=p=0', 'ci/out.mp4'], capture_output=True, text=True).stdout
print(info)
assert 'h264,video' in info and 'aac,audio' in info, info
dur = float([l for l in info.splitlines() if 'video' in l][0].split(',')[-1])
assert abs(dur - cmds['seconds']) < 0.3, dur
print('ffmpeg check ok')
