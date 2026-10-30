"""HyperSwap replay fixture (run after gen_fixtures.py). Numpy twin of the
Dart HyperSwap path in lib/core/pipeline.dart."""
import json, sys, importlib.util
import numpy as np, cv2, onnxruntime as ort

MODELS, OUT = sys.argv[1:3]
spec = importlib.util.spec_from_file_location('g', __file__.replace('gen_hyper_fixture.py', 'gen_fixtures_lib.py'))
g = importlib.util.module_from_spec(spec); spec.loader.exec_module(g)

man = json.load(open(OUT + '/replay.json')); blob = open(OUT + '/replay.bin', 'rb').read()
def floats(n):
    e = man[n]; return np.frombuffer(blob, '<f4', e['length'], e['offset'])

tgt = cv2.imread(OUT + '/target.png')[:, :, ::-1].copy()
faces = man['target.faces']
face = max(faces, key=lambda f: (f['box'][2] - f['box'][0]) * (f['box'][3] - f['box'][1]))
kps = np.array(face['kps']).reshape(5, 2)
emb = floats('embed.out').astype(np.float64)
src = (emb / np.linalg.norm(emb)).astype(np.float32)
m = g.similarity(kps, g.ARC128 * 256)
crop = g.warp(tgt, m, 256)
x = (crop * (1 / 127.5) - 1).transpose(2, 0, 1).astype(np.float32)
sess = ort.InferenceSession(MODELS + '/hyperswap_1c_256.onnx', providers=['CPUExecutionProvider'])
o, mk = sess.run(None, {'target': x[None], 'source': src[None]})
o = o[0]; mk = mk[0, 0]
hwc = np.clip(o.astype(np.float64) * 0.5 + 0.5, 0, 1).transpose(1, 2, 0) * 255
mask = np.minimum(g.box_mask(256), np.clip(mk.astype(np.float64), 0, 1))
out = g.paste(tgt, hwc, mask, m, 256)
cv2.imwrite(OUT + '/expected_hyper.png', out[:, :, ::-1])
blobs = [x.ravel()[::97].astype('<f4').tobytes(), o.astype('<f4').ravel().tobytes(), mk.astype('<f4').ravel().tobytes()]
names = ['hyper.input.samples', 'hyper.out', 'hyper.mask']
meta = {'hyper.input.sum': float(x.astype(np.float64).sum()), 'face_box': face['box']}
off = 0
for n, b in zip(names, blobs):
    meta[n] = {'offset': off, 'length': len(b) // 4}; off += len(b)
open(OUT + '/hyper.bin', 'wb').write(b''.join(blobs))
json.dump(meta, open(OUT + '/hyper.json', 'w'))
print('hyper fixture ok', off / 1e6, 'MB')
