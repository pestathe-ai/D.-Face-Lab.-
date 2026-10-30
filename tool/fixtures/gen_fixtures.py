"""Generates test fixtures for the Dart pipeline replay test.

Pure-numpy twin of lib/core/*.dart. Runs the real models once and records
their inputs/outputs so the Dart test can replay them without ONNX Runtime.
"""
import json, sys, os
import numpy as np, cv2, onnxruntime as ort
from onnx import helper, TensorProto, numpy_helper

MODELS, IMAGES, OUT = sys.argv[1:4]
os.makedirs(OUT, exist_ok=True)
ARC112 = np.array([[0.34191607, 0.46157411], [0.65653393, 0.45983393], [0.50022500, 0.64050536], [0.37097589, 0.82469196], [0.63151696, 0.82325089]])
ARC128 = np.array([[0.36167656, 0.40387734], [0.63696719, 0.40235469], [0.50019687, 0.56044219], [0.38710391, 0.72160547], [0.61507734, 0.72034453]])
DET = 320


def similarity(src, dst):
    ms, md = src.mean(0), dst.mean(0)
    s, d = src - ms, dst - md
    var = (s ** 2).sum()
    a = (s[:, 0] * d[:, 0] + s[:, 1] * d[:, 1]).sum() / var
    b = (s[:, 0] * d[:, 1] - s[:, 1] * d[:, 0]).sum() / var
    return np.array([[a, -b, md[0] - (a * ms[0] - b * ms[1])], [b, a, md[1] - (b * ms[0] + a * ms[1])]])


def invert(m):
    a, b, c, d, e, f = m.ravel()
    det = a * e - b * d
    return np.array([[e / det, -b / det, (b * f - c * e) / det], [-d / det, a / det, (c * d - a * f) / det]])


def bilinear(img, xs, ys):
    h, w = img.shape[:2]
    xs = np.clip(xs, 0, w - 1); ys = np.clip(ys, 0, h - 1)
    x0 = np.floor(xs).astype(int); y0 = np.floor(ys).astype(int)
    x1 = np.minimum(x0 + 1, w - 1); y1 = np.minimum(y0 + 1, h - 1)
    fx = xs - x0; fy = ys - y0
    if img.ndim == 3: fx = fx[..., None]; fy = fy[..., None]
    top = img[y0, x0] * (1 - fx) + img[y0, x1] * fx
    bot = img[y1, x0] * (1 - fx) + img[y1, x1] * fx
    return top * (1 - fy) + bot * fy


def resize(img, nw, nh):
    h, w = img.shape[:2]
    sx = (np.arange(nw) + 0.5) * w / nw - 0.5
    sy = (np.arange(nh) + 0.5) * h / nh - 0.5
    X, Y = np.meshgrid(sx, sy)
    return bilinear(img.astype(np.float64), X, Y)


def warp(img, m, size):
    inv = invert(m)
    scale = np.sqrt(abs(m[0, 0] * m[1, 1] - m[0, 1] * m[1, 0]))
    n = int(min(max(np.ceil(1 / scale), 1), 4))
    v, u = np.mgrid[0:size, 0:size].astype(np.float64)
    acc = np.zeros((size, size, 3))
    for j in range(n):
        for i in range(n):
            uu = u + (i + 0.5) / n - 0.5; vv = v + (j + 0.5) / n - 0.5
            xs = inv[0, 0] * uu + inv[0, 1] * vv + inv[0, 2]
            ys = inv[1, 0] * uu + inv[1, 1] * vv + inv[1, 2]
            acc += bilinear(img.astype(np.float64), xs, ys)
    return acc / (n * n)


def prepare_detect(img):
    h, w = img.shape[:2]
    scale = min(1.0, DET / h, DET / w)
    nw, nh = int(w * scale), int(h * scale)
    small = resize(img, nw, nh) if scale < 1 else img.astype(np.float64)
    t = np.full((3, DET, DET), (0 - 127.5) / 128.0)
    t[0, :nh, :nw] = (small[:, :, 2] - 127.5) / 128.0
    t[1, :nh, :nw] = (small[:, :, 1] - 127.5) / 128.0
    t[2, :nh, :nw] = (small[:, :, 0] - 127.5) / 128.0
    return t.astype(np.float32), w / nw, h / nh


def decode(outs, rw, rh, thr=0.5, nms=0.4):
    cand = []
    for i, st in enumerate([8, 16, 32]):
        sc = outs[i].reshape(-1); bb = outs[i + 3].reshape(-1, 4); kk = outs[i + 6].reshape(-1, 10)
        g = DET // st
        for k in range(len(sc)):
            if sc[k] < thr: continue
            cell = k // 2
            cx, cy = (cell % g) * st, (cell // g) * st
            box = [(cx - bb[k, 0] * st) * rw, (cy - bb[k, 1] * st) * rh, (cx + bb[k, 2] * st) * rw, (cy + bb[k, 3] * st) * rh]
            kps = []
            for j in range(5):
                kps += [(cx + kk[k, 2 * j] * st) * rw, (cy + kk[k, 2 * j + 1] * st) * rh]
            cand.append((float(sc[k]), [float(v) for v in box], [float(v) for v in kps]))
    cand.sort(key=lambda c: -c[0])
    keep = []
    for c in cand:
        ok = True
        for k in keep:
            a, b = c[1], k[1]
            iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0])); ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
            inter = iw * ih
            union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
            if union > 0 and inter / union > nms: ok = False; break
        if ok: keep.append(c)
    return keep


def box_mask(size=128, blur=0.3):
    amount = int(size * 0.5 * blur); area = max(amount // 2, 1)
    m = np.ones((size, size)); m[:area] = 0; m[-area:] = 0; m[:, :area] = 0; m[:, -area:] = 0
    sigma = amount * 0.25; r = int(np.ceil(sigma * 3))
    k = np.exp(-(np.arange(-r, r + 1) ** 2) / (2 * sigma * sigma)); k /= k.sum()
    pad = np.pad(m, r, mode='reflect')
    tmp = np.stack([np.convolve(row, k, 'valid') for row in pad])
    return np.stack([np.convolve(col, k, 'valid') for col in tmp.T]).T


def paste(img, out_hwc, mask, m, size=128):
    inv = invert(m)
    corners = np.array([[0, 0], [size, 0], [size, size], [0, size]], float)
    pts = corners @ inv[:, :2].T + inv[:, 2]
    h, w = img.shape[:2]
    x1 = int(np.clip(np.floor(pts[:, 0].min()), 0, w)); x2 = int(np.clip(np.ceil(pts[:, 0].max()), 0, w))
    y1 = int(np.clip(np.floor(pts[:, 1].min()), 0, h)); y2 = int(np.clip(np.ceil(pts[:, 1].max()), 0, h))
    yy, xx = np.mgrid[y1:y2, x1:x2].astype(float)
    u = m[0, 0] * xx + m[0, 1] * yy + m[0, 2]; v = m[1, 0] * xx + m[1, 1] * yy + m[1, 2]
    inside = (u >= 0) & (u <= size - 1) & (v >= 0) & (v <= size - 1)
    a = bilinear(mask, u, v) * inside
    f = bilinear(out_hwc, u, v)
    res = img.astype(np.float64).copy()
    res[y1:y2, x1:x2] = res[y1:y2, x1:x2] * (1 - a[..., None]) + f * a[..., None]
    return np.clip(np.floor(res + 0.5), 0, 255).astype(np.uint8)


blobs = []; manifest = {}


def put(name, arr):
    arr = np.ascontiguousarray(arr, dtype='<f4').ravel()
    off = sum(len(b) for b in blobs)
    blobs.append(arr.tobytes())
    manifest[name] = {'offset': off, 'length': int(arr.size)}


det = ort.InferenceSession(MODELS + '/scrfd_2.5g.onnx')
rec = ort.InferenceSession(MODELS + '/arcface_w600k_r50.onnx')
swp = ort.InferenceSession(MODELS + '/inswapper_128.onnx')
emap = numpy_helper.to_array(__import__('onnx').load(MODELS + '/inswapper_128.onnx').graph.initializer[-1])

src = cv2.imread(IMAGES + '/Tom_Hanks_54745.png')[:, :, ::-1].copy()
tgt = cv2.imread(IMAGES + '/t1.jpg')[0:600, 440:1280, ::-1].copy()
cv2.imwrite(OUT + '/source.png', src[:, :, ::-1]); cv2.imwrite(OUT + '/target.png', tgt[:, :, ::-1])
src = cv2.imread(OUT + '/source.png')[:, :, ::-1].copy(); tgt = cv2.imread(OUT + '/target.png')[:, :, ::-1].copy()

faces = {}
for name, img in [('source', src), ('target', tgt)]:
    t, rw, rh = prepare_detect(img)
    outs = det.run(None, {'input': t[None]})
    idx = np.arange(0, t.size, 997)
    put(name + '.det.input.samples', t.ravel()[idx])
    manifest[name + '.det.input.sum'] = float(t.astype(np.float64).sum())
    for i, o in enumerate(outs): put('%s.det.out.%d' % (name, i), o)
    faces[name] = decode(outs, rw, rh)
    manifest[name + '.faces'] = [{'score': f[0], 'box': f[1], 'kps': f[2]} for f in faces[name]]
print('faces', {k: len(v) for k, v in faces.items()})

sf = max(faces['source'], key=lambda f: (f[1][2] - f[1][0]) * (f[1][3] - f[1][1]))
m = similarity(np.array(sf[2]).reshape(5, 2), ARC112 * 112)
crop = warp(src, m, 112)
x = (crop * (1 / 127.5) - 1).transpose(2, 0, 1).astype(np.float32)
put('embed.input', x)
e = rec.run(None, {'input': x[None]})[0][0]
put('embed.out', e)
latent = (e @ emap) / np.linalg.norm(e)
put('latent', latent)

mask = box_mask()
put('mask', mask)
out = tgt.copy()
for k, f in enumerate(faces['target']):
    m = similarity(np.array(f[2]).reshape(5, 2), ARC128 * 128)
    crop = warp(out, m, 128)
    x = (crop * (1 / 255.0)).transpose(2, 0, 1).astype(np.float32)
    put('swap.%d.input' % k, x)
    o = swp.run(None, {'target': x[None], 'source': latent[None].astype(np.float32)})[0][0]
    put('swap.%d.out' % k, o)
    hwc = np.clip(o.astype(np.float64), 0, 1).transpose(1, 2, 0) * 255
    out = paste(out, hwc, mask, m)
cv2.imwrite(OUT + '/expected.png', out[:, :, ::-1])
with open(OUT + '/replay.bin', 'wb') as fh:
    for b in blobs: fh.write(b)
with open(OUT + '/replay.json', 'w') as fh:
    json.dump(manifest, fh)

# Tiny ONNX models for the initializer reader test.
a = numpy_helper.from_array(np.arange(6, dtype=np.float32).reshape(2, 3), 'first')
b = numpy_helper.from_array((np.arange(12, dtype=np.float32) * 0.5 - 1).reshape(4, 3), 'emap')
node = helper.make_node('Identity', ['x'], ['y'])
g = helper.make_graph([node], 'g', [helper.make_tensor_value_info('x', TensorProto.FLOAT, [2])],
                      [helper.make_tensor_value_info('y', TensorProto.FLOAT, [2])], initializer=[a, b])
with open(OUT + '/tiny_raw.onnx', 'wb') as fh: fh.write(helper.make_model(g).SerializeToString())
b2 = helper.make_tensor('emap', TensorProto.FLOAT, [4, 3], (np.arange(12, dtype=np.float32) * 0.5 - 1).tolist(), raw=False)
g2 = helper.make_graph([node], 'g', [helper.make_tensor_value_info('x', TensorProto.FLOAT, [2])],
                       [helper.make_tensor_value_info('y', TensorProto.FLOAT, [2])], initializer=[a, b2])
with open(OUT + '/tiny_floatdata.onnx', 'wb') as fh: fh.write(helper.make_model(g2).SerializeToString())
print('ok', sum(len(b) for b in blobs) / 1e6, 'MB')
