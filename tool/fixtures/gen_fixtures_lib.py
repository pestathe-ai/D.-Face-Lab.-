import numpy as np
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


