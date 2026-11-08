"""Read Niantic SPZ (v2/v3, gzip) into numpy arrays, following nianticlabs/spz load-spz.cc.
Values are returned in the file's own frame (SPZ stores RUB: x right, y up, z back)."""
import gzip
import struct

import numpy as np

SH_DIM = {0: 0, 1: 3, 2: 8, 3: 15}


def read_spz(path):
    raw = open(path, "rb").read()
    data = gzip.decompress(raw) if raw[:2] == b"\x1f\x8b" else raw
    magic, version, n, shdeg, fbits, flags, _ = struct.unpack_from("<IIIBBBB", data, 0)
    assert magic == 0x5053474E, hex(magic)
    assert version in (2, 3), f"unsupported spz version {version}"
    off = 16
    pos = np.frombuffer(data, np.uint8, n * 9, off).reshape(n, 3, 3).astype(np.int32)
    off += n * 9
    v = pos[..., 0] | (pos[..., 1] << 8) | (pos[..., 2] << 16)
    v = np.where(v & 0x800000, v - (1 << 24), v)
    means = v.astype(np.float32) / (1 << fbits)
    alpha = np.frombuffer(data, np.uint8, n, off).astype(np.float32) / 255.0
    off += n
    col = np.frombuffer(data, np.uint8, n * 3, off).reshape(n, 3).astype(np.float32)
    off += n * 3
    dc = (col / 255.0 - 0.5) / 0.15
    sc = np.frombuffer(data, np.uint8, n * 3, off).reshape(n, 3).astype(np.float32)
    off += n * 3
    log_scales = sc / 16.0 - 10.0
    if version == 2:
        r = np.frombuffer(data, np.uint8, n * 3, off).reshape(n, 3).astype(np.float32)
        off += n * 3
        xyz = r / 127.5 - 1.0
        w = np.sqrt(np.clip(1 - (xyz ** 2).sum(1), 0, None))
        q_xyzw = np.concatenate([xyz, w[:, None]], 1)
    else:
        comp = np.frombuffer(data, "<u4", n, off).astype(np.uint64)
        off += n * 4
        ilarge = (comp >> np.uint64(30)).astype(np.int64)
        q_xyzw = np.zeros((n, 4), np.float32)
        mask = np.uint64((1 << 9) - 1)
        c = comp.copy()
        for i in (3, 2, 1, 0):
            sel = ilarge != i
            mag = (c & mask).astype(np.float32)
            neg = ((c >> np.uint64(9)) & np.uint64(1)).astype(bool)
            val = np.sqrt(0.5) * mag / float(mask)
            val = np.where(neg, -val, val)
            q_xyzw[sel, i] = val[sel]
            c = np.where(sel, c >> np.uint64(10), c)
        rest = (q_xyzw ** 2).sum(1)
        q_xyzw[np.arange(n), ilarge] = np.sqrt(np.clip(1 - rest, 0, None))
    k = SH_DIM[shdeg]
    if k:
        sh = np.frombuffer(data, np.uint8, n * k * 3, off).reshape(n, k, 3).astype(np.float32)
        off += n * k * 3
        sh = (sh - 128.0) / 128.0
    else:
        sh = np.zeros((n, 0, 3), np.float32)
    assert off == len(data), (off, len(data))
    return dict(version=version, n=n, sh_degree=shdeg, flags=flags, means=means, opacity=alpha,
                sh=np.concatenate([dc[:, None], sh], 1), log_scales=log_scales, q_xyzw=q_xyzw)
