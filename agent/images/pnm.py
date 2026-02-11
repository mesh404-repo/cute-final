"""
Pure-Python PNM (PPM/PGM/PBM) to PNG conversion.

Anthropic vision accepts only image/jpeg, image/png, image/gif, image/webp.
This module converts PNM files to PNG so they can be sent to the API.
"""

from __future__ import annotations

import struct
import zlib
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple


def _read_pnm_token(f) -> bytes:
    tok = bytearray()
    while True:
        ch = f.read(1)
        if not ch:
            return bytes(tok)
        c = ch[0]
        if c in (9, 10, 13, 32):
            if tok:
                return bytes(tok)
            continue
        if c == 35:
            f.readline()
            continue
        tok.append(c)
        break
    while True:
        ch = f.read(1)
        if not ch:
            break
        c = ch[0]
        if c in (9, 10, 13, 32):
            break
        tok.append(c)
    return bytes(tok)


def _parse_pnm_header(path: Path) -> Dict[str, Any]:
    with path.open("rb") as f:
        magic_b = _read_pnm_token(f)
        if not magic_b:
            raise ValueError("Empty file")
        magic = magic_b.decode("ascii", errors="ignore")
        if not magic.startswith("P"):
            raise ValueError(f"Not a PNM file (magic={magic!r})")
        w = int(_read_pnm_token(f))
        h = int(_read_pnm_token(f))
        if magic in {"P1", "P4"}:
            maxval = 1
        else:
            maxval = int(_read_pnm_token(f))
        offset = int(f.tell())
    return {"magic": magic, "width": w, "height": h, "maxval": maxval, "offset": offset}


def _iter_ascii_ints_stream(f, chunk_size: int = 1 << 20) -> Iterable[int]:
    buf = b""
    eof = False
    while not eof:
        chunk = f.read(int(chunk_size))
        if not chunk:
            eof = True
        else:
            buf += chunk
        i = 0
        n = len(buf)
        while i < n:
            b = buf[i]
            if 48 <= b <= 57:
                j = i + 1
                while j < n and 48 <= buf[j] <= 57:
                    j += 1
                if j == n and not eof:
                    break
                yield int(buf[i:j])
                i = j
                continue
            if b == 35:
                nl = buf.find(b"\n", i)
                if nl == -1:
                    if eof:
                        i = n
                        break
                    break
                i = nl + 1
                continue
            i += 1
        buf = buf[i:]
    i = 0
    n = len(buf)
    while i < n:
        b = buf[i]
        if 48 <= b <= 57:
            j = i + 1
            while j < n and 48 <= buf[j] <= 57:
                j += 1
            yield int(buf[i:j])
            i = j
        elif b == 35:
            nl = buf.find(b"\n", i)
            if nl == -1:
                break
            i = nl + 1
        else:
            i += 1


def _load_pnm_rgb(path: Path, *, max_pixels: Optional[int] = None) -> Tuple[int, int, bytes]:
    try:
        from PIL import Image
        with Image.open(path) as im:
            im = im.convert("RGB")
            w, h = im.size
            if w <= 0 or h <= 0:
                raise ValueError(f"Invalid dimensions: {w}x{h}")
            if max_pixels is not None and (w * h) > int(max_pixels):
                raise ValueError(f"Image too large ({w}x{h}) for max_pixels={max_pixels}")
            rgb = im.tobytes()
            if len(rgb) != w * h * 3:
                raise ValueError("Unexpected RGB buffer size from Pillow")
            return int(w), int(h), rgb
    except Exception:
        pass
    hdr = _parse_pnm_header(path)
    magic = str(hdr["magic"])
    w = int(hdr["width"])
    h = int(hdr["height"])
    maxval = int(hdr.get("maxval", 255))
    if w <= 0 or h <= 0:
        raise ValueError(f"Invalid dimensions: {w}x{h}")
    total_px = w * h
    if max_pixels is not None and total_px > int(max_pixels):
        raise ValueError(f"Image too large ({w}x{h}) for max_pixels={max_pixels}")
    scale = 1.0
    if maxval and maxval != 255:
        scale = 255.0 / float(maxval)

    def _norm(v: int) -> int:
        vv = int(float(v) * scale + 0.5) if maxval != 255 else int(v)
        if vv < 0:
            return 0
        if vv > 255:
            return 255
        return vv

    if magic == "P6":
        bytes_per_sample = 2 if maxval > 255 else 1
        need = w * h * 3 * bytes_per_sample
        with path.open("rb") as f:
            f.seek(int(hdr["offset"]))
            raw = f.read(need)
        if len(raw) < need:
            raise ValueError("Unexpected EOF reading P6 pixel data")
        if bytes_per_sample == 1:
            if maxval == 255:
                return w, h, raw
            out = bytearray(w * h * 3)
            for i, b in enumerate(raw[: w * h * 3]):
                out[i] = _norm(int(b))
            return w, h, bytes(out)
        out = bytearray(w * h * 3)
        oi = 0
        for i in range(0, len(raw), 2):
            v16 = (raw[i] << 8) | raw[i + 1]
            out[oi] = _norm(int(v16))
            oi += 1
            if oi >= len(out):
                break
        return w, h, bytes(out)

    if magic == "P5":
        bytes_per_sample = 2 if maxval > 255 else 1
        need = w * h * bytes_per_sample
        with path.open("rb") as f:
            f.seek(int(hdr["offset"]))
            raw = f.read(need)
        if len(raw) < need:
            raise ValueError("Unexpected EOF reading P5 pixel data")
        out = bytearray(w * h * 3)
        if bytes_per_sample == 1:
            for i in range(w * h):
                g = _norm(int(raw[i]))
                j = i * 3
                out[j : j + 3] = bytes((g, g, g))
            return w, h, bytes(out)
        oi = 0
        for i in range(0, len(raw), 2):
            v16 = (raw[i] << 8) | raw[i + 1]
            g = _norm(int(v16))
            out[oi : oi + 3] = bytes((g, g, g))
            oi += 3
            if oi >= len(out):
                break
        return w, h, bytes(out)

    if magic == "P4":
        row_bytes = (w + 7) // 8
        need = row_bytes * h
        with path.open("rb") as f:
            f.seek(int(hdr["offset"]))
            raw = f.read(need)
        if len(raw) < need:
            raise ValueError("Unexpected EOF reading P4 pixel data")
        out = bytearray(w * h * 3)
        for y in range(h):
            row = raw[y * row_bytes : (y + 1) * row_bytes]
            for x in range(w):
                byte = row[x // 8]
                bit = (byte >> (7 - (x % 8))) & 1
                g = 0 if bit == 1 else 255
                i = (y * w + x) * 3
                out[i : i + 3] = bytes((g, g, g))
        return w, h, bytes(out)

    if magic == "P3":
        need = w * h * 3
        out = bytearray(need)
        idx = 0
        with path.open("rb") as f:
            f.seek(int(hdr["offset"]))
            for v in _iter_ascii_ints_stream(f):
                if idx >= need:
                    break
                out[idx] = _norm(int(v))
                idx += 1
        if idx < need:
            raise ValueError(f"Unexpected EOF reading P3 data: got {idx} values, need {need}")
        return w, h, bytes(out)

    if magic == "P2":
        need = w * h
        gray = bytearray(need)
        idx = 0
        with path.open("rb") as f:
            f.seek(int(hdr["offset"]))
            for v in _iter_ascii_ints_stream(f):
                if idx >= need:
                    break
                gray[idx] = _norm(int(v))
                idx += 1
        if idx < need:
            raise ValueError(f"Unexpected EOF reading P2 data: got {idx} values, need {need}")
        out = bytearray(w * h * 3)
        for i in range(need):
            g = int(gray[i])
            j = i * 3
            out[j : j + 3] = bytes((g, g, g))
        return w, h, bytes(out)

    if magic == "P1":
        need = w * h
        out = bytearray(w * h * 3)
        idx = 0
        with path.open("rb") as f:
            f.seek(int(hdr["offset"]))
            for v in _iter_ascii_ints_stream(f):
                if idx >= need:
                    break
                bit = int(v)
                g = 0 if bit == 1 else 255
                j = idx * 3
                out[j : j + 3] = bytes((g, g, g))
                idx += 1
        if idx < need:
            raise ValueError(f"Unexpected EOF reading P1 data: got {idx} values, need {need}")
        return w, h, bytes(out)

    raise ValueError(f"Unsupported PNM type: {magic}")


def _downscale_rgb_nearest(rgb: bytes, w: int, h: int, max_dim: int) -> Tuple[int, int, bytes]:
    max_dim = int(max_dim)
    if max_dim <= 0 or (w <= max_dim and h <= max_dim):
        return w, h, rgb
    scale = float(max(w, h)) / float(max_dim)
    out_w = max(1, int(round(float(w) / scale)))
    out_h = max(1, int(round(float(h) / scale)))
    out = bytearray(out_w * out_h * 3)
    for y_out in range(out_h):
        y_src = min(int(float(y_out) * float(h) / float(out_h)), h - 1)
        row_src = y_src * w * 3
        row_out = y_out * out_w * 3
        for x_out in range(out_w):
            x_src = min(int(float(x_out) * float(w) / float(out_w)), w - 1)
            si = row_src + x_src * 3
            di = row_out + x_out * 3
            out[di : di + 3] = rgb[si : si + 3]
    return out_w, out_h, bytes(out)


def _png_chunk(chunk_type: bytes, data: bytes) -> bytes:
    crc = zlib.crc32(chunk_type)
    crc = zlib.crc32(data, crc)
    return struct.pack(">I", len(data)) + chunk_type + data + struct.pack(">I", crc & 0xFFFFFFFF)


def _write_png_bytes(w: int, h: int, rgb: bytes) -> bytes:
    if len(rgb) != w * h * 3:
        raise ValueError("RGB buffer size mismatch")
    stride = w * 3
    raw = bytearray((stride + 1) * h)
    out_i = 0
    in_i = 0
    for _ in range(h):
        raw[out_i] = 0
        out_i += 1
        raw[out_i : out_i + stride] = rgb[in_i : in_i + stride]
        out_i += stride
        in_i += stride
    comp = zlib.compress(bytes(raw), level=9)
    ihdr = struct.pack(">IIBBBBB", int(w), int(h), 8, 2, 0, 0, 0)
    sig = b"\x89PNG\r\n\x1a\n"
    png = sig + _png_chunk(b"IHDR", ihdr) + _png_chunk(b"IDAT", comp) + _png_chunk(b"IEND", b"")
    return png


def pnm_to_png_bytes(
    path: Path,
    max_width: int = 2048,
    max_height: int = 768,
    max_pixels: Optional[int] = None,
) -> bytes:
    """
    Load a PNM file (PPM/PGM/PBM) and return PNG bytes.
    Downscales if needed to fit max_width x max_height.
    """
    w, h, rgb = _load_pnm_rgb(path, max_pixels=max_pixels)
    w2, h2, rgb2 = _downscale_rgb_nearest(rgb, w, h, max(max_width, max_height))
    return _write_png_bytes(w2, h2, rgb2)
