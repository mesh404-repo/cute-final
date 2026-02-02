"""
Includes PNM (PPM/PGM/PBM) helpers and view_image for preparing/downscaling
images for model vision.
"""

from __future__ import annotations

import os
import shutil
import struct
import subprocess
import time
import zlib
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from src.tools.base import ToolResult


def _resolve(cwd: Path, path: str) -> Path:
    p = Path(path)
    if not p.is_absolute():
        p = cwd / p
    return p.resolve(strict=False)


def _ensure_attach_dir() -> Path:
    d = Path(os.environ.get("TBH_ATTACH_DIR", "/tmp/tbh-attachments"))
    d.mkdir(parents=True, exist_ok=True)
    return d

# ---------------------------------------------------------------------------
# view_image: prepare/downscale/convert images for vision, then attach
# ---------------------------------------------------------------------------

_VISION_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".gif"}
_PNM_EXTS = {".ppm", ".pgm", ".pbm", ".pnm"}


def _read_pnm_token(f) -> bytes:
    """Read next whitespace-delimited token from binary file; skips # comments."""
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
    """Parse Netpbm header; return {magic, width, height, maxval, offset}."""
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
    """Yield integers from ASCII stream; skip # comments."""
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
                    i = n if eof else i
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
    """Load PNM into 8-bit RGB; returns (width, height, rgb_bytes)."""
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
    scale = 255.0 / float(maxval) if maxval and maxval != 255 else 1.0

    def _norm(v: int) -> int:
        vv = int(float(v) * scale + 0.5) if maxval != 255 else int(v)
        return max(0, min(255, vv))

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
    """Downscale RGB so max(width, height) <= max_dim using nearest-neighbor."""
    max_dim = int(max_dim)
    if max_dim <= 0:
        return w, h, rgb
    if w <= max_dim and h <= max_dim:
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


def _write_png(path: Path, w: int, h: int, rgb: bytes) -> None:
    """Write 8-bit RGB as PNG."""
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
    png = b"\x89PNG\r\n\x1a\n"
    png += _png_chunk(b"IHDR", ihdr)
    png += _png_chunk(b"IDAT", comp)
    png += _png_chunk(b"IEND", b"")
    path.write_bytes(png)


def _pnm_to_png_attachment(src: Path, dst_png: Path, *, max_dim: int, max_bytes: int) -> Tuple[bool, str]:
    """Convert PNM -> PNG with downscaling to satisfy max_bytes; pure-Python fallback."""
    max_dim = int(max_dim)
    max_bytes = int(max_bytes)
    candidates = [
        max_dim,
        int(max_dim * 0.75),
        int(max_dim * 0.5),
        int(max_dim * 0.33),
        256,
        192,
        128,
        96,
        64,
    ]
    dims = []
    for d in candidates:
        d = int(d)
        if d <= 0 or d in dims:
            continue
        dims.append(d)
    try:
        w, h, rgb = _load_pnm_rgb(src)
    except Exception as e:
        return False, f"PNM->PNG conversion failed: cannot load PNM: {e}"
    last_err = ""
    for md in dims:
        try:
            w2, h2, rgb2 = _downscale_rgb_nearest(rgb, w, h, md)
            _write_png(dst_png, w2, h2, rgb2)
            if dst_png.exists() and dst_png.stat().st_size <= max_bytes:
                return True, f"Converted PNM -> PNG via pure-Python (downscaled to {w2}x{h2})."
        except Exception as e:
            last_err = str(e)
    return False, f"PNM->PNG conversion failed or output too large. Last error: {last_err}"


def _prepare_image_path(
    cwd: Path,
    path: str,
    max_dim: int,
    max_bytes: int,
) -> Tuple[Optional[Path], str]:
    """
    Resolve path, optionally copy/downscale/convert to a vision-ready file.
    Returns (path_to_load, note) on success, or (None, error_message) on failure.
    """
    src = _resolve(cwd, path)
    if not src.exists():
        return None, f"File not found: {src}"
    if not src.is_file():
        return None, f"Not a file: {src}"

    attach_dir = _ensure_attach_dir()
    ts = int(time.time() * 1000)
    ext = src.suffix.lower()

    if ext in _VISION_EXTS:
        try:
            if src.stat().st_size <= int(max_bytes):
                dst = attach_dir / f"img_{ts}{ext}"
                shutil.copy2(src, dst)
                return dst, "Copied image as-is."
        except Exception:
            pass

    if ext in _PNM_EXTS:
        dst = attach_dir / f"img_{ts}.png"
        convert = shutil.which("convert")
        if convert:
            cmd = [
                convert, str(src),
                "-resize", f"{int(max_dim)}x{int(max_dim)}>",
                "-strip", str(dst),
            ]
            try:
                proc = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
                if proc.returncode == 0 and dst.exists() and dst.stat().st_size <= int(max_bytes):
                    return dst, "Converted via ImageMagick convert."
            except Exception:
                pass
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg:
            cmd = [
                ffmpeg, "-y", "-i", str(src),
                "-frames:v", "1",
                "-vf", f"scale='min(iw,{int(max_dim)}):min(ih,{int(max_dim)})':force_original_aspect_ratio=decrease",
                str(dst),
            ]
            try:
                proc = subprocess.run(cmd, capture_output=True, text=True, timeout=25)
                if proc.returncode == 0 and dst.exists() and dst.stat().st_size <= int(max_bytes):
                    return dst, "Converted via ffmpeg."
            except Exception:
                pass
        ok, note = _pnm_to_png_attachment(src, dst, max_dim=int(max_dim), max_bytes=int(max_bytes))
        if ok:
            return dst, note
        return None, note

    dst = attach_dir / f"img_{ts}.png"
    convert = shutil.which("convert")
    if convert:
        cmd = [
            convert, str(src),
            "-resize", f"{int(max_dim)}x{int(max_dim)}>",
            "-strip", str(dst),
        ]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
            if proc.returncode == 0 and dst.exists() and dst.stat().st_size <= int(max_bytes):
                return dst, "Downscaled via ImageMagick convert."
        except Exception:
            pass
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        cmd = [
            ffmpeg, "-y", "-i", str(src),
            "-frames:v", "1",
            "-vf", f"scale='min(iw,{int(max_dim)}):min(ih,{int(max_dim)})':force_original_aspect_ratio=decrease",
            str(dst),
        ]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=25)
            if proc.returncode == 0 and dst.exists() and dst.stat().st_size <= int(max_bytes):
                return dst, "Downscaled via ffmpeg."
        except Exception:
            pass
    try:
        if src.stat().st_size <= int(max_bytes):
            dst2 = attach_dir / f"img_{ts}{ext or '.bin'}"
            shutil.copy2(src, dst2)
            return dst2, "Copied as-is (format may not be vision-supported)."
    except Exception:
        pass
    return None, "Image too large and no available downscaler succeeded. Install ffmpeg or ImageMagick, or provide a smaller image."


def view_image(
    path: str,
    cwd: Path,
    max_dim: int = 1024,
    max_bytes: int = 300_000,
) -> ToolResult:
    """
    Prepare an image for model vision (copy/downscale/convert as needed) and attach it.

    Use when a task references an image on disk (e.g. code.png, chess_board.ppm).
    Supports PNM (PPM/PGM/PBM) via ImageMagick, ffmpeg, or pure-Python fallback.
    The image is loaded and injected into the conversation for the model to see.
    """
    from src.images.loader import load_image_as_data_uri, make_image_content

    if not path:
        return ToolResult.invalid(
            "Missing required parameter 'path'. "
            "Usage: view_image(path: str, max_dim?: int, max_bytes?: int)"
        )

    prepared_path, note = _prepare_image_path(cwd, path, max_dim, max_bytes)
    if prepared_path is None:
        return ToolResult.fail(note)

    try:
        data_uri = load_image_as_data_uri(prepared_path)
        image_content = make_image_content(data_uri)
        output_msg = f"Prepared image: {path}\n{note}\nAttached for vision."
        return ToolResult(
            success=True,
            output=output_msg,
            inject_content=image_content,
        )
    except FileNotFoundError:
        return ToolResult.fail(f"Image file not found: {prepared_path}")
    except Exception as e:
        return ToolResult.fail(f"Failed to load prepared image: {e}")