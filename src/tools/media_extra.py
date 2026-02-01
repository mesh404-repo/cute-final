"""Extra media tools - image_info, crop_image, sample_image_pixels, image_similarity, image_diff."""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

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


def run_image_info(cwd: Path, path: str) -> ToolResult:
    """Return basic metadata about an image file (format, width, height, size)."""
    src = _resolve(cwd, path)
    if not src.exists():
        return ToolResult.fail(f"File not found: {src}")
    if not src.is_file():
        return ToolResult.fail(f"Not a file: {src}")

    size = int(src.stat().st_size)
    identify = shutil.which("identify")
    if identify:
        try:
            proc = subprocess.run(
                [identify, "-format", "%m %w %h", str(src)],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if proc.returncode == 0 and proc.stdout:
                parts = proc.stdout.strip().split()
                if len(parts) >= 3:
                    return ToolResult.ok(
                        f"path: {src}\nformat: {parts[0]}\nwidth: {parts[1]}\nheight: {parts[2]}\nsize_bytes: {size}"
                    )
        except Exception:
            pass

    file_cmd = shutil.which("file")
    if file_cmd:
        try:
            proc = subprocess.run(
                [file_cmd, "-b", str(src)],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if proc.returncode == 0:
                return ToolResult.ok(f"path: {src}\ndescription: {proc.stdout.strip()}\nsize_bytes: {size}")
        except Exception:
            pass

    return ToolResult.ok(f"path: {src}\nsize_bytes: {size}")


def run_crop_image(
    cwd: Path,
    path: str,
    x: int,
    y: int,
    width: int,
    height: int,
    max_bytes: int = 300000,
) -> ToolResult:
    """Crop an image to a rectangle. Returns path to cropped attachment."""
    src = _resolve(cwd, path)
    if not src.exists():
        return ToolResult.fail(f"File not found: {src}")

    attach_dir = _ensure_attach_dir()
    out = attach_dir / f"crop_{int(time.time() * 1000)}.png"

    convert = shutil.which("convert")
    if convert:
        cmd = [
            convert,
            str(src),
            "-crop", f"{int(width)}x{int(height)}+{int(x)}+{int(y)}",
            "+repage",
            "-strip",
            str(out),
        ]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=25)
            if proc.returncode == 0 and out.exists() and out.stat().st_size <= int(max_bytes):
                return ToolResult.ok(f"path: {src}\nattachment_path: {out}\nnote: Cropped via ImageMagick convert.")
        except Exception as e:
            pass

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        cmd = [
            ffmpeg,
            "-y",
            "-i", str(src),
            "-vf", f"crop={int(width)}:{int(height)}:{int(x)}:{int(y)}",
            str(out),
        ]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=25)
            if proc.returncode == 0 and out.exists() and out.stat().st_size <= int(max_bytes):
                return ToolResult.ok(f"path: {src}\nattachment_path: {out}\nnote: Cropped via ffmpeg.")
        except Exception:
            pass

    return ToolResult.fail("Crop failed: need ImageMagick convert or ffmpeg.")


def run_sample_image_pixels(
    cwd: Path,
    path: str,
    points: List[Dict[str, int]],
    max_points: int = 200,
) -> ToolResult:
    """Sample RGB pixel values from an image. Supports PPM/PNM when ImageMagick/PIL not available."""
    src = _resolve(cwd, path)
    if not src.exists():
        return ToolResult.fail(f"File not found: {src}")
    pts = points[: int(max_points)] if points else []
    if not pts:
        return ToolResult.invalid("points list is required and non-empty")

    try:
        from PIL import Image
        with Image.open(src) as im:
            im = im.convert("RGB")
            w, h = im.size
            rgb = im.tobytes()
    except Exception:
        return ToolResult.fail(
            "sample_image_pixels requires Pillow (pip install Pillow) or PNM images. "
            "Use view_image to load the image for analysis."
        )

    lines = [f"path: {src}", f"width: {w}", f"height: {h}", "samples:"]
    for p in pts:
        px = int(p.get("x", 0))
        py = int(p.get("y", 0))
        if px < 0 or py < 0 or px >= w or py >= h:
            lines.append(f"  x={px} y={py}: out_of_bounds")
            continue
        i = (py * w + px) * 3
        r, g, b = int(rgb[i]), int(rgb[i + 1]), int(rgb[i + 2])
        lines.append(f"  x={px} y={py}: r={r} g={g} b={b}")
    return ToolResult.ok("\n".join(lines))


def run_image_similarity(cwd: Path, path_a: str, path_b: str, max_dim: int = 0) -> ToolResult:
    """Compute cosine similarity between two images' RGB vectors."""
    try:
        from PIL import Image
    except Exception:
        return ToolResult.fail("image_similarity requires Pillow (pip install Pillow).")

    pa = _resolve(cwd, path_a)
    pb = _resolve(cwd, path_b)
    if not pa.exists() or not pb.exists():
        return ToolResult.fail("One or both files not found.")

    try:
        with Image.open(pa) as ima:
            ima = ima.convert("RGB")
            wa, ha = ima.size
            ra = ima.tobytes()
        with Image.open(pb) as imb:
            imb = imb.convert("RGB")
            wb, hb = imb.size
            rb = imb.tobytes()
    except Exception as e:
        return ToolResult.fail(f"Failed to load images: {e}")

    if wa != wb or ha != hb:
        return ToolResult.fail(f"Dimension mismatch: {wa}x{ha} vs {wb}x{hb}")

    if max_dim > 0 and (wa > max_dim or ha > max_dim):
        scale = min(max_dim / wa, max_dim / ha, 1.0)
        nw, nh = max(1, int(wa * scale)), max(1, int(ha * scale))
        with Image.open(pa) as ima:
            ima = ima.convert("RGB").resize((nw, nh), Image.Resampling.NEAREST)
            ra = ima.tobytes()
        with Image.open(pb) as imb:
            imb = imb.convert("RGB").resize((nw, nh), Image.Resampling.NEAREST)
            rb = imb.tobytes()
        wa, ha = nw, nh

    n = len(ra)
    dot = na = nb = 0.0
    for i in range(n):
        aa, bb = float(ra[i]), float(rb[i])
        dot += aa * bb
        na += aa * aa
        nb += bb * bb
    denom = (na ** 0.5) * (nb ** 0.5)
    sim = dot / (denom + 1e-12) if denom else 0.0
    return ToolResult.ok(f"path_a: {pa}\npath_b: {pb}\nwidth: {wa}\nheight: {ha}\nmetric: cosine_rgb\nsimilarity: {sim:.6f}")


def run_image_diff(
    cwd: Path,
    path_a: str,
    path_b: str,
    max_dim: int = 256,
    max_bytes: int = 300000,
) -> ToolResult:
    """Create a visual diff image between two images. Returns path to diff attachment."""
    try:
        from PIL import Image
    except Exception:
        return ToolResult.fail("image_diff requires Pillow (pip install Pillow).")

    pa = _resolve(cwd, path_a)
    pb = _resolve(cwd, path_b)
    if not pa.exists() or not pb.exists():
        return ToolResult.fail("One or both files not found.")

    try:
        with Image.open(pa) as ima:
            ima = ima.convert("RGB")
            wa, ha = ima.size
            ra = ima.tobytes()
        with Image.open(pb) as imb:
            imb = imb.convert("RGB")
            wb, hb = imb.size
            rb = imb.tobytes()
    except Exception as e:
        return ToolResult.fail(f"Failed to load images: {e}")

    if wa != wb or ha != hb:
        return ToolResult.fail(f"Dimension mismatch: {wa}x{ha} vs {wb}x{hb}")

    if max_dim > 0 and (wa > max_dim or ha > max_dim):
        scale = min(max_dim / wa, max_dim / ha, 1.0)
        nw, nh = max(1, int(wa * scale)), max(1, int(ha * scale))
        with Image.open(pa) as ima:
            ima = ima.convert("RGB").resize((nw, nh), Image.Resampling.NEAREST)
            ra = ima.tobytes()
        with Image.open(pb) as imb:
            imb = imb.convert("RGB").resize((nw, nh), Image.Resampling.NEAREST)
            rb = imb.tobytes()
        wa, ha = nw, nh

    diff = bytearray(len(ra))
    mean_abs = max_abs = mse = 0.0
    for i in range(len(ra)):
        da = int(ra[i]) - int(rb[i])
        if da < 0:
            da = -da
        if da > 255:
            da = 255
        diff[i] = min(255, int(da * 4))
        mean_abs += da
        if da > max_abs:
            max_abs = da
        mse += da * da
    n = len(ra)
    mean_abs /= n
    mse /= n
    rmse = mse ** 0.5

    attach_dir = _ensure_attach_dir()
    out = attach_dir / f"diff_{int(time.time() * 1000)}.png"
    try:
        img = Image.frombytes("RGB", (wa, ha), bytes(diff))
        img.save(out, "PNG")
        if out.stat().st_size <= int(max_bytes):
            return ToolResult.ok(
                f"path_a: {pa}\npath_b: {pb}\ndiff_path: {out}\n"
                f"mean_abs: {mean_abs:.4f}\nmax_abs: {int(max_abs)}\nmse: {mse:.4f}\nrmse: {rmse:.4f}"
            )
    except Exception as e:
        return ToolResult.fail(f"Failed to write diff image: {e}")
    return ToolResult.ok(
        f"path_a: {pa}\npath_b: {pb}\nmean_abs: {mean_abs:.4f}\nmax_abs: {int(max_abs)}\nmse: {mse:.4f}\nrmse: {rmse:.4f}\n(diff image too large to save)"
    )
