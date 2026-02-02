"""Extra media tools - crop_image."""

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