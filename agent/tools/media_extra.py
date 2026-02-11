"""Extra media tools - image_info, crop_image."""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.tools.base import ToolResult


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