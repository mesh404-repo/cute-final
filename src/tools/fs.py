"""Filesystem tools - tree, file_info, glob, diff_files."""

from __future__ import annotations

import difflib
import hashlib
from pathlib import Path
from typing import Any, Optional

from src.tools.base import ToolResult


def _resolve(cwd: Path, path: str) -> Path:
    p = Path(path)
    if not p.is_absolute():
        p = cwd / p
    return p.resolve(strict=False)


def _truncate_lines(text: str, max_lines: int) -> tuple[str, bool]:
    lines = text.splitlines()
    if len(lines) <= max_lines:
        return text, False
    head = max(1, int(max_lines * 0.35))
    tail = max(1, max_lines - head)
    out = "\n".join(lines[:head] + ["...<truncated>..."] + lines[-tail:])
    return out, True


def run_tree(cwd: Path, path: str, max_depth: int = 4, max_entries: int = 500) -> ToolResult:
    """Produce a compact directory tree."""
    root = _resolve(cwd, path)
    if not root.exists():
        return ToolResult.fail(f"Path does not exist: {root}")

    lines: list[str] = []
    count = 0

    def walk(dirpath: Path, depth: int, prefix: str) -> None:
        nonlocal count
        if count >= max_entries or depth > max_depth:
            return
        try:
            children = sorted(dirpath.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
        except Exception:
            return
        for child in children:
            if count >= max_entries:
                return
            name = child.name + ("/" if child.is_dir() else "")
            lines.append(f"{prefix}{name}")
            count += 1
            if child.is_dir():
                walk(child, depth + 1, prefix + "  ")

    if root.is_dir():
        lines.append(str(root) + "/")
        walk(root, 1, "  ")
    else:
        lines.append(str(root))

    text = "\n".join(lines)
    text, trunc = _truncate_lines(text, max_lines=1000)
    out = text
    if trunc or count >= max_entries:
        out += "\n\n[... truncated ...]"
    return ToolResult.ok(out)


def run_file_info(
    cwd: Path,
    path: str,
    hash_alg: str = "",
    max_hash_bytes: Optional[int] = None,
) -> ToolResult:
    """Return metadata about a path; optionally compute hash."""
    p = _resolve(cwd, path)
    if not p.exists():
        return ToolResult.ok(f"path: {p}\nexists: False")

    try:
        st = p.stat()
        size = int(st.st_size)
        mtime = float(st.st_mtime)
        mode = int(st.st_mode)
    except Exception as e:
        return ToolResult.fail(f"stat failed: {e}")

    if p.is_dir():
        typ = "dir"
    elif p.is_file():
        typ = "file"
    elif p.is_symlink():
        typ = "symlink"
    else:
        typ = "other"

    lines = [f"path: {p}", f"exists: True", f"type: {typ}", f"size: {size}", f"mtime: {mtime}", f"mode: {mode}"]

    digest = None
    if hash_alg and typ == "file":
        h = None
        if hash_alg == "sha256":
            h = hashlib.sha256()
        elif hash_alg == "sha1":
            h = hashlib.sha1()
        elif hash_alg == "md5":
            h = hashlib.md5()
        else:
            return ToolResult.fail(f"Unsupported hash_alg: {hash_alg}")
        try:
            remaining = int(max_hash_bytes) if max_hash_bytes is not None else None
            with p.open("rb") as f:
                while True:
                    chunk_size = 1024 * 1024
                    if remaining is not None and remaining <= 0:
                        break
                    if remaining is not None:
                        chunk_size = min(chunk_size, remaining)
                    chunk = f.read(chunk_size)
                    if not chunk:
                        break
                    h.update(chunk)
                    if remaining is not None:
                        remaining -= len(chunk)
            digest = h.hexdigest()
        except Exception as e:
            return ToolResult.fail(f"hash failed: {e}")
        lines.append(f"hash_alg: {hash_alg}")
        lines.append(f"hash: {digest}")

    return ToolResult.ok("\n".join(lines))


def run_glob(cwd: Path, pattern: str, root: str, max_matches: int = 200) -> ToolResult:
    """Find files by glob pattern."""
    r = _resolve(cwd, root)
    if not r.exists():
        return ToolResult.fail(f"Root does not exist: {r}")
    if not r.is_dir():
        return ToolResult.fail(f"Not a directory: {r}")

    matches: list[str] = []
    try:
        for p in r.glob(pattern):
            matches.append(str(p))
            if len(matches) >= max_matches:
                break
    except Exception as e:
        return ToolResult.fail(f"glob failed: {e}")

    output = "\n".join(matches) if matches else "No matches"
    if len(matches) >= max_matches:
        output += f"\n\n[... {len(matches)} matches, truncated ...]"
    return ToolResult.ok(output)


def run_diff_files(
    cwd: Path,
    path_a: str,
    path_b: str,
    max_lines: int = 400,
    context: int = 3,
) -> ToolResult:
    """Compute unified diff between two text files."""
    a = _resolve(cwd, path_a)
    b = _resolve(cwd, path_b)
    if not a.exists() or not b.exists():
        return ToolResult.fail(f"One or both files do not exist: {a}, {b}")

    try:
        a_txt = a.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)
        b_txt = b.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)
    except Exception as e:
        return ToolResult.fail(f"Failed to read files: {e}")

    diff = difflib.unified_diff(a_txt, b_txt, fromfile=str(a), tofile=str(b), n=int(context))
    diff_text = "".join(diff)
    diff_text, trunc = _truncate_lines(diff_text, max_lines=int(max_lines))
    if trunc:
        diff_text += "\n\n[... truncated ...]"
    return ToolResult.ok(diff_text if diff_text else "(no differences)")
