"""Search text tool - search for pattern within files with structured matches."""

from __future__ import annotations

import json
import os
import re
import subprocess
import shutil
from pathlib import Path
from typing import Any, Dict, List

from src.tools.base import ToolResult


def _resolve(cwd: Path, path: str) -> Path:
    p = Path(path)
    if not p.is_absolute():
        p = cwd / p
    return p.resolve(strict=False)


def run_search_text(
    cwd: Path,
    pattern: str,
    path: str,
    regex: bool = True,
    glob: str = "",
    max_matches: int = 100,
    context_lines: int = 0,
) -> ToolResult:
    """Search for pattern within text files. Returns structured matches."""
    root = _resolve(cwd, path)
    if not root.exists():
        return ToolResult.fail(f"Path does not exist: {root}")
    max_matches = int(max_matches)
    context_lines = int(context_lines)

    rg = shutil.which("rg")
    if rg:
        return _run_rg(cwd, rg, pattern, root, regex, glob, max_matches, context_lines)
    return _run_python(cwd, pattern, root, regex, glob, max_matches, context_lines)


def _run_rg(
    cwd: Path,
    rg: str,
    pattern: str,
    root: Path,
    regex: bool,
    glob: str,
    max_matches: int,
    context_lines: int,
) -> ToolResult:
    cmd = [
        rg,
        "--json",
        "--no-heading",
        "--line-number",
        "--column",
        "--color=never",
        f"--max-count={max_matches}",
    ]
    if not regex:
        cmd.append("--fixed-strings")
    if context_lines > 0:
        cmd.append(f"--context={context_lines}")
    if glob:
        cmd.append(f"--glob={glob}")
    cmd.append(pattern)
    cmd.append(str(root))
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30, cwd=str(cwd))
    except subprocess.TimeoutExpired:
        return ToolResult.fail("search_text timed out (rg)")
    except Exception as e:
        return ToolResult.fail(f"rg failed: {e}")

    if proc.returncode not in (0, 1):
        return ToolResult.fail(f"rg error (exit {proc.returncode}): {(proc.stderr or '')[:500]}")

    matches: List[Dict[str, Any]] = []
    for line in (proc.stdout or "").splitlines():
        try:
            obj = json.loads(line)
        except Exception:
            continue
        if obj.get("type") != "match":
            continue
        data = obj.get("data", {})
        path_obj = data.get("path", {})
        text_obj = data.get("lines", {})
        m = data.get("submatches", [{}])[0]
        matches.append({
            "file": path_obj.get("text", ""),
            "line": data.get("line_number"),
            "column": (m.get("start") + 1) if isinstance(m.get("start"), int) else data.get("absolute_offset"),
            "text": (text_obj.get("text") or "").rstrip("\n"),
        })
        if len(matches) >= max_matches:
            break

    lines = []
    for m in matches:
        lines.append(f"{m['file']}:{m['line']}:{m.get('column', '')}: {m['text']}")
    output = "\n".join(lines) if lines else "No matches found"
    if len(matches) >= max_matches:
        output += f"\n\n[... {len(matches)} matches, truncated ...]"
    return ToolResult.ok(output)


def _run_python(
    cwd: Path,
    pattern: str,
    root: Path,
    regex: bool,
    glob: str,
    max_matches: int,
    context_lines: int,
) -> ToolResult:
    try:
        cre = re.compile(pattern) if regex else None
    except re.error as e:
        return ToolResult.fail(f"Invalid regex: {e}")

    paths: List[Path] = []
    if root.is_file():
        paths = [root]
    else:
        for dirpath, dirnames, filenames in os.walk(root):
            for fn in filenames:
                if glob and not Path(fn).match(glob):
                    continue
                paths.append(Path(dirpath) / fn)

    matches: List[Dict[str, Any]] = []
    for p in paths:
        try:
            txt = p.read_text(encoding="utf-8", errors="replace").splitlines()
        except Exception:
            continue
        for i, line in enumerate(txt, start=1):
            if regex and cre:
                m = cre.search(line)
                ok = m is not None
                col = (m.start() + 1) if m else None
            else:
                idx = line.find(pattern)
                ok = idx != -1
                col = idx + 1 if idx != -1 else None
            if ok:
                snippet = line
                if context_lines > 0:
                    start = max(0, i - 1 - context_lines)
                    end = min(len(txt), i - 1 + context_lines + 1)
                    snippet = "\n".join(txt[start:end])
                matches.append({"file": str(p), "line": i, "column": col, "text": snippet})
                if len(matches) >= max_matches:
                    break
        if len(matches) >= max_matches:
            break

    lines = []
    for m in matches:
        lines.append(f"{m['file']}:{m['line']}:{m.get('column', '')}: {m['text']}")
    output = "\n".join(lines) if lines else "No matches found"
    if len(matches) >= max_matches:
        output += f"\n\n[... {len(matches)} matches, truncated ...]"
    return ToolResult.ok(output)
