"""
Hashline-based file editing system for stable, verifiable file modifications.

This module provides a content-addressable file editing system where each line
is tagged with a short content hash. This allows the model to reference lines
by their stable identifiers rather than reproducing content, reducing errors
and providing optimistic concurrency control.

Example:
    1:a3|function hello() {
    2:f1|  return "world";
    3:0e|}

When editing: "replace line 2:f1 with '  return "universe";'"
If the file changed since reading, the hash won't match and the edit is rejected.
"""

from .core import (
    HashlineEditor,
    compute_line_hash,
    format_with_hashes,
    parse_hashline,
    EditOperation,
    EditResult,
)
from .tool import HashlineEditTool

__all__ = [
    "HashlineEditor",
    "compute_line_hash",
    "format_with_hashes",
    "parse_hashline",
    "EditOperation",
    "EditResult",
    "HashlineEditTool",
]
