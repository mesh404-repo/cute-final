"""
Hashline edit tool - integrates with SuperAgent tool registry.

Provides stable file editing using content hashes instead of line numbers.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from src.tools.base import ToolResult
from .core import (
    HashlineEditor,
    EditOperation,
    EditType,
)


class HashlineEditTool:
    """Tool for hashline-based file editing."""

    name = "hashline_edit"
    description = """Edit files using content-addressable line hashes for stable modifications.

Each line is tagged with a short hash (e.g., "1:a3|function hello():") that serves
as a stable identifier. Reference lines by their hash to make edits.

Operations:
- read: Read file with hashline format
- replace: Replace a single line by hash
- insert_after: Insert new line after target hash
- insert_before: Insert new line before target hash  
- delete: Delete line by hash
- replace_range: Replace multiple lines (target_hash to end_hash)
- batch: Apply multiple edits atomically

Format: "line_number:hash|content"
Example: 1:a3|function hello() {
         2:f1|  return "world";
         3:b2|}

When editing, reference by hash: "replace line 2:f1 with '  return \"universe\";'"

Benefits over traditional editing:
- Stable identifiers survive minor file changes
- Optimistic concurrency - edits fail if file changed
- No need to reproduce content as "anchors"
- 2-3 character hashes are memorable and verifiable
"""

    def __init__(self, cwd: Path):
        self.cwd = cwd
        self.editor = HashlineEditor(hash_length=2)

    def execute(self, **kwargs: Any) -> ToolResult:
        """Execute hashline edit operation."""
        operation = kwargs.get("operation", "read")
        file_path = kwargs.get("file_path", "")

        if not file_path:
            return ToolResult.fail("No file_path provided")

        path = Path(file_path)
        if not path.is_absolute():
            path = self.cwd / path

        try:
            if operation == "read":
                return self._execute_read(path, kwargs)
            elif operation == "replace":
                return self._execute_replace(path, kwargs)
            elif operation == "insert_after":
                return self._execute_insert_after(path, kwargs)
            elif operation == "insert_before":
                return self._execute_insert_before(path, kwargs)
            elif operation == "delete":
                return self._execute_delete(path, kwargs)
            elif operation == "replace_range":
                return self._execute_replace_range(path, kwargs)
            elif operation == "batch":
                return self._execute_batch(path, kwargs)
            else:
                return ToolResult.fail(f"Unknown operation: {operation}")
        except Exception as e:
            return ToolResult.fail(f"Hashline edit failed: {e}")

    def _execute_read(self, path: Path, kwargs: Dict[str, Any]) -> ToolResult:
        """Read file with hashline format."""
        offset = kwargs.get("offset", 1)
        limit = kwargs.get("limit", 2000)

        if not path.exists():
            return ToolResult.fail(f"File not found: {path}")

        parsed = self.editor.read_file(path)
        formatted = self.editor.format_for_display(parsed, offset, limit)

        return ToolResult.ok(formatted)

    def _execute_replace(self, path: Path, kwargs: Dict[str, Any]) -> ToolResult:
        """Replace a single line by hash."""
        target_hash = kwargs.get("target_hash", "")
        target_line = kwargs.get("target_line")
        content = kwargs.get("content", "")

        if not target_hash or content is None:
            return ToolResult.fail("replace requires target_hash and content")

        op = EditOperation(
            edit_type=EditType.REPLACE,
            target_hash=target_hash,
            target_line=target_line,
            content=content,
        )

        parsed = self.editor.read_file(path)
        result = self.editor.apply_edit(parsed, op)

        if result.success:
            self.editor.write_file(parsed)

        return ToolResult(
            success=result.success,
            output=result.message,
        )

    def _execute_insert_after(self, path: Path, kwargs: Dict[str, Any]) -> ToolResult:
        """Insert line after target hash."""
        target_hash = kwargs.get("target_hash", "")
        target_line = kwargs.get("target_line")
        content = kwargs.get("content", "")

        if not target_hash or content is None:
            return ToolResult.fail("insert_after requires target_hash and content")

        op = EditOperation(
            edit_type=EditType.INSERT_AFTER,
            target_hash=target_hash,
            target_line=target_line,
            content=content,
        )

        parsed = self.editor.read_file(path)
        result = self.editor.apply_edit(parsed, op)

        if result.success:
            self.editor.write_file(parsed)

        return ToolResult(
            success=result.success,
            output=result.message,
        )

    def _execute_insert_before(self, path: Path, kwargs: Dict[str, Any]) -> ToolResult:
        """Insert line before target hash."""
        target_hash = kwargs.get("target_hash", "")
        target_line = kwargs.get("target_line")
        content = kwargs.get("content", "")

        if not target_hash or content is None:
            return ToolResult.fail("insert_before requires target_hash and content")

        op = EditOperation(
            edit_type=EditType.INSERT_BEFORE,
            target_hash=target_hash,
            target_line=target_line,
            content=content,
        )

        parsed = self.editor.read_file(path)
        result = self.editor.apply_edit(parsed, op)

        if result.success:
            self.editor.write_file(parsed)

        return ToolResult(
            success=result.success,
            output=result.message,
        )

    def _execute_delete(self, path: Path, kwargs: Dict[str, Any]) -> ToolResult:
        """Delete line by hash."""
        target_hash = kwargs.get("target_hash", "")
        target_line = kwargs.get("target_line")

        if not target_hash:
            return ToolResult.fail("delete requires target_hash")

        op = EditOperation(
            edit_type=EditType.DELETE,
            target_hash=target_hash,
            target_line=target_line,
        )

        parsed = self.editor.read_file(path)
        result = self.editor.apply_edit(parsed, op)

        if result.success:
            self.editor.write_file(parsed)

        return ToolResult(
            success=result.success,
            output=result.message,
        )

    def _execute_replace_range(self, path: Path, kwargs: Dict[str, Any]) -> ToolResult:
        """Replace a range of lines."""
        target_hash = kwargs.get("target_hash", "")
        end_hash = kwargs.get("end_hash", "")
        target_line = kwargs.get("target_line")
        end_line = kwargs.get("end_line")
        content = kwargs.get("content", "")

        if not target_hash or not end_hash or content is None:
            return ToolResult.fail("replace_range requires target_hash, end_hash, and content")

        op = EditOperation(
            edit_type=EditType.REPLACE_RANGE,
            target_hash=target_hash,
            target_line=target_line,
            end_hash=end_hash,
            end_line=end_line,
            content=content,
        )

        parsed = self.editor.read_file(path)
        result = self.editor.apply_edit(parsed, op)

        if result.success:
            self.editor.write_file(parsed)

        return ToolResult(
            success=result.success,
            output=result.message,
        )

    def _execute_batch(self, path: Path, kwargs: Dict[str, Any]) -> ToolResult:
        """Apply multiple edits atomically."""
        edits = kwargs.get("edits", [])

        if not edits:
            return ToolResult.fail("batch requires edits list")

        operations = []
        for edit in edits:
            op_type = edit.get("operation", "replace")
            op_map = {
                "replace": EditType.REPLACE,
                "insert_after": EditType.INSERT_AFTER,
                "insert_before": EditType.INSERT_BEFORE,
                "delete": EditType.DELETE,
                "replace_range": EditType.REPLACE_RANGE,
            }

            op = EditOperation(
                edit_type=op_map.get(op_type, EditType.REPLACE),
                target_hash=edit.get("target_hash", ""),
                target_line=edit.get("target_line"),
                content=edit.get("content"),
                end_hash=edit.get("end_hash"),
                end_line=edit.get("end_line"),
            )
            operations.append(op)

        all_succeeded, results = self.editor.batch_edit(path, operations)

        messages = [r.message for r in results]
        if all_succeeded:
            return ToolResult.ok(f"Applied {len(results)} edits:\n" + "\n".join(messages))
        else:
            return ToolResult.fail(f"Batch edit failed:\n" + "\n".join(messages))


def get_tool_spec() -> dict:
    """Get tool specification for registry."""
    return {
        "name": "hashline_edit",
        "description": (
            "Edit files using content-addressable line hashes for stable modifications. "
            "Each line is tagged with a short hash that serves as a stable identifier. "
            "If the file changes since reading, the hash will not match and edit is rejected. "
            "Operations: read, replace, insert_after, insert_before, delete, replace_range, batch. "
            "Example format: 1:a3|function hello() - use the hash (a3) to reference lines."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["read", "replace", "insert_after", "insert_before", "delete", "replace_range", "batch"],
                    "description": "The edit operation to perform",
                },
                "file_path": {
                    "type": "string",
                    "description": "Path to the file to edit",
                },
                "target_hash": {
                    "type": "string",
                    "description": "Hash of the target line (2-3 chars)",
                },
                "target_line": {
                    "type": "number",
                    "description": "Optional line number hint for disambiguation",
                },
                "end_hash": {
                    "type": "string",
                    "description": "For replace_range: hash of end line",
                },
                "end_line": {
                    "type": "number",
                    "description": "For replace_range: end line number",
                },
                "content": {
                    "type": "string",
                    "description": "New content for replace/insert operations",
                },
                "offset": {
                    "type": "number",
                    "description": "For read: starting line number (1-indexed)",
                },
                "limit": {
                    "type": "number",
                    "description": "For read: maximum lines to return",
                },
                "edits": {
                    "type": "array",
                    "description": "For batch: list of edit operations",
                    "items": {"type": "object"},
                },
            },
            "required": ["operation", "file_path"],
        },
    }

