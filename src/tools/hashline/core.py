"""
Core hashline implementation - content-addressable line editing.

The hashline system provides stable identifiers for file lines using short
content hashes (2-3 characters). This allows:
1. Precise line referencing without reproducing content
2. Optimistic concurrency control - edits fail if content changed
3. Reduced context usage - short hashes vs full line content
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import List, Optional, Tuple, Union


class EditType(Enum):
    """Types of edit operations."""
    REPLACE = "replace"
    INSERT_AFTER = "insert_after"
    INSERT_BEFORE = "insert_before"
    DELETE = "delete"
    REPLACE_RANGE = "replace_range"


@dataclass
class LineInfo:
    """Information about a single line with its hash."""
    number: int
    hash: str
    content: str
    original: str  # Original line including newline


@dataclass
class EditOperation:
    """A single edit operation targeting lines by hash."""
    edit_type: EditType
    target_hash: str  # Primary target line hash
    target_line: Optional[int] = None  # Optional line number hint
    content: Optional[str] = None  # New content for replace/insert
    end_hash: Optional[str] = None  # For range operations
    end_line: Optional[int] = None  # End line for range


@dataclass
class EditResult:
    """Result of an edit operation."""
    success: bool
    message: str
    modified_lines: List[int] = field(default_factory=list)
    new_hashes: dict[int, str] = field(default_factory=dict)


@dataclass
class ParsedFile:
    """A file parsed into hashline format."""
    path: Path
    lines: List[LineInfo]
    hash_to_lines: dict[str, List[int]] = field(default_factory=dict)

    def __post_init__(self):
        """Build hash index."""
        self.hash_to_lines = {}
        for line in self.lines:
            if line.hash not in self.hash_to_lines:
                self.hash_to_lines[line.hash] = []
            self.hash_to_lines[line.hash].append(line.number)

    def find_line(self, hash_value: str, line_hint: Optional[int] = None) -> Optional[LineInfo]:
        """Find a line by its hash, optionally using line hint for disambiguation."""
        matches = self.hash_to_lines.get(hash_value, [])

        if not matches:
            return None

        if len(matches) == 1:
            idx = matches[0] - 1
            return self.lines[idx] if 0 <= idx < len(self.lines) else None

        # Multiple matches - use line hint if provided
        if line_hint is not None and line_hint in matches:
            idx = line_hint - 1
            return self.lines[idx] if 0 <= idx < len(self.lines) else None

        # Return first match as fallback
        idx = matches[0] - 1
        return self.lines[idx] if 0 <= idx < len(self.lines) else None

    def get_line_by_number(self, number: int) -> Optional[LineInfo]:
        """Get line by line number (1-indexed)."""
        idx = number - 1
        if 0 <= idx < len(self.lines):
            return self.lines[idx]
        return None


def compute_line_hash(content: str, length: int = 2) -> str:
    """
    Compute a short content hash for a line.

    Args:
        content: The line content (without newline)
        length: Length of hash to return (2-3 chars recommended)

    Returns:
        Short hash string
    """
    # Use first N chars of SHA-256 hash, URL-safe
    full_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
    return full_hash[:length]


def parse_hashline(line: str) -> Tuple[Optional[int], Optional[str], str]:
    """
    Parse a hashline-formatted line.

    Args:
        line: Line in format "1:a3|content" or just "content"

    Returns:
        Tuple of (line_number, hash, content) - Nones if not hashline format
    """
    # Pattern: number:hash|content
    pattern = r"^(\d+):([a-f0-9]{2,4})\|(.*)$"
    match = re.match(pattern, line)

    if match:
        number = int(match.group(1))
        hash_value = match.group(2)
        content = match.group(3)
        return number, hash_value, content

    return None, None, line


def format_with_hashes(content: str, hash_length: int = 2, existing_numbers: bool = True) -> str:
    """
    Format file content with hashline prefixes.

    Args:
        content: File content as string
        hash_length: Length of hash to use
        existing_numbers: Whether to use existing line numbers or generate new

    Returns:
        Formatted content with hashlines
    """
    lines = content.split("\n")
    result = []

    for i, line in enumerate(lines, 1):
        # Skip empty final line
        if i == len(lines) and not line:
            continue

        # Check if line already has hashline format
        num, existing_hash, parsed_content = parse_hashline(line)

        if num is not None and existing_numbers:
            # Verify existing hash matches
            computed = compute_line_hash(parsed_content, len(existing_hash) if existing_hash else hash_length)
            if existing_hash and computed == existing_hash:
                result.append(line)
            else:
                # Hash mismatch - recompute
                h = compute_line_hash(parsed_content, hash_length)
                result.append(f"{num}:{h}|{parsed_content}")
        else:
            # Add hashline format
            h = compute_line_hash(line, hash_length)
            result.append(f"{i}:{h}|{line}")

    return "\n".join(result)


def strip_hashes(formatted_content: str) -> str:
    """
    Remove hashline prefixes from content.

    Args:
        formatted_content: Content with hashline prefixes

    Returns:
        Clean content without hashes
    """
    lines = formatted_content.split("\n")
    result = []

    for line in lines:
        _, _, content = parse_hashline(line)
        result.append(content)

    return "\n".join(result)


class HashlineEditor:
    """
    Editor for hashline-based file modifications.

    Provides stable, verifiable file editing by referencing lines through
    content hashes rather than line numbers or content reproduction.
    """

    def __init__(self, hash_length: int = 2):
        """
        Initialize the editor.

        Args:
            hash_length: Length of line hashes (2-3 recommended)
        """
        self.hash_length = hash_length

    def read_file(self, path: Path) -> ParsedFile:
        """
        Read a file and parse into hashline format.

        Args:
            path: Path to file

        Returns:
            ParsedFile with hashline information
        """
        content = path.read_text(encoding="utf-8", errors="replace")
        lines = content.split("\n")

        line_infos = []
        for i, line in enumerate(lines, 1):
            # Skip empty final line
            if i == len(lines) and not line:
                continue

            original = line + ("\n" if i < len(lines) else "")
            h = compute_line_hash(line, self.hash_length)

            line_infos.append(LineInfo(
                number=i,
                hash=h,
                content=line,
                original=original
            ))

        return ParsedFile(path=path, lines=line_infos)

    def format_for_display(self, parsed: ParsedFile, offset: int = 1, limit: int = 2000) -> str:
        """
        Format parsed file for display to LLM.

        Args:
            parsed: ParsedFile to format
            offset: Starting line number (1-indexed)
            limit: Maximum lines to include

        Returns:
            Formatted string with hashlines
        """
        start_idx = max(0, offset - 1)
        end_idx = min(len(parsed.lines), start_idx + limit)

        lines = []
        for i in range(start_idx, end_idx):
            line = parsed.lines[i]
            lines.append(f"{line.number}:{line.hash}|{line.content}")

        if end_idx < len(parsed.lines):
            remaining = len(parsed.lines) - end_idx
            lines.append(f"\n[... {remaining} more lines ...]")

        return "\n".join(lines)

    def apply_edit(self, parsed: ParsedFile, operation: EditOperation) -> EditResult:
        """
        Apply a single edit operation to the parsed file.

        Args:
            parsed: ParsedFile to modify
            operation: EditOperation to apply

        Returns:
            EditResult with success status and information
        """
        # Find target line by hash
        target = parsed.find_line(operation.target_hash, operation.target_line)

        if target is None:
            # Check if hash exists anywhere (maybe line number changed)
            all_matches = parsed.hash_to_lines.get(operation.target_hash, [])
            if all_matches:
                return EditResult(
                    success=False,
                    message=f"Hash '{operation.target_hash}' found at lines {all_matches}, "
                            f"but expected at line {operation.target_line}. "
                            f"File may have changed since reading."
                )
            return EditResult(
                success=False,
                message=f"Hash '{operation.target_hash}' not found. File may have changed since reading."
            )

        target_idx = target.number - 1

        if operation.edit_type == EditType.REPLACE:
            if operation.content is None:
                return EditResult(success=False, message="REPLACE requires content")

            # Verify hash still matches
            current_hash = compute_line_hash(parsed.lines[target_idx].content, self.hash_length)
            if current_hash != operation.target_hash:
                return EditResult(
                    success=False,
                    message=f"Hash mismatch at line {target.number}: expected '{operation.target_hash}', "
                            f"found '{current_hash}'. File changed since reading."
                )

            # Replace the line
            new_line = LineInfo(
                number=target.number,
                hash=compute_line_hash(operation.content, self.hash_length),
                content=operation.content,
                original=operation.content + "\n"
            )
            parsed.lines[target_idx] = new_line

            # Rebuild hash index
            parsed.__post_init__()

            return EditResult(
                success=True,
                message=f"Replaced line {target.number}",
                modified_lines=[target.number],
                new_hashes={target.number: new_line.hash}
            )

        elif operation.edit_type == EditType.DELETE:
            # Verify hash
            current_hash = compute_line_hash(parsed.lines[target_idx].content, self.hash_length)
            if current_hash != operation.target_hash:
                return EditResult(
                    success=False,
                    message=f"Hash mismatch at line {target.number}. File changed since reading."
                )

            # Remove the line
            deleted = parsed.lines.pop(target_idx)

            # Renumber remaining lines
            for i, line in enumerate(parsed.lines[target_idx:], target_idx + 1):
                line.number = i

            # Rebuild hash index
            parsed.__post_init__()

            return EditResult(
                success=True,
                message=f"Deleted line {deleted.number}",
                modified_lines=list(range(target.number, len(parsed.lines) + 2))
            )

        elif operation.edit_type == EditType.INSERT_AFTER:
            if operation.content is None:
                return EditResult(success=False, message="INSERT_AFTER requires content")

            # Insert after target
            new_line = LineInfo(
                number=target.number + 1,
                hash=compute_line_hash(operation.content, self.hash_length),
                content=operation.content,
                original=operation.content + "\n"
            )
            parsed.lines.insert(target_idx + 1, new_line)

            # Renumber remaining lines
            for i, line in enumerate(parsed.lines[target_idx + 1:], target_idx + 2):
                line.number = i

            # Rebuild hash index
            parsed.__post_init__()

            return EditResult(
                success=True,
                message=f"Inserted after line {target.number}",
                modified_lines=list(range(target.number + 1, len(parsed.lines) + 1)),
                new_hashes={target.number + 1: new_line.hash}
            )

        elif operation.edit_type == EditType.INSERT_BEFORE:
            if operation.content is None:
                return EditResult(success=False, message="INSERT_BEFORE requires content")

            # Insert before target
            new_line = LineInfo(
                number=target.number,
                hash=compute_line_hash(operation.content, self.hash_length),
                content=operation.content,
                original=operation.content + "\n"
            )
            parsed.lines.insert(target_idx, new_line)

            # Renumber remaining lines
            for i, line in enumerate(parsed.lines[target_idx:], target_idx + 1):
                line.number = i

            # Rebuild hash index
            parsed.__post_init__()

            return EditResult(
                success=True,
                message=f"Inserted before line {target.number + 1}",
                modified_lines=list(range(target.number, len(parsed.lines) + 1)),
                new_hashes={target.number: new_line.hash}
            )

        elif operation.edit_type == EditType.REPLACE_RANGE:
            if operation.content is None or operation.end_hash is None:
                return EditResult(success=False, message="REPLACE_RANGE requires content and end_hash")

            end_line = parsed.find_line(operation.end_hash, operation.end_line)
            if end_line is None:
                return EditResult(
                    success=False,
                    message=f"End hash '{operation.end_hash}' not found"
                )

            end_idx = end_line.number - 1

            # Verify both hashes still match
            current_start_hash = compute_line_hash(parsed.lines[target_idx].content, self.hash_length)
            current_end_hash = compute_line_hash(parsed.lines[end_idx].content, self.hash_length)

            if current_start_hash != operation.target_hash or current_end_hash != operation.end_hash:
                return EditResult(
                    success=False,
                    message="Hash mismatch in range. File changed since reading."
                )

            # Replace range with new content (split by newlines)
            new_lines_content = operation.content.split("\n")
            new_lines = []

            for i, content in enumerate(new_lines_content):
                line_num = target.number + i
                new_lines.append(LineInfo(
                    number=line_num,
                    hash=compute_line_hash(content, self.hash_length),
                    content=content,
                    original=content + "\n"
                ))

            # Replace in list
            parsed.lines[target_idx:end_idx + 1] = new_lines

            # Renumber all lines after insertion
            for i, line in enumerate(parsed.lines[target_idx:], target_idx + 1):
                line.number = i

            # Rebuild hash index
            parsed.__post_init__()

            modified = list(range(target.number, target.number + len(new_lines)))
            hashes = {ln: line.hash for ln, line in zip(modified, new_lines)}

            return EditResult(
                success=True,
                message=f"Replaced lines {target.number}-{end_line.number} with {len(new_lines)} lines",
                modified_lines=modified,
                new_hashes=hashes
            )

        return EditResult(success=False, message=f"Unknown edit type: {operation.edit_type}")

    def write_file(self, parsed: ParsedFile) -> None:
        """
        Write parsed file back to disk.

        Args:
            parsed: ParsedFile to write
        """
        content = "\n".join(line.content for line in parsed.lines)
        if parsed.lines:
            content += "\n"

        parsed.path.write_text(content, encoding="utf-8")

    def batch_edit(self, path: Path, operations: List[EditOperation]) -> Tuple[bool, List[EditResult]]:
        """
        Apply multiple edits to a file atomically.

        Args:
            path: Path to file
            operations: List of EditOperations to apply

        Returns:
            Tuple of (all_succeeded, list_of_results)
        """
        parsed = self.read_file(path)
        results = []
        all_succeeded = True

        for op in operations:
            result = self.apply_edit(parsed, op)
            results.append(result)
            if not result.success:
                all_succeeded = False
                break

        if all_succeeded:
            self.write_file(parsed)

        return all_succeeded, results
