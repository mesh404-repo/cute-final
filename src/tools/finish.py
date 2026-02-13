"""Finish tool – explicit task completion.

Allows the model to signal task completion with a brief summary.
"""

from __future__ import annotations

from src.tools.base import ToolResult


def execute_finish(summary: str) -> ToolResult:
    """Signal that the task is complete.

    Args:
        summary: Brief summary of what was done and how it was verified.

    Returns:
        ToolResult with success and the summary in output.
    """
    if not summary or not str(summary).strip():
        return ToolResult.invalid(
            "Missing required parameter 'summary'. "
            "Usage: finish(summary: str)"
        )
    return ToolResult.ok(
        f"Task marked complete. Summary: {summary.strip()}",
        data={"ok": True, "summary": summary.strip()},
    )
