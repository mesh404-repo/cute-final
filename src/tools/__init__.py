"""Tools module - registry and tool implementations."""

# Individual tools
from src.tools.apply_patch import ApplyPatchTool
from src.tools.base import BaseTool, ToolMetadata, ToolResult
from src.tools.registry import (
    CachedResult,
    ExecutorConfig,
    ExecutorStats,
    ToolRegistry,
    ToolStats,
)
from src.tools.specs import TOOL_SPECS, get_all_tools, get_tool_spec
from src.tools.view_image import view_image
from src.tools.web_search import web_search

__all__ = [
    # Base
    "ToolResult",
    "BaseTool",
    "ToolMetadata",
    # Registry
    "ToolRegistry",
    "ExecutorConfig",
    "ExecutorStats",
    "ToolStats",
    "CachedResult",
    # Specs
    "get_all_tools",
    "get_tool_spec",
    "TOOL_SPECS",
    # Tools
    "ApplyPatchTool",
    "view_image",
    "web_search",
]
