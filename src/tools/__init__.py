"""Tools module - registry and tool implementations."""

# Individual tools
from src.tools.base import BaseTool, ToolMetadata, ToolResult
from src.tools.registry import (
    CachedResult,
    ExecutorConfig,
    ExecutorStats,
    ToolRegistry,
    ToolStats,
)
from src.tools.search_files import SearchFilesTool
from src.tools.specs import TOOL_SPECS, get_all_tools, get_tool_spec
from src.tools.web_search import web_search
from src.tools.extract_video import extract_video_frames, extract_keyframes
from src.tools.analyze_image import analyze_image

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
    "SearchFilesTool",
    "analyze_image",
    "extract_video_frames",
    "extract_keyframes",
    "web_search",
]
