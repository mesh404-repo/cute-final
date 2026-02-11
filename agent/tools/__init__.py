"""Tools module - registry and tool implementations."""

# Individual tools
from agent.tools.base import BaseTool, ToolMetadata, ToolResult
from agent.tools.list_dir import ListDirTool
from agent.tools.read_file import ReadFileTool
from agent.tools.registry import (
    CachedResult,
    ExecutorConfig,
    ExecutorStats,
    ToolRegistry,
    ToolStats,
)
from agent.tools.search_files import SearchFilesTool
from agent.tools.specs import TOOL_SPECS, get_all_tools, get_tool_spec
from agent.tools.write_file import WriteFileTool
from agent.tools.web_search import web_search
from agent.tools.extract_video import extract_video_frames, extract_keyframes
from agent.tools.analyze_image import analyze_image
from agent.tools.finish import execute_finish

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
    "ReadFileTool",
    "WriteFileTool",
    "ListDirTool",
    "SearchFilesTool",
    "analyze_image",
    "extract_video_frames",
    "extract_keyframes",
    "web_search",
    "execute_finish",
]
