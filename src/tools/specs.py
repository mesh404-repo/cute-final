"""Tool specifications for SuperAgent - defines JSON schemas for all tools."""

from __future__ import annotations

from typing import Any

# Shell command tool
SHELL_COMMAND_SPEC: dict[str, Any] = {
    "name": "shell_command",
    "description": """Runs a shell command and returns its output.
Always set the `workdir` param when using this tool. Do not use `cd` unless absolutely necessary.
Use `rg` (ripgrep) for searching text or files as it's much faster than grep.""",
    "parameters": {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "The shell command to execute",
            },
            "workdir": {
                "type": "string",
                "description": "The working directory to execute the command in",
            },
            "timeout_ms": {
                "type": "number",
                "description": "The timeout for the command in milliseconds",
            },
        },
        "required": ["command"],
    },
}

# Read file tool
READ_FILE_SPEC: dict[str, Any] = {
    "name": "read_file",
    "description": """Reads a local file with hashline format for direct editing compatibility.
Returns file content in format '{line_number}:{hash}|{content}' (e.g. '12:a3|def hello():').
The hash can be used directly as target_hash in hashline_edit — no separate hashline_edit(read) needed.
Supports reading specific ranges with offset and limit parameters.""",
    "parameters": {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "Absolute or relative path to the file",
            },
            "offset": {
                "type": "number",
                "description": "The line number to start reading from (1-indexed, default: 1)",
            },
            "limit": {
                "type": "number",
                "description": "The maximum number of lines to return (default: 2000)",
            },
        },
        "required": ["file_path"],
    },
}

# List directory tool
LIST_DIR_SPEC: dict[str, Any] = {
    "name": "list_dir",
    "description": """Lists entries in a local directory with type indicators.
Directories are marked with '/', symlinks with '@'.
Supports recursive listing with configurable depth.""",
    "parameters": {
        "type": "object",
        "properties": {
            "dir_path": {
                "type": "string",
                "description": "Absolute or relative path to the directory to list",
            },
            "offset": {
                "type": "number",
                "description": "The entry number to start listing from (1-indexed, default: 1)",
            },
            "limit": {
                "type": "number",
                "description": "The maximum number of entries to return (default: 50)",
            },
            "depth": {
                "type": "number",
                "description": "The maximum directory depth to traverse (default: 2)",
            },
        },
        "required": ["dir_path"],
    },
}

# Grep files tool
GREP_FILES_SPEC: dict[str, Any] = {
    "name": "grep_files",
    "description": """Finds files whose contents match the pattern.
Uses ripgrep (rg) for fast searching.
Returns file paths sorted by modification time.""",
    "parameters": {
        "type": "object",
        "properties": {
            "pattern": {
                "type": "string",
                "description": "Regular expression pattern to search for",
            },
            "include": {
                "type": "string",
                "description": "Optional glob to filter which files are searched (e.g., '*.py', '*.{ts,tsx}')",
            },
            "path": {
                "type": "string",
                "description": "Directory or file path to search. Defaults to working directory.",
            },
            "limit": {
                "type": "number",
                "description": "Maximum number of file paths to return (default: 100)",
            },
        },
        "required": ["pattern"],
    },
}

# View image tool
VIEW_IMAGE_SPEC: dict[str, Any] = {
    "name": "view_image",
    "description": """View a local image from the filesystem for visual analysis.
Use this when you need to interpret visual content: (1) images the user points to, or (2) images you generate from data (e.g. rendering coordinates, toolpaths, or geometric data to a bitmap) to read text or shapes—write the image to a file (e.g. PPM with Python stdlib, or PNG/JPEG if available) then call view_image with that path.
Supported formats: PNG, JPEG, GIF, WebP, BMP, PPM.""",
    "parameters": {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Local filesystem path to the image file",
            },
        },
        "required": ["path"],
    },
}

# Write file tool
WRITE_FILE_SPEC: dict[str, Any] = {
    "name": "write_file",
    "description": """Write content to a file.
Creates the file if it doesn't exist, or overwrites if it does.
Parent directories are created automatically.""",
    "parameters": {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "Path to the file to write",
            },
            "content": {
                "type": "string",
                "description": "Content to write to the file",
            },
        },
        "required": ["file_path", "content"],
    },
}

# Finish tool – explicit task completion with summary
FINISH_SPEC: dict[str, Any] = {
    "name": "finish",
    "description": "Signal that the task is complete. Call this when all work is done and verified. Provide a brief completion summary.",
    "parameters": {
        "type": "object",
        "properties": {
            "summary": {
                "type": "string",
                "description": "Brief summary of what was done and how it was verified.",
            },
        },
        "required": ["summary"],
        "additionalProperties": False,
    },
}

# Update plan tool
UPDATE_PLAN_SPEC: dict[str, Any] = {
    "name": "update_plan",
    "description": """Updates the task plan to track progress.
Use this to show the user your planned steps and mark them as completed.
Each step should be 5-7 words maximum.""",
    "parameters": {
        "type": "object",
        "properties": {
            "steps": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "description": {
                            "type": "string",
                            "description": "Short description of the step (5-7 words)",
                        },
                        "status": {
                            "type": "string",
                            "enum": ["pending", "in_progress", "completed"],
                            "description": "Current status of the step",
                        },
                    },
                    "required": ["description", "status"],
                },
                "description": "List of plan steps with their status",
            },
            "explanation": {
                "type": "string",
                "description": "Optional explanation of why the plan changed",
            },
        },
        "required": ["steps"],
    },
}

# Web search tool
WEB_SEARCH_SPEC: dict[str, Any] = {
    "name": "web_search",
    "description": """Search the web for information, documentation, code examples, and solutions to help solve tasks.

Use web search when:
- You encounter unfamiliar technologies, libraries, frameworks, or APIs
- You're stuck on a problem and need to find solutions or examples
- You need documentation, tutorials, or code examples
- You need to research how to accomplish a specific task
- You're working with open source projects and need to understand patterns or best practices

Be specific in queries: include library names, error messages, specific concepts, or task descriptions. Use search_type='code' for code examples, 'docs' for documentation, or 'general' for broad searches.""",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query - be specific and include relevant keywords, library names, or task descriptions",
            },
            "num_results": {
                "type": "number",
                "description": "Number of results to return (default: 5, max: 10)",
            },
            "search_type": {
                "type": "string",
                "enum": ["general", "code", "docs", "news", "images"],
                "description": "Type of search: 'general' for broad searches, 'code' for code examples/GitHub/Stack Overflow, 'docs' for documentation/tutorials, 'news' for recent news, 'images' for image search",
            },
        },
        "required": ["query"],
    },
}

# Spawn process tool
SPAWN_PROCESS_SPEC: dict[str, Any] = {
    "name": "spawn_process",
    "description": "Start a long-running process in the background. Returns PID and log paths. Use wait_for_port to confirm the service is up.",
    "parameters": {
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "Command to run (via bash -lc)."},
            "cwd": {"type": "string", "description": "Working directory (default: workspace)."},
            "stdout_path": {"type": "string", "description": "File for stdout (default: auto in /tmp)."},
            "stderr_path": {"type": "string", "description": "File for stderr (default: auto in /tmp)."},
        },
        "required": ["command"],
    },
}

# Kill process tool
KILL_PROCESS_SPEC: dict[str, Any] = {
    "name": "kill_process",
    "description": "Terminate a process by PID. Use TERM first; use KILL if needed.",
    "parameters": {
        "type": "object",
        "properties": {
            "pid": {"type": "integer", "description": "Process ID to terminate."},
            "signal": {"type": "string", "enum": ["TERM", "KILL"], "description": "Signal (default TERM)."},
        },
        "required": ["pid"],
    },
}

# Wait for port tool
WAIT_FOR_PORT_SPEC: dict[str, Any] = {
    "name": "wait_for_port",
    "description": "Wait until a TCP host:port is accepting connections (or timeout). Use after spawn_process to ensure server is up.",
    "parameters": {
        "type": "object",
        "properties": {
            "host": {"type": "string", "description": "Hostname or IP (default 127.0.0.1)."},
            "port": {"type": "integer", "description": "TCP port to check."},
            "timeout_sec": {"type": "number", "description": "Timeout in seconds (default 15)."},
        },
        "required": ["port"],
    },
}

# Wait for file tool
WAIT_FOR_FILE_SPEC: dict[str, Any] = {
    "name": "wait_for_file",
    "description": "Wait until a filesystem path exists (or timeout). Useful when a program generates an output artifact asynchronously.",
    "parameters": {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path to wait for."},
            "timeout_sec": {"type": "number", "description": "Timeout seconds (default 15)."},
            "min_size_bytes": {"type": "integer", "description": "Require size >= this many bytes (default 0)."},
        },
        "required": ["path"],
    },
}

# Run until file tool
RUN_UNTIL_FILE_SPEC: dict[str, Any] = {
    "name": "run_until_file",
    "description": "Run a command until a target file exists (or timeout), then terminate. Useful for interactive programs that render a frame/file as readiness.",
    "parameters": {
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "Command to run (via bash -lc)."},
            "file_path": {"type": "string", "description": "File to wait for."},
            "cwd": {"type": "string", "description": "Working directory (default: workspace)."},
            "timeout_sec": {"type": "number", "description": "Timeout seconds (default 30)."},
            "min_size_bytes": {"type": "integer", "description": "Require file size >= this (default 1)."},
        },
        "required": ["command", "file_path"],
    },
}

# Extract video frames tool
EXTRACT_VIDEO_FRAMES_SPEC: dict[str, Any] = {
    "name": "extract_video_frames",
    "description": """Extract frames from a video file at regular intervals.

Use this to analyze video content by extracting frames that can then be analyzed with view_image.
Useful for: game footage analysis, tutorial extraction, action detection, etc.

The frames are saved as images in the specified output directory.
After extraction, use view_image on individual frames to analyze them.""",
    "parameters": {
        "type": "object",
        "properties": {
            "video_path": {
                "type": "string",
                "description": "Path to the video file (mp4, avi, mkv, mov, webm)",
            },
            "output_dir": {
                "type": "string",
                "description": "Directory to save extracted frames",
            },
            "fps": {
                "type": "number",
                "description": "Frames per second to extract (default: 1.0). Lower = fewer frames.",
            },
            "max_frames": {
                "type": "integer",
                "description": "Maximum number of frames to extract (default: 30)",
            },
            "start_time": {
                "type": "number",
                "description": "Start time in seconds (optional)",
            },
            "end_time": {
                "type": "number",
                "description": "End time in seconds (optional)",
            },
            "scale": {
                "type": "string",
                "description": "Output scale, e.g., '640:-1' for 640px width with auto height (optional)",
            },
        },
        "required": ["video_path", "output_dir"],
    },
}

# Extract keyframes tool
EXTRACT_KEYFRAMES_SPEC: dict[str, Any] = {
    "name": "extract_keyframes",
    "description": """Extract keyframes (scene changes) from a video.

More efficient than fixed-interval extraction - only extracts frames where
significant visual changes occur. Good for understanding video structure
and key moments.

After extraction, use view_image on individual keyframes to analyze them.""",
    "parameters": {
        "type": "object",
        "properties": {
            "video_path": {
                "type": "string",
                "description": "Path to the video file",
            },
            "output_dir": {
                "type": "string",
                "description": "Directory to save extracted keyframes",
            },
            "max_frames": {
                "type": "integer",
                "description": "Maximum keyframes to extract (default: 20)",
            },
            "threshold": {
                "type": "number",
                "description": "Scene change threshold 0.0-1.0 (default: 0.3). Lower = more frames.",
            },
        },
        "required": ["video_path", "output_dir"],
    },
}


# Hashline edit tool - content-addressable line editing
HASHLINE_EDIT_SPEC: dict[str, Any] = {
    "name": "hashline_edit",
    "description": """Edit files using content-addressable line hashes for stable, verifiable modifications.

Each line is tagged with a short hash (2-3 chars) like "1:a3|function hello() {".
This provides stable identifiers - reference lines by hash, not content or line numbers.

If the file changes since reading, the hash won't match and the edit is rejected.
This prevents corruption from concurrent modifications or stale context.

Operations:
- read: Read file with hashline format (e.g., "1:a3|content")
- replace: Replace line by hash ("replace line 2:f1 with 'new content'")
- insert_after: Insert after line with given hash
- insert_before: Insert before line with given hash
- delete: Delete line by hash
- replace_range: Replace lines from target_hash to end_hash
- batch: Apply multiple edits atomically (all succeed or none)

Example:
  Original: "1:a3|def hello():\n2:f1|  return 'world'\n3:0e|"
  Edit: replace line "2:f1" with "  return 'universe'"
  Result: "1:a3|def hello():\n2:b2|  return 'universe'\n3:0e|"

Benefits:
- No need to reproduce content as "anchor" - just use the hash
- Detects file changes automatically (hash mismatch)
- More reliable than line numbers (which shift)
- More precise than fuzzy matching""",
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
                "description": "Hash of the target line (2-3 chars from hashline format)",
            },
            "target_line": {
                "type": "number",
                "description": "Optional line number hint for disambiguation when multiple lines have same hash",
            },
            "end_hash": {
                "type": "string",
                "description": "For replace_range: hash of the end line",
            },
            "end_line": {
                "type": "number",
                "description": "For replace_range: end line number hint",
            },
            "content": {
                "type": "string",
                "description": "New content for replace/insert operations",
            },
            "offset": {
                "type": "number",
                "description": "For read: starting line number (1-indexed, default: 1)",
            },
            "limit": {
                "type": "number",
                "description": "For read: maximum lines to return (default: 2000)",
            },
            "edits": {
                "type": "array",
                "description": "For batch: list of edit operations to apply atomically",
                "items": {"type": "object"},
            },
        },
        "required": ["operation", "file_path"],
    },
}


# String replace tool - find and replace exact strings in files (no prior read needed)
STR_REPLACE_SPEC: dict[str, Any] = {
    "name": "str_replace",
    "description": """Find and replace an exact string in a file. No need to read the file first.

Use this when you know the exact text to find and replace — e.g., from a refactor plan,
error message, or known pattern. This is faster than hashline_edit because it does NOT
require reading the file first to get line hashes.

The old_str must match EXACTLY (including whitespace and indentation).
If old_str appears multiple times, only the FIRST occurrence is replaced (unless replace_all=true).
If old_str is not found, the tool returns an error.

Prefer this tool over hashline_edit when:
- You already know the exact string to find (from a plan, error, or task description)
- You want to skip the read-file step
- You're doing simple find-and-replace operations

Use hashline_edit instead when:
- You need to edit by line number/hash after reading
- You need insert_before/insert_after/delete operations""",
    "parameters": {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "Path to the file to edit",
            },
            "old_str": {
                "type": "string",
                "description": "Exact string to find in the file",
            },
            "new_str": {
                "type": "string",
                "description": "String to replace it with",
            },
            "replace_all": {
                "type": "boolean",
                "description": "If true, replace ALL occurrences (default: false, replaces first only)",
            },
        },
        "required": ["file_path", "old_str", "new_str"],
    },
}

# All tool specs
TOOL_SPECS: dict[str, dict[str, Any]] = {
    "shell_command": SHELL_COMMAND_SPEC,
    "read_file": READ_FILE_SPEC,
    "write_file": WRITE_FILE_SPEC,
    "list_dir": LIST_DIR_SPEC,
    "grep_files": GREP_FILES_SPEC,
    "view_image": VIEW_IMAGE_SPEC,
    "finish": FINISH_SPEC,
    "update_plan": UPDATE_PLAN_SPEC,
    "web_search": WEB_SEARCH_SPEC,
    "extract_video_frames": EXTRACT_VIDEO_FRAMES_SPEC,
    "extract_keyframes": EXTRACT_KEYFRAMES_SPEC,
    "spawn_process": SPAWN_PROCESS_SPEC,
    "kill_process": KILL_PROCESS_SPEC,
    "wait_for_port": WAIT_FOR_PORT_SPEC,
    "wait_for_file": WAIT_FOR_FILE_SPEC,
    "run_until_file": RUN_UNTIL_FILE_SPEC,
    "hashline_edit": HASHLINE_EDIT_SPEC,
    "str_replace": STR_REPLACE_SPEC,
 }


def get_all_tools() -> list[dict[str, Any]]:
    """Get all tool specifications as a list.
    
    Returns:
        List of tool specification dicts
    """
    return list(TOOL_SPECS.values())


def get_tool_spec(name: str) -> dict[str, Any] | None:
    """Get a specific tool specification.
    
    Args:
        name: Name of the tool
        
    Returns:
        Tool specification dict or None if not found
    """
    return TOOL_SPECS.get(name)