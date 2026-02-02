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
                "description": """The timeout for the command in milliseconds (default: 60000, max: 180000). Timeout guidance:
* 100ms: Immediate commands (cd, ls, echo, cat, pwd, test, [ -f file ])
* 1000-5000ms: Quick commands (grep, find, head, tail, wc, sort, uniq, basic file ops)
* 5000-15000ms: Moderate commands (pip install <small>, npm install <small>, compilation, small scripts)
* 15000-30000ms: Longer operations (package installs, downloads, medium scripts, docker builds)
* 30000-180000ms: Long-running operations (training, large downloads, complex builds)
If an operation requires more than 180000ms, break it down into smaller steps (each command has max timeout of 180000ms)""",
            },
        },
        "required": ["command"],
    },
}

# Read file tool
READ_FILE_SPEC: dict[str, Any] = {
    "name": "read_file",
    "description": """Reads a local file with 1-indexed line numbers.
Returns file content with line numbers in format 'L{number}: {content}'.
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

# Apply patch tool
APPLY_PATCH_SPEC: dict[str, Any] = {
    "name": "apply_patch",
    "description": """Applies file patches to create, update, or delete files.

Patch format:
*** Begin Patch
*** Add File: <path>
+line to add
*** Update File: <path>
@@ context line
-old line
+new line
*** Delete File: <path>
*** End Patch

Rules:
- Use @@ with context to identify where to make changes
- Prefix new lines with + (even for new files)
- Prefix removed lines with -
- Use 3 lines of context before and after changes
- File paths must be relative, never absolute""",
    "parameters": {
        "type": "object",
        "properties": {
            "patch": {
                "type": "string",
                "description": "The patch content following the format described above",
            },
        },
        "required": ["patch"],
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

TRANSCRIPT_SPEC: dict[str, Any] = {
    "name": "transcript",
    "description": """Analyze video content.

This tool uploads the video and asks it to analyze/transcribe based on your instruction.

**CRITICAL: Be VERY PRECISE in your instruction!**
- What exactly you want extracted (dialogue, text on screen, actions, inputs, etc.)
- The format you need (list, transcript, step-by-step, etc.)
- Any specific details to focus on
- The purpose/goal of the transcription

**Good instruction examples:**
- "Extract all text items shown on screen in this video. Output as a list of items, one per line."
- "Extract ALL text inputs shown on screen in the exact order they appear. List each input on a separate line. Include every input: single characters, multi-word items, items with parameters, and sequences. Output only the inputs, one per line, no explanations."
- "Transcribe all spoken dialogue in this video, with speaker labels if possible."
- "List all the steps shown in this tutorial, in order."
- "Extract the code snippets visible on screen in this programming tutorial."

**Bad instruction examples:**
- "Transcribe this" (too vague)
- "What's in this video?" (not specific enough)
- "Get the items" (not specific about format or completeness)

Supports: YouTube, Twitter/X, TikTok, Vimeo, and direct video URLs.

Returns the analysis/transcription based on your instruction.""",
    "parameters": {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "The video URL (YouTube, Twitter, TikTok, Vimeo, or direct video URL)",
            },
            "instruction": {
                "type": "string",
                "description": "REQUIRED: Precise instruction for what to extract/transcribe from the video. Be specific about format, content type, and purpose.",
            },
        },
        "required": ["url", "instruction"],
        "additionalProperties": False,
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

# All tool specs
TOOL_SPECS: dict[str, dict[str, Any]] = {
    "shell_command": SHELL_COMMAND_SPEC,
    "read_file": READ_FILE_SPEC,
    "write_file": WRITE_FILE_SPEC,
    "list_dir": LIST_DIR_SPEC,
    "grep_files": GREP_FILES_SPEC,
    "view_image": VIEW_IMAGE_SPEC,
    "update_plan": UPDATE_PLAN_SPEC,
    "web_search": WEB_SEARCH_SPEC,
    "transcript": TRANSCRIPT_SPEC,
    "spawn_process": SPAWN_PROCESS_SPEC,
    "kill_process": KILL_PROCESS_SPEC,
    "wait_for_port": WAIT_FOR_PORT_SPEC,
    "wait_for_file": WAIT_FOR_FILE_SPEC,
    "run_until_file": RUN_UNTIL_FILE_SPEC,
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