"""Tool registry for SuperAgent - manages and dispatches tool calls."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, TYPE_CHECKING

from src.tools.base import ToolResult
from src.tools.specs import get_all_tools

if TYPE_CHECKING:
    pass  # AgentContext is duck-typed (has shell(), cwd, etc.)


@dataclass
class ExecutorConfig:
    """Configuration for tool execution."""
    max_concurrent: int = 4
    default_timeout: float = 120.0
    cache_enabled: bool = True
    cache_ttl: float = 300.0  # 5 minutes


@dataclass
class CachedResult:
    """A cached tool result with timestamp."""
    result: ToolResult
    cached_at: float  # timestamp from time.time()
    
    def is_valid(self, ttl: float) -> bool:
        """Check if the cached result is still valid."""
        return (time.time() - self.cached_at) < ttl


@dataclass
class ToolStats:
    """Per-tool execution statistics."""
    executions: int = 0
    successes: int = 0
    total_ms: int = 0
    
    def success_rate(self) -> float:
        """Get the success rate for this tool."""
        if self.executions == 0:
            return 0.0
        return self.successes / self.executions
    
    def avg_ms(self) -> float:
        """Get average execution time in milliseconds."""
        if self.executions == 0:
            return 0.0
        return self.total_ms / self.executions


@dataclass
class ExecutorStats:
    """Aggregate execution statistics."""
    total_executions: int = 0
    successful_executions: int = 0
    failed_executions: int = 0
    cache_hits: int = 0
    total_duration_ms: int = 0
    by_tool: Dict[str, ToolStats] = field(default_factory=dict)
    
    def success_rate(self) -> float:
        """Get overall success rate."""
        if self.total_executions == 0:
            return 0.0
        return self.successful_executions / self.total_executions
    
    def cache_hit_rate(self) -> float:
        """Get cache hit rate."""
        if self.total_executions == 0:
            return 0.0
        return self.cache_hits / self.total_executions
    
    def avg_duration_ms(self) -> float:
        """Get average execution duration in milliseconds."""
        if self.total_executions == 0:
            return 0.0
        return self.total_duration_ms / self.total_executions


class ToolRegistry:
    """Registry for managing and dispatching tool calls.
    
    Tools receive AgentContext for shell execution.
    Includes caching and execution statistics.
    """

    def __init__(
        self,
        cwd: Optional[Path] = None,
        config: Optional[ExecutorConfig] = None,
    ):
        """Initialize the registry.
        
        Args:
            cwd: Current working directory for tools (optional, can be set later)
            config: Executor configuration (optional, uses defaults)
        """
        self.cwd = cwd or Path("/app")
        self._plan: list[dict[str, str]] = []
        self._config = config or ExecutorConfig()
        self._cache: Dict[str, CachedResult] = {}
        self._stats = ExecutorStats()
        self._process_runner: Optional[Any] = None  # ProcessToolRunner, lazy init

    def _get_process_runner(self) -> Any:
        """Lazy init ProcessToolRunner for spawn_process / kill_process."""
        if self._process_runner is None:
            from src.tools.process import ProcessToolRunner
            self._process_runner = ProcessToolRunner()
        return self._process_runner
    
    def execute(
        self,
        ctx: "AgentContext",
        name: str,
        arguments: dict[str, Any],
    ) -> ToolResult:
        """Execute a tool by name.
        
        Args:
            ctx: Agent context with shell() method
            name: Tool name
            arguments: Tool arguments
            
        Returns:
            ToolResult from the tool execution
        """
        start_time = time.time()
        
        # Check cache first if enabled
        if self._config.cache_enabled:
            cache_key = self._cache_key(name, arguments)
            cached = self._get_cached(cache_key)
            if cached is not None:
                duration_ms = int((time.time() - start_time) * 1000)
                self._record_execution(name, duration_ms, success=True, cached=True)
                return cached
        
        cwd = Path(ctx.cwd) if hasattr(ctx, 'cwd') else self.cwd
        
        try:
            if name == "shell_command":
                result = self._run_shell(cwd, arguments)
            elif name == "read_file":
                result = self._run_read_file(cwd, arguments)
            elif name == "write_file":
                result = self._run_write_file(cwd, arguments)
            elif name == "str_replace":
                result = self._run_str_replace(cwd, arguments)
            elif name == "hashline_edit":
                result = self._run_hashline_edit(cwd, arguments)
            elif name == "list_dir":
                result = self._run_list_dir(cwd, arguments)
            elif name == "grep_files":
                result = self._run_grep(ctx, cwd, arguments)
            elif name == "view_image":
                result = self._run_view_image(cwd, arguments)
            elif name == "extract_video_frames":
                result = self._run_extract_video_frames(cwd, arguments)
            elif name == "extract_keyframes":
                result = self._run_extract_keyframes(cwd, arguments)
            elif name == "finish":
                result = self._run_finish(arguments)
            elif name == "update_plan":
                result = self._run_update_plan(arguments)
            elif name == "web_search":
                result = self._run_web_search(arguments)
            elif name == "spawn_process":
                result = self._run_spawn_process(cwd, arguments)
            elif name == "kill_process":
                result = self._run_kill_process(arguments)
            elif name == "wait_for_port":
                result = self._run_wait_for_port(arguments)
            elif name == "wait_for_file":
                result = self._run_wait_for_file(arguments)
            elif name == "run_until_file":
                result = self._run_run_until_file(cwd, arguments)   
            else:
                result = ToolResult.fail(f"Unknown tool: {name}")
                
        except Exception as e:
            result = ToolResult.fail(f"Tool {name} failed: {e}")
        
        # Record execution stats
        duration_ms = int((time.time() - start_time) * 1000)
        self._record_execution(name, duration_ms, success=result.success, cached=False)
        
        # Cache successful results
        if self._config.cache_enabled and result.success:
            cache_key = self._cache_key(name, arguments)
            self._cache_result(cache_key, result)
        
        return result
    

    def _run_hashline_edit(self, cwd: Path, args: dict[str, Any]) -> ToolResult:
        """Execute hashline edit operation."""
        from src.tools.hashline.tool import HashlineEditTool

        tool = HashlineEditTool(cwd)
        return tool.execute(**args)
    
    def _run_shell(
        self,        
        cwd: Path,
        args: dict[str, Any],
    ) -> ToolResult:
        """Execute shell command using subprocess directly."""
        command = args.get("command", "")
        workdir = args.get("workdir")
        timeout_ms = args.get("timeout_ms", 60000)
        
        if not command:
            return ToolResult.invalid(
                "Missing required parameter 'command'. "
                "Usage: shell_command(command: str, workdir?: str, timeout_ms?: int)"
            )
        
        # Resolve working directory
        effective_cwd = cwd
        if workdir:
            wd = Path(workdir)
            effective_cwd = wd if wd.is_absolute() else cwd / wd
                    
        timeout_sec = max(1, timeout_ms // 1000)

        # Build environment with standard system directories guaranteed in PATH.
        # Container environments often have minimal PATH that excludes /usr/bin,
        # /usr/sbin etc., causing "command not found" for newly installed packages.
        shell_env = {**os.environ, "TERM": "dumb"}
        current_path = shell_env.get("PATH", "")
        path_parts = current_path.split(":") if current_path else []
        for d in ("/usr/local/sbin", "/usr/local/bin", "/usr/sbin", "/usr/bin", "/sbin", "/bin"):
            if d not in path_parts:
                path_parts.append(d)
        shell_env["PATH"] = ":".join(path_parts)

        try:
            result = subprocess.run(
                ["sh", "-lc", command],
                cwd=str(effective_cwd),
                capture_output=True,
                text=True,
                timeout=timeout_sec,
                env=shell_env,
            )
            
            output_parts = []
            
            if result.stdout:
                stdout = result.stdout                
                output_parts.append(stdout)
            
            if result.stderr:
                stderr = result.stderr                
                if output_parts:
                    output_parts.append(f"\nstderr:\n{stderr}")
                else:
                    output_parts.append(stderr)
            
            output = "".join(output_parts).strip()
            
            # Add exit code info if non-zero
            if result.returncode != 0:
                output = f"{output}\n\nExit code: {result.returncode}" if output else f"Exit code: {result.returncode}"
            
            if not output:
                output = "(no output)"
            
            # Return result based on exit code
            if result.returncode == 0:
                return ToolResult.ok(output)
            else:
                return ToolResult.ok(output)
            
        except subprocess.TimeoutExpired as e:
            partial_output = ""
            if e.stdout:
                partial_output = e.stdout if isinstance(e.stdout, str) else e.stdout.decode("utf-8", errors="replace")
            if e.stderr:
                stderr = e.stderr if isinstance(e.stderr, str) else e.stderr.decode("utf-8", errors="replace")
                partial_output = f"{partial_output}\nstderr:\n{stderr}" if partial_output else stderr
            
            if not partial_output:
                partial_output = "(no output before timeout)"

            return ToolResult(
                success=True,
                output=f"""Command timed out after {timeout_sec}s.
                
The command may still be running in the background. Consider: 
1. Check if the process is still running: `ps aux | grep <process>` 
2. Increase timeout if the operation legitimately needs more time 
3. Check if the command is waiting for input (use -y flags, heredocs, etc.) 
4. Break the command into smaller steps

Partial output before timeout:
{partial_output}
""",
            )
        
        except PermissionError:
            return ToolResult.fail(f"Permission denied executing: {command}")
            
        except Exception as e:
            return ToolResult.fail(f"Command failed: {str(e)}")
    
    def _run_read_file(self, cwd: Path, args: dict[str, Any]) -> ToolResult:
        """Read file contents with hashline format for direct editing compatibility."""
        file_path = args.get("file_path", "")
        offset = args.get("offset", 1)
        limit = args.get("limit", 2000)
        
        if not file_path:
            return ToolResult.invalid(
                "Missing required parameter 'file_path'. "
                "Usage: read_file(file_path: str, offset?: int, limit?: int)"
            )
        
        path = Path(file_path)
        if not path.is_absolute():
            path = cwd / path
        
        if not path.exists():
            return ToolResult.fail(f"File not found: {path}")
        
        if not path.is_file():
            return ToolResult.fail(f"Not a file: {path}")
        
        try:
            from src.tools.hashline.core import compute_line_hash
            
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                lines = f.readlines()
            
            # Apply offset and limit (1-indexed)
            start = max(0, offset - 1)
            end = start + limit
            selected = lines[start:end]
            
            # Format with hashline format: {line_number}:{hash}|{content}
            # Hashes can be used directly with hashline_edit for editing
            output_lines = []
            for i, line in enumerate(selected, start=start + 1):
                content = line.rstrip()
                h = compute_line_hash(content, 2)
                output_lines.append(f"{i}:{h}|{content}")
            
            output = "\n".join(output_lines)
            
            if len(lines) > end:
                output += f"\n\n[... {len(lines) - end} more lines ...]"
            
            return ToolResult.ok(output)
            
        except Exception as e:
            return ToolResult.fail(f"Failed to read file: {e}")    

    def _run_str_replace(self, cwd: Path, args: dict[str, Any]) -> ToolResult:
        """Execute exact string find-and-replace in a file (no prior read needed)."""
        file_path = args.get("file_path", "")
        old_str = args.get("old_str")
        new_str = args.get("new_str")
        replace_all = args.get("replace_all", False)

        missing = []
        if not file_path:
            missing.append("file_path")
        if old_str is None:
            missing.append("old_str")
        if new_str is None:
            missing.append("new_str")
        if missing:
            return ToolResult.invalid(
                f"Missing required parameter(s): {', '.join(missing)}. "
                "Usage: str_replace(file_path: str, old_str: str, new_str: str, replace_all?: bool)"
            )

        path = Path(file_path)
        if not path.is_absolute():
            path = cwd / path

        if not path.exists():
            return ToolResult.fail(f"File not found: {path}")
        if not path.is_file():
            return ToolResult.fail(f"Not a file: {path}")

        try:
            content = path.read_text(encoding="utf-8")

            if old_str not in content:
                # Provide helpful context: show first 200 chars of file
                preview = content[:200].replace("\n", "\\n")
                return ToolResult.fail(
                    f"String not found in {path}. "
                    f"File starts with: {preview}..."
                )

            if replace_all:
                count = content.count(old_str)
                new_content = content.replace(old_str, new_str)
                path.write_text(new_content, encoding="utf-8")
                context_preview = self._str_replace_context(new_content, new_str)
                return ToolResult.ok(
                    f"Replaced {count} occurrence(s) in {path}\n{context_preview}"
                )
            else:
                count = content.count(old_str)
                new_content = content.replace(old_str, new_str, 1)
                path.write_text(new_content, encoding="utf-8")
                suffix = f" ({count} total occurrences, replaced first only)" if count > 1 else ""
                context_preview = self._str_replace_context(new_content, new_str)
                return ToolResult.ok(
                    f"Replaced 1 occurrence in {path}{suffix}\n{context_preview}"
                )

        except Exception as e:
            return ToolResult.fail(f"Failed to edit file: {e}")
        
    def _run_write_file(self, cwd: Path, args: dict[str, Any]) -> ToolResult:
        """Write content to a file."""
        file_path = args.get("file_path", "")
        content = args.get("content")
        
        missing_params = []
        if not file_path:
            missing_params.append("file_path")
        if content is None:
            missing_params.append("content")
        
        if missing_params:
            return ToolResult.invalid(
                f"Missing required parameter(s): {', '.join(missing_params)}. "
                "Usage: write_file(file_path: str, content: str)"
            )
        
        path = Path(file_path)
        if not path.is_absolute():
            path = cwd / path
        
        try:
            # Ensure parent directory exists
            path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
            
            return ToolResult.ok(f"Wrote {len(content)} bytes to {path}")
            
        except Exception as e:
            return ToolResult.fail(f"Failed to write file: {e}")
    
    def _run_list_dir(self, cwd: Path, args: dict[str, Any]) -> ToolResult:
        """List directory contents."""
        dir_path = args.get("dir_path", ".")
        depth = args.get("depth", 2)
        limit = args.get("limit", 50)
        
        path = Path(dir_path)
        if not path.is_absolute():
            path = cwd / path
        
        if not path.exists():
            return ToolResult.fail(f"Directory not found: {path}")
        
        if not path.is_dir():
            return ToolResult.fail(f"Not a directory: {path}")
        
        try:
            entries = []
            self._list_recursive(path, path, entries, depth, limit)
            
            if not entries:
                return ToolResult.ok("(empty directory)")
            
            output = "\n".join(entries[:limit])
            if len(entries) > limit:
                output += f"\n\n[... {len(entries) - limit} more entries ...]"
            
            return ToolResult.ok(output)
            
        except Exception as e:
            return ToolResult.fail(f"Failed to list directory: {e}")
    
    def _list_recursive(
        self,
        base: Path,
        current: Path,
        entries: list,
        max_depth: int,
        max_entries: int,
        current_depth: int = 0,
    ) -> None:
        """Recursively list directory contents."""
        if current_depth > max_depth or len(entries) >= max_entries:
            return
        
        try:
            items = sorted(current.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower()))
            
            for item in items:
                if len(entries) >= max_entries:
                    break
                
                rel_path = item.relative_to(base)
                
                if item.is_dir():
                    entries.append(f"{rel_path}/")
                    self._list_recursive(base, item, entries, max_depth, max_entries, current_depth + 1)
                elif item.is_symlink():
                    entries.append(f"{rel_path}@")
                else:
                    entries.append(str(rel_path))
                    
        except PermissionError:
            pass
    
    def _run_grep(
        self,
        ctx: "AgentContext",
        cwd: Path,
        args: dict[str, Any],
    ) -> ToolResult:
        """Search files using ripgrep."""
        pattern = args.get("pattern", "")
        include = args.get("include", "")
        search_path = args.get("path", ".")
        limit = args.get("limit", 100)
        
        if not pattern:
            return ToolResult.invalid(
                "Missing required parameter 'pattern'. "
                "Usage: grep_files(pattern: str, include?: str, path?: str, limit?: int)"
            )
        
        # Build ripgrep command
        cmd_parts = ["rg", "-l", "--color=never"]
        
        if include:
            cmd_parts.extend(["-g", include])
        
        cmd_parts.append(pattern)
        cmd_parts.append(search_path)
        
        cmd = " ".join(f'"{p}"' if " " in p else p for p in cmd_parts)
        
        try:
            result = subprocess.run(
                ["sh", "-c", cmd],
                cwd=str(cwd),
                capture_output=True,
                text=True,
                timeout=30,
            )
            
            files = [f for f in result.stdout.strip().split("\n") if f]
            
            if not files:
                return ToolResult.ok("No matches found")
            
            output = "\n".join(files[:limit])
            if len(files) > limit:
                output += f"\n\n[... {len(files) - limit} more files ...]"
            
            return ToolResult.ok(output)
            
        except subprocess.TimeoutExpired:
            return ToolResult.fail("Search timed out")
        except Exception as e:
            return ToolResult.fail(f"Search failed: {e}")
    
    def _run_view_image(self, cwd: Path, args: dict[str, Any]) -> ToolResult:
        """View an image file."""
        path = args.get("path", "")
        
        if not path:
            return ToolResult.invalid(
                "Missing required parameter 'path'. "
                "Usage: view_image(path: str)"
            )
        
        from src.tools.view_image import view_image
        return view_image(path, cwd)

    def _run_extract_video_frames(self, cwd: Path, args: dict[str, Any]) -> ToolResult:
        """Extract frames from a video file."""
        video_path = args.get("video_path", "")
        output_dir = args.get("output_dir", "")
        
        if not video_path:
            return ToolResult.invalid(
                "Missing required parameter 'video_path'. "
                "Usage: extract_video_frames(video_path: str, output_dir: str, fps?: float, max_frames?: int, start_time?: float, end_time?: float, scale?: str, format?: str)"
            )
        if not output_dir:
            return ToolResult.invalid(
                "Missing required parameter 'output_dir'. "
                "Usage: extract_video_frames(video_path: str, output_dir: str, fps?: float, max_frames?: int, start_time?: float, end_time?: float, scale?: str, format?: str)"
            )
        
        from src.tools.extract_video import extract_video_frames
        return extract_video_frames(
            video_path=video_path,
            output_dir=output_dir,
            cwd=cwd,
            fps=args.get("fps", 1.0),
            max_frames=args.get("max_frames", 30),
            start_time=args.get("start_time"),
            end_time=args.get("end_time"),
            scale=args.get("scale"),
            format=args.get("format", "png"),
        )
    
    def _run_extract_keyframes(self, cwd: Path, args: dict[str, Any]) -> ToolResult:
        """Extract keyframes (scene changes) from a video file."""
        video_path = args.get("video_path", "")
        output_dir = args.get("output_dir", "")
        
        if not video_path:
            return ToolResult.invalid(
                "Missing required parameter 'video_path'. "
                "Usage: extract_keyframes(video_path: str, output_dir: str, max_frames?: int, threshold?: float, format?: str)"
            )
        if not output_dir:
            return ToolResult.invalid(
                "Missing required parameter 'output_dir'. "
                "Usage: extract_keyframes(video_path: str, output_dir: str, max_frames?: int, threshold?: float, format?: str)"
            )
        
        from src.tools.extract_video import extract_keyframes
        return extract_keyframes(
            video_path=video_path,
            output_dir=output_dir,
            cwd=cwd,
            max_frames=args.get("max_frames", 20),
            threshold=args.get("threshold", 0.3),
            format=args.get("format", "png"),
        )
    
    def _run_finish(self, args: dict[str, Any]) -> ToolResult:
        """Signal task completion with a summary."""
        summary = args.get("summary", "")
        from src.tools.finish import execute_finish
        return execute_finish(summary)

    def _run_update_plan(self, args: dict[str, Any]) -> ToolResult:
        """Update the task plan."""
        steps = args.get("steps")
        explanation = args.get("explanation")
        
        if steps is None:
            return ToolResult.invalid(
                "Missing required parameter 'steps'. "
                "Usage: update_plan(steps: [{description: str, status: 'pending'|'in_progress'|'completed'}], explanation?: str)"
            )
        
        self._plan = steps
        
        # Format plan for output
        lines = ["Plan updated:"]
        for i, step in enumerate(steps, 1):
            status_icon = {
                "pending": "[ ]",
                "in_progress": "[>]",
                "completed": "[x]",
            }.get(step.get("status", "pending"), "[ ]")
            lines.append(f"  {status_icon} {i}. {step.get('description', '')}")
        
        if explanation:
            lines.append(f"\nReason: {explanation}")
        
        return ToolResult.ok("\n".join(lines))    
    
    def _run_web_search(self, args: dict[str, Any]) -> ToolResult:
        """Search the web for information."""
        query = args.get("query", "")
        num_results = args.get("num_results", 5)
        search_type = args.get("search_type", "general")
        
        if not query:
            return ToolResult.invalid(
                "Missing required parameter 'query'. "
                "Usage: web_search(query: str, num_results?: int, search_type?: str)"
            )
        
        from src.tools.web_search import web_search
        return web_search(query, num_results, search_type)

    def _run_spawn_process(self, cwd: Path, args: dict[str, Any]) -> ToolResult:
        """Start a long-running process in the background."""
        command = args.get("command", "")
        if not command:
            return ToolResult.invalid("Missing required parameter 'command'. Usage: spawn_process(command: str, ...)")
        runner = self._get_process_runner()
        return runner.spawn_process(
            cwd,
            command=command,
            workdir=args.get("cwd"),
            stdout_path=args.get("stdout_path"),
            stderr_path=args.get("stderr_path"),
        )

    def _run_kill_process(self, args: dict[str, Any]) -> ToolResult:
        """Terminate a process by PID."""
        pid = args.get("pid")
        if pid is None:
            return ToolResult.invalid("Missing required parameter 'pid'. Usage: kill_process(pid: int, ...)")
        runner = self._get_process_runner()
        return runner.kill_process(int(pid), sig=args.get("signal", "TERM"))

    def _run_wait_for_port(self, args: dict[str, Any]) -> ToolResult:
        """Wait until TCP host:port is accepting connections."""
        port = args.get("port")
        if port is None:
            return ToolResult.invalid("Missing required parameter 'port'. Usage: wait_for_port(port: int, ...)")
        from src.tools.process import run_wait_for_port
        return run_wait_for_port(
            host=args.get("host", "127.0.0.1"),
            port=int(port),
            timeout_sec=float(args.get("timeout_sec", 15)),
            poll_interval_sec=float(args.get("poll_interval_sec", 0.2)),
        )

    def _run_wait_for_file(self, args: dict[str, Any]) -> ToolResult:
        """Wait until a filesystem path exists."""
        path = args.get("path", "")
        if not path:
            return ToolResult.invalid("Missing required parameter 'path'. Usage: wait_for_file(path: str, ...)")
        from src.tools.process import run_wait_for_file
        return run_wait_for_file(
            path=path,
            timeout_sec=float(args.get("timeout_sec", 15)),
            poll_interval_sec=float(args.get("poll_interval_sec", 0.1)),
            min_size_bytes=int(args.get("min_size_bytes", 0)),
        )

    def _run_run_until_file(self, cwd: Path, args: dict[str, Any]) -> ToolResult:
        """Run command until target file exists, then terminate."""
        command = args.get("command", "")
        file_path = args.get("file_path", "")
        if not command or not file_path:
            return ToolResult.invalid("Missing required parameters 'command' and 'file_path'. Usage: run_until_file(command: str, file_path: str, ...)")
        from src.tools.process import run_run_until_file
        return run_run_until_file(
            cwd,
            command=command,
            file_path=file_path,
            workdir=args.get("cwd"),
            timeout_sec=float(args.get("timeout_sec", 30)),
            poll_interval_sec=float(args.get("poll_interval_sec", 0.1)),
            min_size_bytes=int(args.get("min_size_bytes", 1)),
            terminate_grace_sec=float(args.get("terminate_grace_sec", 2.0)),
        )    

    # -------------------------------------------------------------------------
    # Caching methods
    # -------------------------------------------------------------------------
    
    def _cache_key(self, name: str, arguments: dict[str, Any]) -> str:
        """Generate a cache key for a tool call."""
        args_json = json.dumps(arguments, sort_keys=True, default=str)
        content = f"{name}:{args_json}"
        return hashlib.sha256(content.encode()).hexdigest()[:32]
    
    def _get_cached(self, key: str) -> Optional[ToolResult]:
        """Get a cached result if valid."""
        cached = self._cache.get(key)
        if cached is not None and cached.is_valid(self._config.cache_ttl):
            return cached.result
        return None
    
    def _cache_result(self, key: str, result: ToolResult) -> None:
        """Cache a tool result."""
        self._cache[key] = CachedResult(result=result, cached_at=time.time())
        
        # Evict old entries if cache is too large
        if len(self._cache) > 1000:
            self._evict_expired_cache()
    
    def _evict_expired_cache(self) -> None:
        """Remove expired entries from cache."""
        now = time.time()
        expired_keys = [
            key for key, cached in self._cache.items()
            if not cached.is_valid(self._config.cache_ttl)
        ]
        for key in expired_keys:
            del self._cache[key]
    
    def clear_cache(self) -> None:
        """Clear the entire cache."""
        self._cache.clear()
    
    # -------------------------------------------------------------------------
    # Statistics methods
    # -------------------------------------------------------------------------
    
    def _record_execution(
        self,
        tool_name: str,
        duration_ms: int,
        success: bool,
        cached: bool,
    ) -> None:
        """Record execution statistics."""
        self._stats.total_executions += 1
        self._stats.total_duration_ms += duration_ms
        
        if success:
            self._stats.successful_executions += 1
        else:
            self._stats.failed_executions += 1
        
        if cached:
            self._stats.cache_hits += 1
        
        # Per-tool stats
        if tool_name not in self._stats.by_tool:
            self._stats.by_tool[tool_name] = ToolStats()
        
        tool_stats = self._stats.by_tool[tool_name]
        tool_stats.executions += 1
        tool_stats.total_ms += duration_ms
        if success:
            tool_stats.successes += 1
    
    def stats(self) -> ExecutorStats:
        """Get execution statistics."""
        return self._stats
    
    # -------------------------------------------------------------------------
    # Batch execution
    # -------------------------------------------------------------------------
    
    def execute_batch(
        self,
        ctx: "AgentContext",
        calls: List[Tuple[str, dict]],
    ) -> List[ToolResult]:
        """Execute multiple tool calls in parallel.
        
        Args:
            ctx: Agent context with shell() method
            calls: List of (tool_name, arguments) tuples
            
        Returns:
            List of ToolResults in the same order as input calls
        """
        if not calls:
            return []
        
        # For single call, just execute directly
        if len(calls) == 1:
            name, args = calls[0]
            return [self.execute(ctx, name, args)]
        
        # Execute in parallel using ThreadPoolExecutor
        results: List[Optional[ToolResult]] = [None] * len(calls)
        
        with ThreadPoolExecutor(max_workers=self._config.max_concurrent) as executor:
            future_to_index = {
                executor.submit(self.execute, ctx, name, args): i
                for i, (name, args) in enumerate(calls)
            }
            
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    results[index] = future.result()
                except Exception as e:
                    results[index] = ToolResult.fail(f"Batch execution failed: {e}")
        
        # Ensure all results are filled (shouldn't happen, but just in case)
        return [r if r is not None else ToolResult.fail("No result") for r in results]
    
    def get_plan(self) -> list[dict[str, str]]:
        """Get the current plan."""
        return self._plan.copy()
    
    def get_tools_for_llm(self) -> list:
        """Get tool specifications formatted for the LLM.
        
        Returns tools in OpenAI-compatible format for litellm.
        """
        specs = get_all_tools()
        tools = []
        
        for spec in specs:
            tools.append({
                "name": spec["name"],
                "description": spec.get("description", ""),
                "parameters": spec.get("parameters", {}),
            })
        
        return tools