"""Process tools - spawn_process, kill_process, wait_for_port, wait_for_file, run_until_file."""

from __future__ import annotations

import os
import signal
import socket
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, Optional

from src.tools.base import ToolResult


def _resolve_cwd(cwd: Path, workdir: Optional[str]) -> Path:
    if not workdir:
        return cwd
    p = Path(workdir)
    if p.is_absolute():
        return p
    return (cwd / p).resolve(strict=False)


class ProcessToolRunner:
    """Holds spawned process refs for kill_process. Registry should create one per context."""

    def __init__(self) -> None:
        self._procs: Dict[int, subprocess.Popen] = {}
        self._log_dir = Path(os.environ.get("TBH_TOOL_LOG_DIR", "/tmp/tbh-tool-logs"))

    def spawn_process(
        self,
        cwd: Path,
        command: str,
        workdir: Optional[str] = None,
        stdout_path: Optional[str] = None,
        stderr_path: Optional[str] = None,
    ) -> ToolResult:
        """Start a long-running process in the background."""
        run_cwd = _resolve_cwd(cwd, workdir)
        self._log_dir.mkdir(parents=True, exist_ok=True)
        ts = int(time.time() * 1000)
        out_path = Path(stdout_path) if stdout_path else (self._log_dir / f"spawn_{ts}.out.log")
        err_path = Path(stderr_path) if stderr_path else (self._log_dir / f"spawn_{ts}.err.log")
        try:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            err_path.parent.mkdir(parents=True, exist_ok=True)
            out_f = out_path.open("a", encoding="utf-8")
            err_f = err_path.open("a", encoding="utf-8")
        except Exception as e:
            return ToolResult.fail(f"Failed to open log files: {e}")

        try:
            proc = subprocess.Popen(
                ["bash", "-lc", command],
                cwd=str(run_cwd),
                stdout=out_f,
                stderr=err_f,
                text=True,
                env=os.environ.copy(),
                preexec_fn=os.setsid if hasattr(os, "setsid") else None,
            )
            self._procs[int(proc.pid)] = proc
        except Exception as e:
            try:
                out_f.close()
                err_f.close()
            except Exception:
                pass
            return ToolResult.fail(f"spawn failed: {e}")

        out = f"pid: {proc.pid}\ncommand: {command}\ncwd: {run_cwd}\nstdout_path: {out_path}\nstderr_path: {err_path}"
        return ToolResult.ok(out)

    def kill_process(self, pid: int, sig: str = "TERM") -> ToolResult:
        """Terminate a process by PID."""
        sig_val = signal.SIGKILL if (sig or "TERM").upper() == "KILL" else signal.SIGTERM
        pid_int = int(pid)
        proc = self._procs.get(pid_int)
        try:
            if proc is not None and hasattr(os, "killpg"):
                try:
                    os.killpg(os.getpgid(pid_int), sig_val)
                except Exception:
                    os.kill(pid_int, sig_val)
            else:
                os.kill(pid_int, sig_val)
        except ProcessLookupError:
            return ToolResult.ok(f"pid {pid_int}: already dead")
        except PermissionError as e:
            return ToolResult.fail(f"Permission error: {e}")
        except Exception as e:
            return ToolResult.fail(f"Failed to kill pid {pid_int}: {e}")
        if pid_int in self._procs:
            del self._procs[pid_int]
        return ToolResult.ok(f"pid {pid_int}: sent {sig}")


def run_wait_for_port(
    host: str = "127.0.0.1",
    port: int = 0,
    timeout_sec: float = 15.0,
    poll_interval_sec: float = 0.2,
) -> ToolResult:
    """Wait until a TCP host:port is accepting connections."""
    deadline = time.time() + float(timeout_sec)
    last_err = None
    while time.time() < deadline:
        try:
            with socket.create_connection((host, int(port)), timeout=1.0):
                return ToolResult.ok(f"host={host} port={port} ready=True")
        except Exception as e:
            last_err = str(e)
            time.sleep(float(poll_interval_sec))
    return ToolResult.fail(f"timeout: {last_err or 'no connection'}")


def run_wait_for_file(
    path: str,
    timeout_sec: float = 15.0,
    poll_interval_sec: float = 0.1,
    min_size_bytes: int = 0,
) -> ToolResult:
    """Wait until a filesystem path exists (or timeout)."""
    p = Path(path)
    deadline = time.time() + float(timeout_sec)
    while time.time() < deadline:
        try:
            if p.exists():
                st = p.stat()
                if int(st.st_size) >= int(min_size_bytes):
                    return ToolResult.ok(
                        f"path={p} exists=True size_bytes={st.st_size} mtime={st.st_mtime}"
                    )
        except Exception:
            pass
        time.sleep(float(poll_interval_sec))
    return ToolResult.fail(f"timeout waiting for {path}")


def run_run_until_file(
    cwd: Path,
    command: str,
    file_path: str,
    workdir: Optional[str] = None,
    timeout_sec: float = 30.0,
    poll_interval_sec: float = 0.1,
    min_size_bytes: int = 1,
    terminate_grace_sec: float = 2.0,
) -> ToolResult:
    """Run command until target file exists (or timeout), then terminate."""
    run_cwd = _resolve_cwd(cwd, workdir)
    log_dir = Path(os.environ.get("TBH_TOOL_LOG_DIR", "/tmp/tbh-tool-logs"))
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = int(time.time() * 1000)
    out_path = log_dir / f"run_until_{ts}.out.log"
    err_path = log_dir / f"run_until_{ts}.err.log"
    try:
        out_f = out_path.open("a", encoding="utf-8")
        err_f = err_path.open("a", encoding="utf-8")
    except Exception as e:
        return ToolResult.fail(f"Failed to open log files: {e}")

    target = Path(file_path)
    proc: Optional[subprocess.Popen] = None
    started = time.time()
    try:
        proc = subprocess.Popen(
            ["bash", "-lc", command],
            cwd=str(run_cwd),
            stdout=out_f,
            stderr=err_f,
            text=True,
            env=os.environ.copy(),
            preexec_fn=os.setsid if hasattr(os, "setsid") else None,
        )
        pid = int(proc.pid)
        deadline = started + float(timeout_sec)
        found = False
        while time.time() < deadline:
            try:
                if target.exists() and target.stat().st_size >= int(min_size_bytes):
                    found = True
                    break
            except Exception:
                pass
            if proc.poll() is not None:
                break
            time.sleep(float(poll_interval_sec))

        if proc.poll() is None:
            try:
                if hasattr(os, "killpg"):
                    os.killpg(os.getpgid(pid), signal.SIGTERM)
                else:
                    os.kill(pid, signal.SIGTERM)
            except Exception:
                pass
            grace_deadline = time.time() + float(terminate_grace_sec)
            while time.time() < grace_deadline and proc.poll() is None:
                time.sleep(0.05)
            if proc.poll() is None:
                try:
                    if hasattr(os, "killpg"):
                        os.killpg(os.getpgid(pid), signal.SIGKILL)
                    else:
                        os.kill(pid, signal.SIGKILL)
                except Exception:
                    pass

        exit_code = proc.poll()
        dt = time.time() - started
        out_lines = [
            f"ok={found}",
            f"pid={pid}",
            f"file_path={target}",
            f"found={found}",
            f"duration_sec={dt:.2f}",
            f"exit_code={exit_code}",
            f"stdout_path={out_path}",
            f"stderr_path={err_path}",
        ]
        return ToolResult.ok("\n".join(out_lines))
    except Exception as e:
        return ToolResult.fail(f"run_until_file failed: {e}")
    finally:
        for f in (out_f, err_f):
            try:
                if f is not None:
                    f.close()
            except Exception:
                pass
