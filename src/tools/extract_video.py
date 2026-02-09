"""
Extract video frames tool for SuperAgent.

Extracts frames from video files using ffmpeg for analysis.
Useful for video understanding tasks like game action detection,
tutorial extraction, etc.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.tools.base import ToolResult


def _check_ffmpeg() -> bool:
    """Check if ffmpeg is installed and available."""
    return shutil.which("ffmpeg") is not None


def _get_video_info(video_path: Path) -> Dict[str, Any]:
    """Get video metadata using ffprobe."""
    info = {
        "duration": None,
        "width": None,
        "height": None,
        "fps": None,
        "codec": None,
    }
    
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=width,height,r_frame_rate,codec_name",
                "-show_entries", "format=duration",
                "-of", "json",
                str(video_path),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        
        if result.returncode == 0:
            data = json.loads(result.stdout)
            
            if "format" in data:
                duration = data["format"].get("duration")
                if duration:
                    info["duration"] = float(duration)
            
            if "streams" in data and data["streams"]:
                stream = data["streams"][0]
                info["width"] = stream.get("width")
                info["height"] = stream.get("height")
                info["codec"] = stream.get("codec_name")
                
                fps_str = stream.get("r_frame_rate", "")
                if "/" in fps_str:
                    num, den = fps_str.split("/")
                    if float(den) > 0:
                        info["fps"] = round(float(num) / float(den), 2)
                elif fps_str:
                    info["fps"] = float(fps_str)
                    
    except Exception:
        pass
    
    return info


def extract_video_frames(
    video_path: str,
    output_dir: str,
    cwd: Path,
    fps: float = 1.0,
    max_frames: Optional[int] = 30,
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
    scale: Optional[str] = None,
    format: str = "png",
) -> ToolResult:
    """
    Extract frames from a video file using ffmpeg.
    
    Args:
        video_path: Path to the video file (relative or absolute)
        output_dir: Directory to save extracted frames
        cwd: Current working directory
        fps: Frames per second to extract (default: 1.0)
        max_frames: Maximum number of frames to extract (default: 30, None for unlimited)
        start_time: Start time in seconds (optional)
        end_time: End time in seconds (optional)
        scale: Output scale, e.g., "640:480" or "640:-1" for auto height (optional)
        format: Output format: "png" or "jpg" (default: png)
        
    Returns:
        ToolResult with list of extracted frame paths and metadata
    """
    # Check ffmpeg availability
    if not _check_ffmpeg():
        return ToolResult.fail(
            "ffmpeg is not installed. Install it with: apt-get install ffmpeg"
        )
    
    # Resolve video path
    video = Path(video_path)
    if not video.is_absolute():
        video = cwd / video
    video = video.resolve()
    
    if not video.exists():
        return ToolResult.fail(f"Video file not found: {video}")
    
    if not video.is_file():
        return ToolResult.fail(f"Not a file: {video}")
    
    # Check video extension
    valid_extensions = {".mp4", ".avi", ".mkv", ".mov", ".webm", ".flv", ".wmv", ".m4v"}
    if video.suffix.lower() not in valid_extensions:
        return ToolResult.fail(
            f"Unsupported video format: {video.suffix} "
            f"(supported: {', '.join(valid_extensions)})"
        )
    
    # Resolve output directory
    out_dir = Path(output_dir)
    if not out_dir.is_absolute():
        out_dir = cwd / out_dir
    out_dir = out_dir.resolve()
    
    # Create output directory
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        return ToolResult.fail(f"Failed to create output directory: {e}")
    
    # Get video info
    video_info = _get_video_info(video)
    duration = video_info.get("duration")
    
    # Validate time range
    if start_time is not None and start_time < 0:
        start_time = 0
    if end_time is not None and duration and end_time > duration:
        end_time = duration
    if start_time is not None and end_time is not None and start_time >= end_time:
        return ToolResult.fail(
            f"Invalid time range: start ({start_time}s) >= end ({end_time}s)"
        )
    
    # Calculate effective duration for frame limiting
    effective_start = start_time or 0
    effective_end = end_time or duration or 60  # Default 60s if unknown
    effective_duration = effective_end - effective_start
    
    # Adjust FPS if max_frames would be exceeded
    original_fps = fps
    expected_frames = int(effective_duration * fps)
    if max_frames and expected_frames > max_frames:
        fps = max_frames / effective_duration
        expected_frames = max_frames
    
    # Validate output format
    if format.lower() not in ("png", "jpg", "jpeg"):
        format = "png"
    ext = "jpg" if format.lower() in ("jpg", "jpeg") else "png"
    
    # Build ffmpeg command
    cmd = ["ffmpeg", "-y"]  # -y to overwrite
    
    # Input options
    if start_time is not None:
        cmd.extend(["-ss", str(start_time)])
    
    cmd.extend(["-i", str(video)])
    
    if end_time is not None:
        duration_arg = end_time - (start_time or 0)
        cmd.extend(["-t", str(duration_arg)])
    
    # Video filters
    vf_parts = [f"fps={fps}"]
    
    if scale:
        vf_parts.append(f"scale={scale}")
    
    cmd.extend(["-vf", ",".join(vf_parts)])
    
    # Frame limit
    if max_frames:
        cmd.extend(["-frames:v", str(max_frames)])
    
    # Output options
    if ext == "jpg":
        cmd.extend(["-q:v", "2"])  # High quality JPEG
    
    # Output pattern
    output_pattern = out_dir / f"frame_%04d.{ext}"
    cmd.append(str(output_pattern))
    
    # Execute ffmpeg
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout
            cwd=str(cwd),
        )
        
        if result.returncode != 0:
            error_msg = result.stderr[-1000:] if result.stderr else "Unknown error"
            return ToolResult.fail(f"ffmpeg failed: {error_msg}")
            
    except subprocess.TimeoutExpired:
        return ToolResult.fail("Frame extraction timed out (5 minute limit)")
    except Exception as e:
        return ToolResult.fail(f"Failed to run ffmpeg: {e}")
    
    # List extracted frames
    frames = sorted(out_dir.glob(f"frame_*.{ext}"))
    frame_count = len(frames)
    
    if frame_count == 0:
        return ToolResult.fail(
            "No frames were extracted. The video may be too short or corrupted."
        )
    
    # Build output message
    frame_list = [str(f.relative_to(cwd)) for f in frames[:10]]
    if frame_count > 10:
        frame_list.append(f"... and {frame_count - 10} more")
    
    output_lines = [
        f"Extracted {frame_count} frames from {video.name}",
        f"",
        f"Video info:",
        f"  - Duration: {video_info.get('duration', 'unknown')}s",
        f"  - Resolution: {video_info.get('width', '?')}x{video_info.get('height', '?')}",
        f"  - Original FPS: {video_info.get('fps', 'unknown')}",
        f"",
        f"Extraction settings:",
        f"  - Sample FPS: {fps:.2f}" + (f" (reduced from {original_fps})" if fps != original_fps else ""),
        f"  - Output dir: {out_dir.relative_to(cwd)}",
        f"  - Format: {ext.upper()}",
        f"",
        f"Extracted frames:",
    ]
    output_lines.extend([f"  - {f}" for f in frame_list])
    output_lines.append("")
    output_lines.append("Use view_image to analyze individual frames.")
    
    return ToolResult(
        success=True,
        output="\n".join(output_lines),
        data={
            "frame_count": frame_count,
            "frames": [str(f) for f in frames],
            "output_dir": str(out_dir),
            "video_info": video_info,
            "extraction_fps": fps,
        },
    )


def extract_keyframes(
    video_path: str,
    output_dir: str,
    cwd: Path,
    max_frames: int = 20,
    threshold: float = 0.3,
    format: str = "png",
) -> ToolResult:
    """
    Extract keyframes (scene changes) from a video using ffmpeg.
    
    This extracts frames where significant visual changes occur,
    which is more efficient than fixed-interval extraction for
    understanding video content.
    
    Args:
        video_path: Path to the video file
        output_dir: Directory to save extracted frames
        cwd: Current working directory
        max_frames: Maximum keyframes to extract (default: 20)
        threshold: Scene change threshold 0.0-1.0 (default: 0.3, lower = more frames)
        format: Output format: "png" or "jpg"
        
    Returns:
        ToolResult with list of extracted keyframe paths
    """
    if not _check_ffmpeg():
        return ToolResult.fail(
            "ffmpeg is not installed. Install it with: apt-get install ffmpeg"
        )
    
    # Resolve paths
    video = Path(video_path)
    if not video.is_absolute():
        video = cwd / video
    video = video.resolve()
    
    if not video.exists():
        return ToolResult.fail(f"Video file not found: {video}")
    
    out_dir = Path(output_dir)
    if not out_dir.is_absolute():
        out_dir = cwd / out_dir
    out_dir = out_dir.resolve()
    
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        return ToolResult.fail(f"Failed to create output directory: {e}")
    
    # Validate format
    ext = "jpg" if format.lower() in ("jpg", "jpeg") else "png"
    
    # Build ffmpeg command for scene detection
    output_pattern = out_dir / f"keyframe_%04d.{ext}"
    
    cmd = [
        "ffmpeg", "-y",
        "-i", str(video),
        "-vf", f"select='gt(scene,{threshold})',showinfo",
        "-vsync", "vfr",
        "-frames:v", str(max_frames),
    ]
    
    if ext == "jpg":
        cmd.extend(["-q:v", "2"])
    
    cmd.append(str(output_pattern))
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,
            cwd=str(cwd),
        )
        # ffmpeg may return non-zero even on partial success
        
    except subprocess.TimeoutExpired:
        return ToolResult.fail("Keyframe extraction timed out")
    except Exception as e:
        return ToolResult.fail(f"Failed to run ffmpeg: {e}")
    
    # List extracted keyframes
    keyframes = sorted(out_dir.glob(f"keyframe_*.{ext}"))
    frame_count = len(keyframes)
    
    if frame_count == 0:
        return ToolResult.fail(
            f"No keyframes detected with threshold {threshold}. "
            "Try lowering the threshold or use extract_video_frames for fixed-interval extraction."
        )
    
    # Get video info
    video_info = _get_video_info(video)
    
    # Build output
    frame_list = [str(f.relative_to(cwd)) for f in keyframes[:10]]
    if frame_count > 10:
        frame_list.append(f"... and {frame_count - 10} more")
    
    output_lines = [
        f"Extracted {frame_count} keyframes (scene changes) from {video.name}",
        f"",
        f"Video info:",
        f"  - Duration: {video_info.get('duration', 'unknown')}s",
        f"  - Resolution: {video_info.get('width', '?')}x{video_info.get('height', '?')}",
        f"",
        f"Extraction settings:",
        f"  - Scene threshold: {threshold}",
        f"  - Max frames: {max_frames}",
        f"",
        f"Keyframes:",
    ]
    output_lines.extend([f"  - {f}" for f in frame_list])
    output_lines.append("")
    output_lines.append("Use view_image to analyze individual keyframes.")
    
    return ToolResult(
        success=True,
        output="\n".join(output_lines),
        data={
            "frame_count": frame_count,
            "frames": [str(f) for f in keyframes],
            "output_dir": str(out_dir),
            "video_info": video_info,
            "threshold": threshold,
        },
    )


# =============================================================================
# Tool Specifications for LLM
# =============================================================================

EXTRACT_VIDEO_FRAMES_SPEC: Dict[str, Any] = {
    "name": "extract_video_frames",
    "description": """Extract frames from a video file at regular intervals.

Use this to analyze video content by extracting frames that can then be viewed with view_image.
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

EXTRACT_KEYFRAMES_SPEC: Dict[str, Any] = {
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
