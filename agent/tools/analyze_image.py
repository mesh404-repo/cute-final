"""
Analyze image tool - uses a vision model to analyze a local image per user instructions.

Takes image path and instructions; returns the analysis as text (no image injection).
Use when the main model does not support images (e.g. GLM-4.7-TEE).
"""

from __future__ import annotations

from pathlib import Path

from agent.tools.base import ToolResult
from agent.images.loader import load_image_as_data_uri, make_image_content


def analyze_image(
    file_path: str,
    instructions: str,
    cwd: Path,
) -> ToolResult:
    """
    Load a local image, send it to the vision model with the given instructions, return analysis text.

    Args:
        file_path: Path to the image file (relative or absolute)
        instructions: What to analyze (e.g. "Transcribe all text", "Describe the UI")
        cwd: Current working directory

    Returns:
        ToolResult with success status and analysis text or error message
    """
    path = Path(file_path)
    if not path.is_absolute():
        path = cwd / path
    path = path.resolve()

    if not path.exists():
        return ToolResult(success=False, output=f"Image not found: {path}")

    if not path.is_file():
        return ToolResult(success=False, output=f"Not a file: {path}")

    valid_extensions = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".ppm", ".pgm", ".pbm", ".pnm"}
    if path.suffix.lower() not in valid_extensions:
        return ToolResult(
            success=False,
            output=f"Not a valid image file: {path} (supported: {', '.join(valid_extensions)})",
        )

    if not (instructions or "").strip():
        return ToolResult(
            success=False,
            output="Missing or empty 'instructions'. Say what to analyze (e.g. 'Transcribe all text', 'Describe the diagram').",
        )

    try:
        data_uri = load_image_as_data_uri(path)
        image_content = make_image_content(data_uri)
    except Exception as e:
        return ToolResult(success=False, output=f"Failed to load image: {e}")

    try:
        from agent.llm.vision import analyze_image_with_instructions
        text, _cost = analyze_image_with_instructions(image_content, instructions.strip())
        return ToolResult(success=True, output=text)
    except ValueError as e:
        return ToolResult(success=False, output=str(e))
    except Exception as e:
        return ToolResult(success=False, output=f"Vision analysis failed: {e}")
