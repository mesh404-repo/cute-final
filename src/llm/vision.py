"""
Image analysis via a vision-capable model.

When the main model does not support images (e.g. GLM-4.7-TEE), images from
view_image are analyzed by a separate vision model (e.g. Kimi K2.5-TEE) and
the text description is injected into the conversation instead of raw image content.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from src.llm.client import LLMClient, LLMError

# Prompt for the vision model to describe the image for the coding agent
VISION_ANALYSIS_PROMPT = """Describe this image in detail for a coding agent. Include:
- Any text visible in the image (exact strings, code, file paths, terminal output).
- UI elements, buttons, menus, or dialogs.
- Diagrams, charts, or geometric shapes.
- File contents, error messages, or logs if visible.
- Anything relevant to completing a programming or system task.
Be concise but complete. If the image is a screenshot of code or a terminal, transcribe the content accurately."""


def analyze_image(
    vision_client: LLMClient,
    image_content: Dict[str, Any],
    tool_name: str = "view_image",
) -> Tuple[str, float]:
    """
    Send the image to the vision model and return a text description.

    Args:
        vision_client: LLM client configured with a vision-capable model.
        image_content: Content block dict, e.g. {"type": "image_url", "image_url": {"url": "data:..."}}.
        tool_name: Name of the tool that produced the image (for logging).

    Returns:
        Tuple of (description_text, cost).
    """
    content: List[Dict[str, Any]] = [
        {"type": "text", "text": VISION_ANALYSIS_PROMPT},
        image_content,
    ]
    messages = [{"role": "user", "content": content}]
    try:
        response = vision_client.chat(messages, max_tokens=4096)
        text = (response.text or "").strip()
        cost = getattr(response, "cost", 0.0)
        return text or "[Vision model returned no description.]", cost
    except LLMError as e:
        return f"[Image analysis failed: {e.message}]", 0.0


def analyze_images_with_vision(
    vision_client: LLMClient,
    pending_images: List[Dict[str, Any]],
    max_images: int = 5,
) -> Tuple[str, float]:
    """
    Analyze multiple image blocks with the vision model and return combined text and total cost.

    Args:
        vision_client: LLM client for vision model.
        pending_images: List of {"tool_name": str, "content": image_content_dict}.
        max_images: Maximum number of images to process per turn.

    Returns:
        Tuple of (combined_analysis_text, total_vision_cost).
    """
    combined: List[str] = []
    total_cost = 0.0
    to_process = pending_images[:max_images]
    for i, img in enumerate(to_process):
        tool_name = img.get("tool_name", "view_image")
        content = img.get("content")
        if not content or not isinstance(content, dict):
            combined.append(f"Image {i + 1} from {tool_name}: [invalid content]")
            continue
        text, cost = analyze_image(vision_client, content, tool_name=tool_name)
        total_cost += cost
        combined.append(f"Image from {tool_name}:\n{text}")
    return "\n\n---\n\n".join(combined), total_cost
