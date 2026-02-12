"""
Image analysis via a vision-capable model.

Provides a vision client and analysis for the analyze_image tool:
load image, send to vision model with custom instructions, return text result.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from agent.llm.client import LLMClient, LLMError
import time

KIMI_2_5_TEE = "moonshotai/Kimi-K2.5-TEE"
DEEPSEEK_3_2_TEE = "deepseek-ai/DeepSeek-V3.2-TEE"
VISION_MODELS = [KIMI_2_5_TEE, DEEPSEEK_3_2_TEE]

def analyze_image_with_instructions(
    image_content: Dict[str, Any],
    instructions: str,
    vision_llm: "LLMClient",
) -> Tuple[str, float]:
    """
    Send the image and instructions to the vision model and return (text, cost).

    Args:
        image_content: Content block dict, e.g. {"type": "image_url", "image_url": {"url": "data:..."}}.
        instructions: User instructions for what to analyze (e.g. "Transcribe all text").

    Returns:
        Tuple of (analysis_text, cost).
    """
    content: List[Dict[str, Any]] = [
        {"type": "text", "text": instructions},
        image_content,
    ]
    messages = [{"role": "user", "content": content}]    

    retry = 0
    model = DEEPSEEK_3_2_TEE
    error_msg = ""
    while retry < 3:
        try:
            response = vision_llm.chat(messages, model=model, max_tokens=4096)
            text = (response.text or "").strip()
            cost = getattr(response, "cost", 0.0)
            return text or "[Vision model returned no content.]", cost
        except LLMError as e:
            error_msg = e.message

            model_index = VISION_MODELS.index(model) if model in VISION_MODELS else -1
            model = VISION_MODELS[(model_index + 1) % len(VISION_MODELS)]

            retry += 1

        time.sleep(4)        
    
    return f"[Image analysis failed: {error_msg}]", 0.0
