"""
Context compaction system for SuperAgent.

Implements intelligent context management like OpenCode/Codex:
1. Token-based overflow detection
2. Tool output pruning (clear old outputs, keep recent)
3. AI-powered conversation compaction (summarization)

This replaces naive sliding window truncation which breaks cache.
"""

from __future__ import annotations

import sys
import time
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from src.llm.client import LiteLLMClient

# =============================================================================
# Constants (matching OpenCode)
# =============================================================================

# Token estimation
APPROX_CHARS_PER_TOKEN = 4

# Context limits
MODEL_CONTEXT_LIMIT = 200_000  # Claude Opus 4.5 context window
OUTPUT_TOKEN_MAX = 32_000  # Max output tokens to reserve
AUTO_COMPACT_THRESHOLD = 0.85  # Trigger compaction at 85% of usable context

# Pruning constants (from OpenCode)
PRUNE_PROTECT = 40_000  # Protect this many tokens of recent tool output
PRUNE_MINIMUM = 20_000  # Only prune if we can recover at least this many tokens
PRUNE_MARKER = "[Old tool result content cleared]"

# Image management constants
IMAGE_LIMIT = 10  # Maximum number of images to keep in context

# Compaction prompts (from Codex)
COMPACTION_PROMPT = """You are performing a CONTEXT CHECKPOINT COMPACTION. Create a handoff summary for another LLM that will resume the task.

Include:
- Current progress and key decisions made
- Important context, constraints, or user preferences
- What remains to be done (clear next steps)
- Any critical data, examples, or references needed to continue
- Which files were modified and how
- Any errors encountered and how they were resolved

Be concise, structured, and focused on helping the next LLM seamlessly continue the work. Use bullet points and clear sections."""

SUMMARY_PREFIX = """Another language model started to solve this problem and produced a summary of its thinking process. You also have access to the state of the tools that were used. Use this to build on the work that has already been done and avoid duplicating work.

Here is the summary from the previous context:

"""


# =============================================================================
# Token Estimation
# =============================================================================

def estimate_tokens(text: str) -> int:
    """Estimate tokens from text length (4 chars per token heuristic)."""
    return max(0, len(text or "") // APPROX_CHARS_PER_TOKEN)


def estimate_message_tokens(msg: Dict[str, Any]) -> int:
    """Estimate tokens for a single message."""
    tokens = 0
    
    # Content tokens
    content = msg.get("content")
    if isinstance(content, str):
        tokens += estimate_tokens(content)
    elif isinstance(content, list):
        for part in content:
            if isinstance(part, dict):
                tokens += estimate_tokens(part.get("text", ""))
                # Images count as ~1000 tokens roughly
                if part.get("type") == "image_url":
                    tokens += 1000
    
    # Tool calls tokens (function name + arguments)
    tool_calls = msg.get("tool_calls", [])
    for tc in tool_calls:
        func = tc.get("function", {})
        tokens += estimate_tokens(func.get("name", ""))
        tokens += estimate_tokens(func.get("arguments", ""))
    
    # Role overhead (~4 tokens)
    tokens += 4
    
    return tokens


def estimate_total_tokens(messages: List[Dict[str, Any]]) -> int:
    """Estimate total tokens for all messages."""
    return sum(estimate_message_tokens(m) for m in messages)


# =============================================================================
# Overflow Detection
# =============================================================================

def get_usable_context() -> int:
    """Get usable context window (total - reserved for output)."""
    return MODEL_CONTEXT_LIMIT - OUTPUT_TOKEN_MAX


def is_overflow(total_tokens: int, threshold: float = AUTO_COMPACT_THRESHOLD) -> bool:
    """Check if context is overflowing based on token count."""
    usable = get_usable_context()
    return total_tokens > usable * threshold


def needs_compaction(messages: List[Dict[str, Any]]) -> bool:
    """Check if messages need compaction."""
    total_tokens = estimate_total_tokens(messages)
    return is_overflow(total_tokens)


# =============================================================================
# Image Management
# =============================================================================

def count_images_in_messages(messages: List[Dict[str, Any]]) -> Tuple[int, int, int]:
    """
    Count images in messages and determine which are analyzed.
    
    An image is considered "analyzed" if it appears in a user message
    that comes before an assistant message (meaning the LLM has seen it).
    
    Returns:
        Tuple of (analyzed_count, unanalyzed_count, total_count)
    """
    total_images = 0
    analyzed_images = 0
    
    # Track which messages have images and their positions
    image_positions: List[int] = []
    
    # First pass: find all images
    for i, msg in enumerate(messages):
        content = msg.get("content")
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "image_url":
                    total_images += 1
                    image_positions.append(i)
    
    # Second pass: determine which are analyzed
    # An image is analyzed if there's an assistant message after it
    for img_pos in image_positions:
        # Check if there's an assistant message after this image
        for i in range(img_pos + 1, len(messages)):
            if messages[i].get("role") == "assistant":
                analyzed_images += 1
                break
    
    unanalyzed_images = total_images - analyzed_images
    return analyzed_images, unanalyzed_images, total_images


def prune_images_from_messages(
    messages: List[Dict[str, Any]],
    limit: int = IMAGE_LIMIT,
) -> Tuple[List[Dict[str, Any]], int, int]:
    """
    Prune old images from messages, keeping only the most recent ones.
    
    Strategy:
    1. Go backwards through messages
    2. Keep images from the most recent messages
    3. Remove images from older messages once limit is reached
    
    Args:
        messages: List of messages
        limit: Maximum number of images to keep
        
    Returns:
        Tuple of (pruned_messages, images_removed_count, messages_affected_count)
    """
    if not messages:
        return messages, 0, 0
    
    # Count current images
    _, _, total_images = count_images_in_messages(messages)
    
    if total_images <= limit:
        return messages, 0, 0
    
    # Track images and their positions
    image_data: List[Tuple[int, int]] = []  # (message_index, content_index)
    
    for msg_idx, msg in enumerate(messages):
        content = msg.get("content")
        if isinstance(content, list):
            for content_idx, part in enumerate(content):
                if isinstance(part, dict) and part.get("type") == "image_url":
                    image_data.append((msg_idx, content_idx))
    
    # Keep only the last `limit` images
    images_to_remove = len(image_data) - limit
    if images_to_remove <= 0:
        return messages, 0, 0
    
    # Remove images from oldest messages first
    images_to_remove_list = image_data[:-limit] if limit > 0 else image_data
    
    # Group by message index for efficient removal
    messages_to_modify: Dict[int, List[int]] = {}
    for msg_idx, content_idx in images_to_remove_list:
        if msg_idx not in messages_to_modify:
            messages_to_modify[msg_idx] = []
        messages_to_modify[msg_idx].append(content_idx)
    
    messages_affected = len(messages_to_modify)
    
    # Build new messages with images removed
    result = []
    for msg_idx, msg in enumerate(messages):
        if msg_idx not in messages_to_modify:
            result.append(msg)
            continue
        
        # This message has images to remove
        content = msg.get("content")
        if isinstance(content, list):
            # Remove image parts at specified indices (in reverse order to maintain indices)
            indices_to_remove = set(messages_to_modify[msg_idx])
            new_content = [
                part for i, part in enumerate(content)
                if i not in indices_to_remove
            ]
            # If content becomes empty or only has one text part, simplify
            if len(new_content) == 1 and isinstance(new_content[0], dict) and new_content[0].get("type") == "text":
                result.append({**msg, "content": new_content[0].get("text", "")})
            elif new_content:
                result.append({**msg, "content": new_content})
            else:
                # Empty content, skip this message
                continue
        else:
            result.append(msg)
    
    return result, len(images_to_remove_list), messages_affected


# =============================================================================
# Tool Output Pruning
# =============================================================================

def _log(msg: str) -> None:
    """Log to stderr."""
    timestamp = time.strftime("%H:%M:%S")
    print(f"[{timestamp}] [compaction] {msg}", file=sys.stderr, flush=True)


def prune_old_tool_outputs(
    messages: List[Dict[str, Any]],
    protect_last_turns: int = 2,
) -> List[Dict[str, Any]]:
    """
    Prune old tool outputs to save tokens.
    
    Strategy (exactly like OpenCode compaction.ts lines 49-89):
    1. Go backwards through messages
    2. Skip first 2 user turns (most recent)
    3. Accumulate tool output tokens
    4. Once we've accumulated PRUNE_PROTECT (40K) tokens, start marking for prune
    5. Only actually prune if we can recover > PRUNE_MINIMUM (20K) tokens
    
    Args:
        messages: List of messages
        protect_last_turns: Number of recent user turns to skip (default: 2)
        
    Returns:
        Messages with old tool outputs pruned (content replaced with PRUNE_MARKER)
    """
    if not messages:
        return messages
    
    total = 0  # Total tool output tokens seen (going backwards)
    pruned = 0  # Tokens that will be pruned
    to_prune: List[int] = []  # Indices to prune
    turns = 0  # User turn counter
    
    # Go backwards through messages (like OpenCode)
    for msg_index in range(len(messages) - 1, -1, -1):
        msg = messages[msg_index]
        
        # Count user turns
        if msg.get("role") == "user":
            turns += 1
        
        # Skip the first N user turns (most recent)
        if turns < protect_last_turns:
            continue
        
        # Process tool messages
        if msg.get("role") == "tool":
            content = msg.get("content", "")
            
            # Skip already pruned
            if content == PRUNE_MARKER:
                # Already compacted, stop here (like OpenCode: break loop)
                break
            
            estimate = estimate_tokens(content)
            total += estimate
            
            # Once we've accumulated more than PRUNE_PROTECT tokens,
            # start marking older outputs for pruning
            if total > PRUNE_PROTECT:
                pruned += estimate
                to_prune.append(msg_index)
    
    _log(f"Prune scan: {total} total tokens, {pruned} prunable")
    
    # Only prune if we can recover enough tokens
    if pruned <= PRUNE_MINIMUM:
        _log(f"Prune skipped: only {pruned} tokens recoverable (min: {PRUNE_MINIMUM})")
        return messages
    
    _log(f"Pruning {len(to_prune)} tool outputs, recovering ~{pruned} tokens")
    
    # Create new messages with pruned content
    indices_to_prune = set(to_prune)
    result = []
    for i, msg in enumerate(messages):
        if i in indices_to_prune:
            result.append({
                **msg,
                "content": PRUNE_MARKER,
            })
        else:
            result.append(msg)
    
    return result


# =============================================================================
# AI Compaction
# =============================================================================

def run_compaction(
    llm: "LiteLLMClient",
    messages: List[Dict[str, Any]],
    system_prompt: str,
    model: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Compact conversation history using AI summarization.
    
    Process (like Codex):
    1. Send all messages + compaction prompt to LLM
    2. Get summary response
    3. Create new message list:
       - Original system prompt
       - Summary as user message (with prefix)
       - Ready for continuation
    
    Args:
        llm: LLM client for summarization
        messages: Current message history
        system_prompt: Original system prompt to preserve
        model: Model to use (defaults to current)
        
    Returns:
        Compacted message list
    """
    _log("Starting AI compaction...")
    
    # Build compaction request
    compaction_messages = messages.copy()
    compaction_messages.append({
        "role": "user",
        "content": COMPACTION_PROMPT,
    })
    
    try:
        # Call LLM for summary (no tools, just text)
        response = llm.chat(
            compaction_messages,
            model=model,
            max_tokens=4096,  # Summary should be concise
        )
        
        summary = response.text or ""
        
        if not summary:
            _log("Compaction failed: empty response")
            return messages
        
        summary_tokens = estimate_tokens(summary)
        _log(f"Compaction complete: {summary_tokens} token summary")
        
        # Build new message list
        compacted = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": SUMMARY_PREFIX + summary},
        ]
        
        return compacted
        
    except Exception as e:
        _log(f"Compaction failed: {e}")
        # Return original messages if compaction fails
        return messages


# =============================================================================
# Main Context Management
# =============================================================================

def manage_context(
    messages: List[Dict[str, Any]],
    system_prompt: str,
    llm: "LiteLLMClient",
    force_compaction: bool = False,
) -> List[Dict[str, Any]]:
    """
    Main context management function.
    
    Called before each LLM request to ensure context fits.
    
    Strategy:
    1. Estimate current token usage
    2. Log image status
    3. Prune images if over limit
    4. If under threshold, return as-is
    5. Try pruning old tool outputs first
    6. If still over threshold, run AI compaction
    
    Args:
        messages: Current message history
        system_prompt: Original system prompt (preserved through compaction)
        llm: LLM client (for compaction)
        force_compaction: Force compaction even if under threshold
        
    Returns:
        Managed message list (possibly compacted)
    """
    # Log image status
    analyzed, unanalyzed, total = count_images_in_messages(messages)
    _log(f"Image count: {total} (limit: {IMAGE_LIMIT})")
    _log(f"Image status: {analyzed} analyzed, {unanalyzed} unanalyzed, {total} total")
    
    # Prune images if over limit
    if total > IMAGE_LIMIT:
        messages, images_removed, messages_affected = prune_images_from_messages(messages, limit=IMAGE_LIMIT)
        # if images_removed > 0:
        #     _log(f"Image pruning: removing {images_removed} images from {messages_affected} messages")
        #     _log(f"Image pruning complete: removed images from {messages_affected} messages")
    
    total_tokens = estimate_total_tokens(messages)
    usable = get_usable_context()
    usage_pct = (total_tokens / usable) * 100
    
    _log(f"Context: {total_tokens} tokens ({usage_pct:.1f}% of {usable})")
    
    # Check if we need to do anything
    if not force_compaction and not is_overflow(total_tokens):
        return messages
    
    _log(f"Context overflow detected, managing...")
    
    # Step 1: Try pruning old tool outputs
    pruned = prune_old_tool_outputs(messages)
    pruned_tokens = estimate_total_tokens(pruned)
    
    if not is_overflow(pruned_tokens) and not force_compaction:
        _log(f"Pruning sufficient: {total_tokens} -> {pruned_tokens} tokens")
        return pruned
    
    # Step 2: Run AI compaction
    _log(f"Pruning insufficient ({pruned_tokens} tokens), running AI compaction...")
    compacted = run_compaction(llm, pruned, system_prompt)
    compacted_tokens = estimate_total_tokens(compacted)
    
    _log(f"Compaction result: {total_tokens} -> {compacted_tokens} tokens")
    
    return compacted
