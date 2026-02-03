"""
Main agent loop - the heart of the SuperAgent system.

Implements the agentic loop that:
1. Receives instruction via --instruction argument
2. Calls LLM with tools (using litellm)
3. Executes tool calls
4. Loops until task is complete
5. Emits JSONL events throughout

Context management strategy (like OpenCode/Codex):
- Token-based overflow detection (not message count)
- Tool output pruning (clear old outputs first)
- AI compaction when needed (summarize conversation)
- Stable system prompt for cache hits
"""

from __future__ import annotations

import json
import copy
import time
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from src.llm.client import LLMError, CostLimitExceeded

from src.output.jsonl import (
    emit,
    next_item_id,
    reset_item_counter,
    ThreadStartedEvent,
    TurnStartedEvent,
    TurnCompletedEvent,
    TurnFailedEvent,
    ItemStartedEvent,
    ItemCompletedEvent,
    ErrorEvent,
    make_agent_message_item,
    make_command_execution_item,
    make_file_change_item,
)
from src.prompts.system import get_system_prompt
from src.prompts.templates import (
    VERIFICATION_PROMPT_TEMPLATE,
    VERIFICATION_CONFIRMATION_TEMPLATE,
    TOOL_FAILURE_GUIDANCE_TEMPLATE,
    TOOL_INVALID_GUIDANCE_TEMPLATE,
)
from src.utils.truncate import middle_out_truncate, APPROX_BYTES_PER_TOKEN
from src.core.compaction import (
    manage_context,
    estimate_total_tokens,
    needs_compaction,
)

if TYPE_CHECKING:
    from src.llm.client import LiteLLMClient
    from src.tools.registry import ToolRegistry


def _log(msg: str) -> None:
    """Log to stderr."""
    timestamp = time.strftime("%H:%M:%S")
    print(f"[{timestamp}] [loop] {msg}", file=sys.stderr, flush=True)


def _add_cache_control_to_message(
    msg: Dict[str, Any],
    cache_control: Dict[str, str],
) -> Dict[str, Any]:
    """Add cache_control to a message, converting to multipart if needed."""
    content = msg.get("content")
    
    if isinstance(content, list):
        has_cache = any(
            isinstance(p, dict) and "cache_control" in p
            for p in content
        )
        if has_cache:
            return msg
        
        new_content = list(content)
        for i in range(len(new_content) - 1, -1, -1):
            part = new_content[i]
            # Only add cache_control to non-empty text blocks
            if isinstance(part, dict) and part.get("type") == "text" and part.get("text"):
                new_content[i] = {**part, "cache_control": cache_control}
                break
        return {**msg, "content": new_content}
    
    if isinstance(content, str):
        # Don't add cache_control to empty strings
        if not content:
            return msg
        return {
            **msg,
            "content": [
                {
                    "type": "text",
                    "text": content,
                    "cache_control": cache_control,
                }
            ],
        }
    
    return msg

def _apply_caching(
    messages: List[Dict[str, Any]],
    enabled: bool = True,
) -> List[Dict[str, Any]]:
    """
    Apply prompt caching like OpenCode does:
    - Cache first 2 system messages (stable prefix)
    - Cache last 2 non-system messages (extends cache to cover conversation history)
    
    How Anthropic caching works:
    - Cache is based on IDENTICAL PREFIX
    - A cache_control breakpoint tells Anthropic to cache everything BEFORE it
    - By marking the last messages, we cache the entire conversation history
    - Each new request only adds new messages after the cached prefix
    
    Anthropic limits:
    - Maximum 4 cache_control breakpoints
    - Minimum tokens per breakpoint: 1024 (Sonnet), 4096 (Opus 4.5 on Bedrock)
    
    Reference: OpenCode transform.ts applyCaching()
    """
    if not enabled or not messages:
        return messages
    
    cache_control = {"type": "ephemeral"}
    
    # Separate system and non-system message indices
    system_indices = []
    non_system_indices = []
    
    for i, msg in enumerate(messages):
        if msg.get("role") == "system":
            system_indices.append(i)
        else:
            non_system_indices.append(i)
    
    # Determine which messages to cache:
    # 1. First 2 system messages (stable system prompt)
    # 2. Last 2 non-system messages (extends cache to conversation history)
    # Total: up to 4 breakpoints (Anthropic limit)
    indices_to_cache = set()
    
    # Add first 2 system messages
    for idx in system_indices[:2]:
        indices_to_cache.add(idx)
    
    # Add last 2 non-system messages
    for idx in non_system_indices[-2:]:
        indices_to_cache.add(idx)
    
    # Build result with cache_control added to selected messages
    result = []
    for i, msg in enumerate(messages):
        if i in indices_to_cache:
            result.append(_add_cache_control_to_message(msg, cache_control))
        else:
            result.append(msg)
    
    cached_system = len([i for i in indices_to_cache if i in system_indices])
    cached_final = len([i for i in indices_to_cache if i in non_system_indices])
    
    if indices_to_cache:
        _log(f"Prompt caching: {cached_system} system + {cached_final} final messages marked ({len(indices_to_cache)} breakpoints)")
    
    return result


def run_agent_loop(
    llm: "LiteLLMClient",
    tools: "ToolRegistry",
    ctx: Any,
    config: Dict[str, Any],
) -> None:
    """
    Run the main agent loop.
    
    Args:
        llm: LiteLLM client
        tools: Tool registry with available tools
        ctx: Agent context with instruction, shell(), done()
        config: Configuration dictionary
    """
    # Reset item counter for fresh session
    reset_item_counter()
    
    # Generate session ID
    session_id = f"sess_{int(time.time() * 1000)}"
    
    # 1. Emit thread.started
    emit(ThreadStartedEvent(thread_id=session_id))
    
    # 2. Emit turn.started
    emit(TurnStartedEvent())
    
    # 3. Build initial messages
    cwd = Path(ctx.cwd)
    system_prompt = get_system_prompt(cwd=cwd)
    
    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": ctx.instruction},
    ]
    
    # 4. Get initial terminal state
    _log("Getting initial state...")
    initial_result = ctx.shell("pwd && ls -la")
    max_output_tokens = config.get("max_output_tokens", 2500)
    initial_state = middle_out_truncate(initial_result.output, max_tokens=max_output_tokens)
    
    messages.append({
        "role": "user",
        "content": f"Current directory and files:\n```\n{initial_state}\n```",
    })
    
    # 5. Initialize tracking
    total_input_tokens = 0
    total_output_tokens = 0
    total_cached_tokens = 0
    pending_completion = False
    last_agent_message = ""
    verification_phase: Optional[str] = None  # None | "first" | "confirmation"
    verification_result = ""
    
    max_iterations = config.get("max_iterations", 200)
    cache_enabled = config.get("cache_enabled", True)
    
    # 6. Main loop
    iteration = 0
    total_cost = 0.0
    
    cost_limit = config.get("cost_limit", 100.0)

    concequtive_failed_attempts = 0

    # Keep a deep copy of the last known good state
    prev_messages = copy.deepcopy(messages)

    while iteration < max_iterations:
        iteration += 1
        _log(f"Iteration {iteration}/{max_iterations}")
        
        temperature = 0.0
        try:
            # ================================================================
            # Context Management (replaces sliding window)
            # ================================================================
            # Check token usage and apply pruning/compaction if needed
            context_messages = manage_context(
                messages=messages,
                system_prompt=system_prompt,
                llm=llm,
            )
            
            # If compaction happened, update our messages reference
            if len(context_messages) < len(messages):
                _log(f"Context compacted: {len(messages)} -> {len(context_messages)} messages")
                messages = context_messages
            
            # ================================================================
            # Apply caching (system prompt only for stability)
            # ================================================================
            cached_messages = _apply_caching(context_messages, enabled=cache_enabled)
            # Get tool specs
            tool_specs = tools.get_tools_for_llm()
            
            # ================================================================
            # Call LLM with retry logic
            # ================================================================
            max_retries = 5
            response = None
            last_error = None
            
            for attempt in range(1, max_retries + 1):
                try:
                    # Build extra_body - only include reasoning for models that support it
                    extra_body = {}
                    reasoning_effort = config.get("reasoning_effort", "none")
                    if reasoning_effort and reasoning_effort != "none":
                        extra_body["reasoning"] = {"effort": reasoning_effort}
                    
                    response = llm.chat(
                        cached_messages,
                        tools=tool_specs,
                        max_tokens=config.get("max_tokens", 16384),
                        extra_body=extra_body if extra_body else None,
                    )
                    
                    prev_messages = copy.deepcopy(messages)

                    total_cost += response.cost

                    _log(f"current cost: ${response.cost:.4f} total cost: ${total_cost:.4f}")
                    
                    # Track token usage from response
                    if hasattr(response, "tokens") and response.tokens:
                        tokens = response.tokens
                        if isinstance(tokens, dict):
                            total_input_tokens += tokens.get("input", 0)
                            total_output_tokens += tokens.get("output", 0)
                            total_cached_tokens += tokens.get("cached", 0)
                    
                    break  # Success, exit retry loop
                    
                except CostLimitExceeded:
                    raise  # Don't retry cost limit errors
                    
                except LLMError as e:
                    last_error = e
                    error_msg = str(e.message) if hasattr(e, 'message') else str(e)
                    _log(f"LLM error (attempt {attempt}/{max_retries}): {e.code} - {error_msg}")
                    
                    # Don't retry authentication errors
                    if e.code in ("authentication_error", "invalid_api_key"):
                        raise
                    
                    if "BadRequestError" in error_msg:
                        _log("BadRequestError")
                        
                        messages = copy.deepcopy(prev_messages)
                        cached_messages = _apply_caching(messages, enabled=cache_enabled)

                    if attempt < max_retries:
                        wait_time = 2 * attempt  # 10s, 20s, 30s, 40s
                        _log(f"Retrying in {wait_time} seconds...")
                        time.sleep(wait_time)
                    else:
                        raise
                        
                except Exception as e:
                    last_error = e
                    error_msg = str(e)
                    _log(f"Unexpected error (attempt {attempt}/{max_retries}): {type(e).__name__}: {error_msg}")                    
                    
                    if attempt < max_retries:
                        wait_time = 2 * attempt
                        _log(f"Retrying in {wait_time} seconds...")
                        time.sleep(wait_time)
                    else:
                        raise
            
        except CostLimitExceeded as e:
            _log(f"Cost limit exceeded: {e}")
            emit(TurnFailedEvent(error={"message": f"Cost limit exceeded: {e}"}))
            ctx.done()
            return
            
        except LLMError as e:
            _log(f"LLM error (fatal): {e.code} - {e.message}")
            emit(TurnFailedEvent(error={"message": str(e)}))          
            
            continue
        
        except Exception as e:
            _log(f"Unexpected error (fatal): {type(e).__name__}: {e}")
            emit(TurnFailedEvent(error={"message": str(e)}))
            continue
        
        # Process response text
        response_text = response.text or ""
        
        if response_text:
            last_agent_message = response_text
            
            # Emit agent message
            item_id = next_item_id()
            emit(ItemCompletedEvent(
                item=make_agent_message_item(item_id, response_text)
            ))
        
        # Check for function calls
        has_function_calls = response.has_function_calls() if hasattr(response, "has_function_calls") else bool(response.function_calls)
        
        if not has_function_calls:
            # No tool calls - agent thinks it's done or verification/confirmation complete
            _log("No tool calls in response")

            _log(f"response_text: {response_text}")

            if total_cost >= cost_limit:
                break

            # Verification workflow: first → confirmation → complete
            if verification_phase == "confirmation":
                # LLM confirmed using previous verification (or did missing-only checks)
                _log("Task completion confirmed after verification confirmation")

                if "task incomplete" in response_text.lower():
                    verification_phase = None
                    messages.append({"role": "assistant", "content": response_text})
                    messages.append({
                        "role": "user",
                        "content": "The task is incomplete. Please use the appropriate tools to complete the task. Address any missing verifications, unmet requirements, or unresolved issues you identified, then continue working until the task is done.",
                    })
                    _log("Task incomplete – requesting completion via tools")
                    continue
                break

            if verification_phase == "first":
                # First verification round just completed – store result, request confirmation
                verification_result = response_text
                verification_phase = "confirmation"
                messages.append({"role": "assistant", "content": response_text})

                confirmation_prompt = VERIFICATION_CONFIRMATION_TEMPLATE.format(
                    instruction=ctx.instruction,
                    previous_verification_result=verification_result,
                )
                messages.append({
                    "role": "user",
                    "content": confirmation_prompt,
                })
                _log("Requesting confirmation: use previous verification, confirm or fix gaps only")
                continue

            # No verification yet – request first self-verification
            verification_phase = "first"
            messages.append({"role": "assistant", "content": response_text})

            verification_prompt = VERIFICATION_PROMPT_TEMPLATE.format(
                instruction=ctx.instruction
            )
            messages.append({
                "role": "user",
                "content": verification_prompt,
            })
            _log("Requesting self-verification before completion")
            continue        
        
        # Add assistant message with tool calls
        assistant_msg: Dict[str, Any] = {"role": "assistant", "content": response_text}
        
        # Build tool_calls for message history
        tool_calls_data = []
        for call in response.function_calls:
            tool_calls_data.append({
                "id": call.id,
                "type": "function",
                "function": {
                    "name": call.name,
                    "arguments": json.dumps(call.arguments) if isinstance(call.arguments, dict) else call.arguments,
                },
            })
        
        if tool_calls_data:
            assistant_msg["tool_calls"] = tool_calls_data
        
        messages.append(assistant_msg)

        # Execute each tool call and collect results
        # We must add ALL tool results before any other messages (Anthropic API requirement)
        tool_results = []
        pending_images = []
        
        
        for call in response.function_calls:
            tool_name = call.name
            tool_args = call.arguments if isinstance(call.arguments, dict) else {}
            
            _log(f"tool name: {tool_name}")
            _log(f"tool args: {tool_args}")
            
            # Emit item.started
            item_id = next_item_id()
            emit(ItemStartedEvent(
                item=make_command_execution_item(
                    item_id=item_id,
                    command=f"{tool_name}({tool_args})",
                    status="in_progress",
                )
            ))
            
            # Execute tool
            result = tools.execute(ctx, tool_name, tool_args)
            
            # Get output with error information if failed
            raw_output = result.to_message()
            
            # If tool failed, add guidance to help LLM correct the issue
            if result.invalid_param:
                _log("Tool result is invalid")
                
                # Extract missing parameter info from error message
                error_details = result.error or "Invalid parameters provided"
                raw_output += TOOL_INVALID_GUIDANCE_TEMPLATE.format(
                    tool_name=tool_name,
                    error_details=error_details
                )

            elif not result.success:
                concequtive_failed_attempts += 1
                _log(f"Consecutive failed attempts: {concequtive_failed_attempts}")

                raw_output += TOOL_FAILURE_GUIDANCE_TEMPLATE.format(tool_name=tool_name)
            else:
                concequtive_failed_attempts = 0
            
            # Truncate output using middle-out (keeps beginning and end)
            output = middle_out_truncate(raw_output or "no output", max_tokens=max_output_tokens)

            emit(ItemCompletedEvent(
                item=make_command_execution_item(
                    item_id=item_id,
                    command=f"{tool_name}",
                    status="completed" if result.success else "failed",
                    aggregated_output=output,
                    exit_code=0 if result.success else 1,
                )
            ))
            
            # Collect tool result
            tool_results.append({
                "role": "tool",
                "tool_call_id": call.id,
                "content": output,
                "tool_name": tool_name,
                "invalid": result.invalid_param,
            })
            
            # Collect image for later (after all tool results)
            if result.inject_content:
                pending_images.append({
                    "tool_name": tool_name,
                    "content": result.inject_content,
                })        
        
        # Add ALL tool results first (required by Anthropic API)
        for tool_result in tool_results:
            messages.append({
                "role": tool_result.get("role", "user"),
                "tool_call_id": tool_result.get("tool_call_id", None),
                "content": tool_result.get("content", ""),
            })
            if tool_result.get("invalid", False):
                messages.append({
                    "role": "user",
                    "content": f"The tool '{tool_result.get('tool_name', '')}' was called with invalid parameters. Please review the error above and return a corrected tool call with all required parameters properly specified."
                })        
        
        if total_cost >= cost_limit:
            break
        # Now add any images as a separate user message (after tool results)
        # Limit to max 5 images per turn to avoid hitting API limits
        MAX_IMAGES_PER_TURN = 5
        if pending_images:
            images_to_add = pending_images[:MAX_IMAGES_PER_TURN]
            if len(pending_images) > MAX_IMAGES_PER_TURN:
                _log(f"Limiting images: {len(pending_images)} requested, adding {MAX_IMAGES_PER_TURN}")
            
            image_content = []
            for img in images_to_add:
                image_content.append({"type": "text", "text": f"Image from {img['tool_name']}:"})
                image_content.append(img["content"])
            
            messages.append({
                "role": "user",
                "content": image_content,
            })
            
            # Immediately prune if we've exceeded image limits
            # This prevents hitting API limits before next manage_context() call
            from src.core.compaction import count_total_images, prune_old_images, MAX_IMAGES_PER_REQUEST
            total_imgs = count_total_images(messages)
            if total_imgs > MAX_IMAGES_PER_REQUEST - 10:  # Leave buffer
                _log(f"Immediate image prune: {total_imgs} images in context")
                messages = prune_old_images(messages)
    
    # 7. Emit turn.completed
    emit(TurnCompletedEvent(usage={
        "input_tokens": total_input_tokens,
        "cached_input_tokens": total_cached_tokens,
        "output_tokens": total_output_tokens,
    }))
    
    _log(f"Loop complete after {iteration} iterations")
    _log(f"Tokens: {total_input_tokens} input, {total_cached_tokens} cached, {total_output_tokens} output")
    ctx.done()
