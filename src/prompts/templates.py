"""Prompt templates for SuperAgent."""

from __future__ import annotations

# Template for plan updates
PLAN_UPDATE_TEMPLATE = """
Current plan:
{current_plan}

Please update the plan based on recent actions and findings.
"""

# Template for reasoning
REASONING_TEMPLATE = """
Please analyze the current situation and decide on the next steps.
Consider:
1. What has been done so far?
2. What information is missing?
3. What is the most efficient way to proceed?
"""

# Template for self-verification before task completion
VERIFICATION_PROMPT_TEMPLATE = """<system-reminder>
Quick verification before completing. Re-read the original task:

{instruction}

Run 1-2 commands to verify the key outputs exist and are correct. Also clean up: remove any intermediate artifacts you created during testing (compiled binaries, temp files, build outputs, test scripts) from output directories — only the requested deliverables should remain. List the output directory to confirm. If everything checks out, call finish() immediately. If something is wrong, fix it. Do NOT re-do work you already verified. Be concise — this should take ONE turn.
</system-reminder>"""

# Template for shell command timeout errors
SHELL_TIMEOUT_TEMPLATE = """Command timed out after {timeout}s.

The command may still be running in the background. Consider:
1. Check if the process is still running: `ps aux | grep <process>`
2. Increase timeout if the operation legitimately needs more time
3. Check if the command is waiting for input (use -y flags, heredocs, etc.)
4. Break the command into smaller steps

Partial output before timeout:
{output}"""

# Template for shell command execution errors
SHELL_ERROR_TEMPLATE = """Command failed with exit code {exit_code}.

Error analysis:
1. What does this error message indicate?
2. Is this a syntax error, missing dependency, permission issue, or logic error?
3. What is the root cause?
4. How can you fix it?

Output:
{output}"""


# Template for tool invalid parameters guidance
TOOL_INVALID_GUIDANCE_TEMPLATE = """
---
[INVALID TOOL PARAMETERS] The tool '{tool_name}' was called with invalid or missing parameters.

{error_details}

Please return the correct tool call with all required parameters properly specified."""

# Template for tool failure guidance
TOOL_FAILURE_GUIDANCE_TEMPLATE = """
---
[TOOL CALL FAILED] The tool '{tool_name}' did not execute successfully.

Please analyze and fix:
1. Check the error message above - what does it indicate?
2. Verify all required parameters were provided correctly
3. If it's a file/path issue: check if the path exists and is correct
4. If it's a command issue: verify the command syntax is valid
5. If parameters are missing: review the tool's required parameters

Next action: Fix the issue and retry with the correct tool call, or use an alternative approach."""
