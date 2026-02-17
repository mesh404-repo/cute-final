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
# Self-Verification Required - CRITICAL

You indicated the task might be complete. Before finishing, you MUST perform a thorough self-verification.

## Original Task (re-read carefully):
{instruction}

## Self-Verification Checklist:

### 1. Requirements Analysis
- Re-read the ENTIRE original task above word by word
- List EVERY requirement, constraint, and expected outcome mentioned
- Check if there are any implicit requirements you might have missed

### 2. Work Verification  
- For EACH requirement identified, verify it was completed:
  - Run commands to check file contents, test outputs, or verify state
  - Do NOT assume something works - actually verify it
  - If you created code, run it to confirm it works
  - If you modified files, read them back to confirm changes are correct
  
### 3. Component & Functionality Testing
1. Identify ALL components, modules, or functions mentioned in the task
2. For EACH component, write and run a test command that actually CALLS/EXERCISES the functionality (not just imports)
3. Run any custom tests you created to validate edge cases
4. If ANY test fails, analyze the failure, fix your solution, and re-run the tests
5. DO NOT give up if tests fail - iterate until all tests pass

### 4. Edge Cases & Quality
- Are there any edge cases the task mentioned that you haven't handled?
- Did you follow any specific format/style requirements mentioned?
- Are there any errors, warnings, or issues in your implementation?

### 5. Final Decision
After completing the above verification:
- If EVERYTHING is verified and correct (including ALL critical constraints): Summarize what was done and confirm completion
- If ANY constraint is violated or ANYTHING is missing or broken: Fix it now using the appropriate tools
- If files are in wrong location: MOVE them to correct location

## CRITICAL REMINDERS:
- You are running in HEADLESS mode - DO NOT ask questions to the user
- DO NOT ask for confirmation or clarification - make reasonable decisions
- If something is ambiguous, make the most reasonable choice and proceed
- If you find issues during verification, FIX THEM before completing
- Only complete if you have VERIFIED (not assumed) that everything works

Proceed with verification now.
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
