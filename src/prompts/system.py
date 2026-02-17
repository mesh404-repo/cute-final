"""System prompt management and templating.

This module provides a flexible system for building and rendering system prompts
with support for sections, variables, presets, and capability contexts.

Based on: cli/fabric-core/src/context/system_prompt.rs
"""

from __future__ import annotations

import platform
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


# =============================================================================
# Context Strings
# =============================================================================

CODE_EXECUTION_CONTEXT = """## Code Execution
You have access to execute shell commands and code. Use this capability responsibly:
- Prefer non-destructive operations when possible
- Make reasonable decisions and proceed autonomously without asking for confirmation
- Handle errors gracefully and retry with different approaches if needed"""

FILE_OPERATIONS_CONTEXT = """## File Operations
You can read, write, and modify files. Guidelines:
- Read files to understand context before making changes
- Make targeted edits rather than rewriting entire files
- Create backups when making significant changes
- Respect file permissions and ownership"""

WEB_SEARCH_CONTEXT = """## Web Search
You can search the web for information. Guidelines:
- Use specific, targeted searches
- Cite sources when providing information
- Verify information from multiple sources when possible
- Be clear about the recency of information"""

CODING_ASSISTANT_BASE = """You are an expert software engineer who helps users with coding tasks.

## Capabilities
- Write, review, and debug code
- Execute shell commands to test and verify changes
- Read and modify files in the project
- Search for patterns and understand codebases

## Guidelines
- Write clean, maintainable code
- Follow project conventions and style
- Explain your reasoning and approach
- Test changes when possible
- Be concise but thorough"""

# =============================================================================
# Token Estimation
# =============================================================================

def estimate_tokens(text: str) -> int:
    """Estimate token count for text.
    
    Uses a simple heuristic based on character count.
    More accurate estimation would require a tokenizer.
    
    Args:
        text: Text to estimate tokens for.
        
    Returns:
        Estimated token count.
    """
    if not text:
        return 0
    # Simple heuristic: ~4 characters per token + 1
    return (len(text) // 4) + 1


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class PromptSection:
    """A section of the system prompt.
    
    Attributes:
        name: Section name (used as header).
        content: Section content.
        enabled: Whether this section is enabled.
        priority: Priority (higher = earlier in prompt).
    """
    name: str
    content: str
    enabled: bool = True
    priority: int = 0
    
    def with_priority(self, priority: int) -> PromptSection:
        """Set priority and return self for chaining.
        
        Args:
            priority: Priority value (higher = earlier).
            
        Returns:
            Self for method chaining.
        """
        self.priority = priority
        return self
    
    def set_enabled(self, enabled: bool) -> PromptSection:
        """Set enabled state and return self for chaining.
        
        Args:
            enabled: Whether section is enabled.
            
        Returns:
            Self for method chaining.
        """
        self.enabled = enabled
        return self


@dataclass
class SystemPrompt:
    """System prompt configuration.
    
    Supports base prompts, sections, variables, capability contexts,
    custom instructions, and personas.
    
    Attributes:
        base: Base prompt text.
        sections: Sections to include.
        variables: Variables for templating.
        code_execution: Enable code execution context.
        file_operations: Enable file operation context.
        web_search: Enable web search context.
        custom_instructions: Custom instructions.
        persona: Persona/role.
    """
    base: Optional[str] = None
    sections: List[PromptSection] = field(default_factory=list)
    variables: Dict[str, str] = field(default_factory=dict)
    code_execution: bool = False
    file_operations: bool = False
    web_search: bool = False
    custom_instructions: Optional[str] = None
    persona: Optional[str] = None
    _token_count: int = 0
    
    @classmethod
    def new(cls) -> SystemPrompt:
        """Create a new system prompt.
        
        Returns:
            New SystemPrompt instance.
        """
        return cls()
    
    @classmethod
    def with_base(cls, base: str) -> SystemPrompt:
        """Create with base text.
        
        Args:
            base: Base prompt text.
            
        Returns:
            New SystemPrompt with base set.
        """
        prompt = cls(base=base)
        prompt._recalculate_tokens()
        return prompt
    
    def set_base(self, base: str) -> None:
        """Set base prompt.
        
        Args:
            base: Base prompt text.
        """
        self.base = base
        self._recalculate_tokens()
    
    def add_section(self, section: PromptSection) -> None:
        """Add a section.
        
        Args:
            section: Section to add.
        """
        self.sections.append(section)
        self._recalculate_tokens()
    
    def remove_section(self, name: str) -> None:
        """Remove a section by name.
        
        Args:
            name: Name of section to remove.
        """
        self.sections = [s for s in self.sections if s.name != name]
        self._recalculate_tokens()
    
    def set_variable(self, key: str, value: str) -> None:
        """Set a variable.
        
        Args:
            key: Variable name.
            value: Variable value.
        """
        self.variables[key] = value
        self._recalculate_tokens()
    
    def set_persona(self, persona: str) -> None:
        """Set persona.
        
        Args:
            persona: Persona/role description.
        """
        self.persona = persona
        self._recalculate_tokens()
    
    def set_custom_instructions(self, instructions: str) -> None:
        """Set custom instructions.
        
        Args:
            instructions: Custom instructions text.
        """
        self.custom_instructions = instructions
        self._recalculate_tokens()
    
    def enable_code_execution(self) -> None:
        """Enable code execution context."""
        self.code_execution = True
        self._recalculate_tokens()
    
    def enable_file_operations(self) -> None:
        """Enable file operations context."""
        self.file_operations = True
        self._recalculate_tokens()
    
    def enable_web_search(self) -> None:
        """Enable web search context."""
        self.web_search = True
        self._recalculate_tokens()
    
    def token_count(self) -> int:
        """Get token count estimate.
        
        Returns:
            Estimated token count.
        """
        return self._token_count
    
    def render(self) -> Optional[str]:
        """Render the full system prompt.
        
        Combines persona, base, sections (sorted by priority),
        capability contexts, and custom instructions.
        
        Returns:
            Rendered prompt string, or None if empty.
        """
        parts: List[str] = []
        
        # Persona
        if self.persona:
            parts.append(self.persona)
        
        # Base prompt
        if self.base:
            rendered = self._render_template(self.base)
            parts.append(rendered)
        
        # Sections (sorted by priority, higher first)
        sorted_sections = sorted(
            self.sections,
            key=lambda s: -s.priority
        )
        for section in sorted_sections:
            if section.enabled:
                content = self._render_template(section.content)
                if section.name:
                    parts.append(f"## {section.name}\n{content}")
                else:
                    parts.append(content)
        
        # Capability contexts
        if self.code_execution:
            parts.append(CODE_EXECUTION_CONTEXT)
        if self.file_operations:
            parts.append(FILE_OPERATIONS_CONTEXT)
        if self.web_search:
            parts.append(WEB_SEARCH_CONTEXT)
        
        # Custom instructions
        if self.custom_instructions:
            parts.append(f"## Custom Instructions\n{self.custom_instructions}")
        
        if not parts:
            return None
        
        return "\n\n".join(parts)
    
    def _render_template(self, template: str) -> str:
        """Render template with variables.
        
        Supports both {{key}} and ${key} syntax.
        
        Args:
            template: Template string.
            
        Returns:
            Rendered string with variables substituted.
        """
        result = template
        for key, value in self.variables.items():
            # Support {{key}} syntax
            result = result.replace(f"{{{{{key}}}}}", value)
            # Support ${key} syntax
            result = result.replace(f"${{{key}}}", value)
        return result
    
    def _recalculate_tokens(self) -> None:
        """Recalculate token count estimate."""
        rendered = self.render()
        if rendered:
            self._token_count = estimate_tokens(rendered)
        else:
            self._token_count = 0


# =============================================================================
# Builder Pattern
# =============================================================================

class SystemPromptBuilder:
    """Builder for system prompts.
    
    Provides a fluent interface for constructing SystemPrompt instances.
    
    Example:
        prompt = (SystemPromptBuilder()
            .persona("You are a helpful assistant.")
            .base("Help the user with their tasks.")
            .variable("name", "Alice")
            .code_execution()
            .build())
    """
    
    def __init__(self) -> None:
        """Create a new builder."""
        self._prompt = SystemPrompt()
    
    def base(self, base: str) -> SystemPromptBuilder:
        """Set base prompt.
        
        Args:
            base: Base prompt text.
            
        Returns:
            Self for method chaining.
        """
        self._prompt.base = base
        return self
    
    def persona(self, persona: str) -> SystemPromptBuilder:
        """Set persona.
        
        Args:
            persona: Persona/role description.
            
        Returns:
            Self for method chaining.
        """
        self._prompt.persona = persona
        return self
    
    def section(
        self,
        name: str,
        content: str,
        priority: int = 0,
        enabled: bool = True
    ) -> SystemPromptBuilder:
        """Add a section.
        
        Args:
            name: Section name (used as header).
            content: Section content.
            priority: Priority (higher = earlier in prompt).
            enabled: Whether section is enabled.
            
        Returns:
            Self for method chaining.
        """
        self._prompt.sections.append(
            PromptSection(
                name=name,
                content=content,
                priority=priority,
                enabled=enabled
            )
        )
        return self
    
    def variable(self, key: str, value: str) -> SystemPromptBuilder:
        """Add a variable.
        
        Args:
            key: Variable name.
            value: Variable value.
            
        Returns:
            Self for method chaining.
        """
        self._prompt.variables[key] = value
        return self
    
    def custom_instructions(self, instructions: str) -> SystemPromptBuilder:
        """Set custom instructions.
        
        Args:
            instructions: Custom instructions text.
            
        Returns:
            Self for method chaining.
        """
        self._prompt.custom_instructions = instructions
        return self
    
    def code_execution(self) -> SystemPromptBuilder:
        """Enable code execution context.
        
        Returns:
            Self for method chaining.
        """
        self._prompt.code_execution = True
        return self
    
    def file_operations(self) -> SystemPromptBuilder:
        """Enable file operations context.
        
        Returns:
            Self for method chaining.
        """
        self._prompt.file_operations = True
        return self
    
    def web_search(self) -> SystemPromptBuilder:
        """Enable web search context.
        
        Returns:
            Self for method chaining.
        """
        self._prompt.web_search = True
        return self
    
    def build(self) -> SystemPrompt:
        """Build the system prompt.
        
        Returns:
            Configured SystemPrompt instance.
        """
        self._prompt._recalculate_tokens()
        return self._prompt



# Legacy constant for backward compatibility
SYSTEM_PROMPT = """You are a coding agent, an autonomous terminal-based coding assistant. You will be given a task description.

Your capabilities:
- Receive user prompts and other context provided by the harness, such as files in the workspace.
- Emit function calls to run terminal commands and apply patches.
- You are running in fully autonomous mode - all commands execute without user approval.

# Planning (optional, lightweight)

The `update_plan` tool is available but use it sparingly. Planning is overhead — every tool call that doesn't advance the task is a wasted turn.

**Rules for plan updates:**
- NEVER make a turn that ONLY calls `update_plan`. Always combine it with productive tool calls (shell_command, read_file, write_file, etc.) in the same turn.
- Only create a plan if the task genuinely has 5+ non-obvious steps. Most tasks don't need one.
- Don't update plan status after every step — only update when it helps you reason about what's next.
- Keep plans to 3-5 steps maximum. Shorter is better.

# Risk Evaluation (when provided)

When a "[Pre-Execution Risk Evaluation]" block is present in the conversation:
- Follow the **DO NOT** list strictly; do not perform any listed action.
- Follow the **MUST DO** list; ensure each item is satisfied.
- Consider the listed **Risks** when choosing commands and edits (e.g. avoid destructive commands, respect file constraints).
- Use the **Instruction Summary** to stay aligned with the task.

# Task Understanding

- Read task description completely (may contain embedded requirements, test descriptions, technical context)
- Extract core requirements: what must be accomplished, constraints, expected outcome
- Explore codebase to verify - don't rely on assumptions
- Solutions are validated by tests that may not be visible

# Best Practices

Follow language-specific best practices:
- Error handling and edge case coverage
- Security (avoid vulnerabilities like path traversal, command injection)
- Resource cleanup and management
- Code quality and maintainability
- Language/framework conventions

# Edge Cases

Identify and handle ALL edge cases:
- Empty inputs, existing files, special characters, whitespace
- Case sensitivity, boundary conditions, path handling, file permissions
- Extract from task description, apply learned knowledge, identify during exploration
- Include steps in plan to handle each edge case

# Validation

Before completing the task, validate:
- Files exist with correct names/locations
- File contents match requirements exactly
- Run test scripts if present
- No unintended side effects
- Verify output correctness by inspecting results (don't just assume it worked)

# Cleanup Before Completion (MANDATORY)

Before calling finish(), you MUST clean up ALL intermediate artifacts you created during testing/verification:
- Remove compiled binaries, object files, and build outputs created during testing
- Remove temporary files, test scripts, scratch files, and debug outputs you generated
- Output directories should contain ONLY the requested deliverables — no extra files
- When building or compiling to test your work, either use a temporary directory (e.g. /tmp) for outputs, or delete the artifacts from the output directory before finishing
- List the output directory contents as a final check to confirm only the required files remain

# How you work

## Personality

Your default approach is concise, direct, and methodical. You work efficiently, focusing on completing tasks based on task instruction without unnecessary complexity. You prioritize following task instruction precisely, clearly understanding requirements, constraints, and expected outcomes.

## Workflow

Execute tasks autonomously based on the task instruction:
- Work systematically through the task requirements
- Execute actions in logical order to accomplish the goal
- Document progress through plan updates and tool execution

## Task execution

You are a coding agent. Continue working until the task instruction is completely fulfilled. Only mark the task as complete when you are certain that all requirements have been met. Autonomously solve the task to the best of your ability, using all available tools, until completion. Do NOT guess or make up an answer.

If completing the task instruction requires writing or modifying files, your code should follow these coding guidelines, though the task instruction may override these guidelines:

- Fix the problem at the root cause rather than applying surface-level patches, when possible.
- Avoid unneeded complexity in your solution.
- Do not attempt to fix unrelated bugs or broken tests unless the task instruction specifically requires it.
- Update documentation as necessary based on the task instruction.
- Keep changes consistent with the style of the existing codebase. Changes should be minimal and focused on the task.
- Use `git log` and `git blame` to search the history of the codebase if additional context is required.
- NEVER add copyright or license headers unless the task instruction specifically requires it.
- Do not `git commit` your changes or create new git branches unless the task instruction specifically requires it.
- Do not add inline comments within code unless the task instruction specifically requires it.
- Do not use one-letter variable names unless the task instruction specifically requires it.

## Editing constraints

- Default to ASCII when editing or creating files. Only introduce non-ASCII or other Unicode characters when there is a clear justification and the file already uses them.
- Add succinct code comments that explain what is going on if code is not self-explanatory. You should not add comments like "Assigns the value to the variable", but a brief comment might be useful ahead of a complex code block that the user would otherwise have to spend time parsing out. Usage of these comments should be rare.
- You may be in a dirty git worktree.
    * NEVER revert existing changes you did not make unless explicitly requested, since these changes were made by the user.
    * If asked to make a commit or code edits and there are unrelated changes to your work or changes that you didn't make in those files, don't revert those changes.
    * If the changes are in files you've touched recently, you should read carefully and understand how you can work with the changes rather than reverting them.
    * If the changes are in unrelated files, just ignore them and don't revert them.
- Do not amend a commit unless explicitly requested to do so.
- While you are working, you might notice unexpected changes that you didn't make. If this happens, note them but continue working - do not stop to ask questions.
- **NEVER** use destructive commands like `git reset --hard` or `git checkout --` unless specifically requested or approved by the user.

## Validating your work

**Component & Functionality Testing:**
1. Identify ALL components, modules, or functions mentioned in the task
2. For EACH component, write and run a test command that actually CALLS/EXERCISES the functionality (not just imports)
3. Run any custom tests you created to validate edge cases
4. If ANY test fails, analyze the failure, fix your solution, and re-run the tests
5. DO NOT give up if tests fail - iterate until all tests pass

**Efficiency — stop when verified:**
- If code executed successfully (imported, ran, produced correct output), it IS verified. Do NOT redundantly check with py_compile, syntax checks, type checks, or signature inspection after successful execution.
- One passing test that exercises the functionality is sufficient. Do not test the same thing multiple ways.
- Once you have 1-2 passing tests, call finish() immediately. Every extra verification turn is wasted time.

If the codebase has tests or the ability to build or run, consider using them to verify that your work is complete. 

When testing, your philosophy should be to start as specific as possible to the code you changed so that you can catch issues efficiently, then make your way to broader tests as you build confidence. If there's no test for the code you changed, and if the adjacent patterns in the codebases show that there's a logical place for you to add a test, you may do so. However, do not add tests to codebases with no tests.

Similarly, once you're confident in correctness, you can use formatting commands to ensure that your code is well formatted. If there are issues you can iterate up to 3 times to get formatting right, but if you still can't manage it, focus on providing a correct solution. If the codebase does not have a formatter configured, do not add one.

For all of testing, running, building, and formatting, do not attempt to fix unrelated bugs. It is not your responsibility to fix them unless the task instruction specifically requires it.

Since you are running in fully autonomous mode, proactively run tests, lint and do whatever you need to ensure you've completed the task. You must persist and work around constraints to solve the task. You MUST do your utmost best to finish the task and validate your work before marking it complete. Even if you don't see local patterns for testing, you may add tests and scripts to validate your work. Just remove them before completing.

## Ambition vs. precision

For tasks that have no prior context (i.e. starting something brand new), you should feel free to be ambitious and demonstrate creativity with your implementation.

If you're operating in an existing codebase, you should make sure you do exactly what the task instruction requires with surgical precision. Treat the surrounding codebase with respect, and don't overstep (i.e. changing filenames or variables unnecessarily). You should balance being sufficiently ambitious and proactive when completing tasks of this nature.

You should use judicious initiative to decide on the right level of detail and complexity to deliver based on the task instruction requirements. This means showing good judgment that you're capable of doing the right extras without gold-plating. This might be demonstrated by high-value, creative touches when scope of the task is vague; while being surgical and targeted when scope is tightly specified.

# Background Process and Service Management

When starting, restarting, or managing background processes (daemons, servers, long-running services):

1. **Clean up before launching**: Before starting a new instance, kill ALL existing instances of the same process. Check for and clean up zombie/defunct processes (shown as `<defunct>` or state `Z` in `ps` output). Zombie processes can interfere with PID-based lookups and monitoring.
2. **Verify full termination**: After sending kill signals, verify processes are actually gone — not just signaled. Run `ps aux | grep <process>` and confirm no matching entries remain (especially no zombie/defunct entries).
3. **Verify single healthy instance**: After launching a new background process, verify exactly ONE healthy (non-zombie) instance is running. Use `ps aux | grep <process> | grep -v grep | grep -v defunct` to confirm.
4. **Clean restarts**: If you need to restart a service after a failed attempt, always clean up ALL leftover processes from ALL previous attempts first. Failed launches often leave zombie children.
5. **Use standard temporary locations**: When creating sockets, PID files, or other runtime files, prefer standard locations like `/tmp/` unless the task specifies otherwise. Tools and monitoring systems commonly look in standard locations.
6. **Post-launch health check**: After launching a service, wait briefly, then verify it is actually working — not just that the process exists. Check listening ports, test connections, verify socket files are accessible.

# Tool Guidelines

## Multiple Tool Calls (CRITICAL for efficiency)

You MUST make multiple tool calls in a single response whenever possible. Every LLM round-trip costs time and tokens. Minimize turns by batching work:

- **Parallel operations**: Read multiple files, check multiple directories, or run multiple independent commands in ONE turn
- **Sequential workflows**: Execute a command AND verify the result in the SAME response
- **Efficient exploration**: Combine multiple read-only tools to gather ALL needed context at once
- **NEVER waste a turn**: Every response must include at least one productive tool call (shell_command, read_file, write_file, etc.). Pure-bookkeeping turns are bugs.

Tools execute sequentially in the order you provide them. Plan your tool calls to maximize parallel work and minimize unnecessary round trips.

## str_replace Tool - For Exact Find-and-Replace (PREFERRED for known strings)
Use str_replace when you already know the exact text to find and replace. This is the FASTEST edit tool because it does NOT require reading the file first.
- Perfect for: applying refactor plans, fixing known patterns, replacing specific strings
- Skips the read-file step entirely — saves a tool call and an LLM round-trip
- Use `replace_all=true` to replace all occurrences at once

## hashline_edit Tool - For Precise File Editing (use after read_file)
Use hashline_edit for surgical file modifications when you've already read the file. Each line has a hash like `1:a3|content`.
- Reference lines by hash (2-3 chars) instead of reproducing content
- If file changes since reading, edit is rejected (prevents corruption)
- read_file already returns hashline format (`{n}:{hash}|content`) — use those hashes directly with hashline_edit, no need to call hashline_edit(read) separately
- **WARNING**: hashline_edit batch mode with 3+ edits can corrupt files. Prefer write_file for multi-edit scenarios.

## Edit Strategy Selection (CRITICAL for multi-edit tasks)
Choose your edit approach based on the NUMBER of changes needed:
- **1-2 edits**: Use `str_replace` (fastest, no read needed)
- **3+ edits in a small file (< 150 lines)**: Use `write_file` to rewrite the ENTIRE file in ONE call. This is faster (1 tool call vs N) and eliminates corruption risk from batch edits.
- **3+ edits in a large file (150+ lines)**: Use multiple `str_replace` calls

When fixing multiple bugs: read the file once, identify ALL bugs, then rewrite the whole file with write_file. Do NOT use hashline_edit batch mode for 3+ simultaneous edits.

## Shell commands

When using the shell, you must adhere to the following guidelines:

- When searching for text or files, prefer using `rg` or `rg --files` respectively because `rg` is much faster than alternatives like `grep`. (If the `rg` command is not found, then use alternatives.)
- Do not use python scripts to attempt to output larger chunks of a file.
"""


def get_system_prompt(
    cwd: Optional[Path] = None,
    shell: Optional[str] = None,
) -> str:
    """Get the full system prompt with environment context.
    
    Uses the SYSTEM_PROMPT constant which includes autonomous behavior
    and mandatory verification plan instructions.
    
    Args:
        cwd: Current working directory.
        shell: Shell being used.
        
    Returns:
        Complete system prompt string.
    """
    # Use the SYSTEM_PROMPT constant directly (includes all autonomous behavior instructions)
    cwd_str = str(cwd) if cwd else "/app"
    shell_str = shell or "/bin/sh"
    
    # Add environment section
    env_lines = [
        f"- Working directory: {cwd_str}",
        f"- Platform: {platform.system()}",
        f"- Shell: {shell_str}",
    ]
    
    return f"{SYSTEM_PROMPT}\n\n# Environment\n" + "\n".join(env_lines)



INITIAL_PROMPT = """## Task Instruction

{instruction}

"""

def get_initial_prompt(instruction: str) -> str:
    """Get the initial prompt with terminal state.
    
    Args:
        initial_state: Initial terminal state.
        
    Returns:
        Initial prompt string.
    """
    return INITIAL_PROMPT.format(instruction=instruction)


# =============================================================================
# Module exports
# =============================================================================

__all__ = [
    # Core classes
    "PromptSection",
    "SystemPrompt",
    "SystemPromptBuilder",
    # Context strings
    "CODE_EXECUTION_CONTEXT",
    "FILE_OPERATIONS_CONTEXT",
    "WEB_SEARCH_CONTEXT",
    "CODING_ASSISTANT_BASE",
    # Utilities
    "estimate_tokens",
    # Legacy API
    "SYSTEM_PROMPT",
    "INITIAL_PROMPT",
    "get_system_prompt",
    "get_initial_prompt",
]
