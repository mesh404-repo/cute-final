"""
Risk evaluation before task execution.

Runs one LLM call to analyze the instruction and workspace state for
irreversible actions, traps, and risks. The report is injected into the
conversation so the main agent can follow DO NOT / MUST DO lists.

Reference: _echo_term RiskEvaluatorAgent and pre-execution flow.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict

if TYPE_CHECKING:
    from src.llm.client import LLMClient


def _log(msg: str) -> None:
    timestamp = time.strftime("%H:%M:%S")
    print(f"[{timestamp}] [risk_eval] {msg}", file=sys.stderr, flush=True)


# Model used for risk evaluation (fast, cheap)
GLM_5_TEE = "zai-org/GLM-5-TEE"
KIMI_2_5_TEE = "moonshotai/Kimi-K2.5-TEE"
GLM_4_6_TEE = "zai-org/GLM-4.6-TEE"
GLM_4_7_TEE = "zai-org/GLM-4.7-TEE"

RISK_EVAL_MODELS = [KIMI_2_5_TEE, GLM_4_7_TEE, GLM_4_6_TEE]

RISK_EVALUATION_TOOL = {
    "name": "submit_risk_evaluation",
    "description": "Submit the risk evaluation result. Call this with your analysis.",
    "parameters": {
        "type": "object",
        "properties": {
            "instruction_summary": {
                "type": "string",
                "description": "1-2 sentence summary of the instruction.",
            },
            "overall_level": {
                "type": "string",
                "enum": ["high", "medium", "low"],
                "description": "Overall risk level.",
            },
            "risks": {
                "type": "array",
                "description": "Top risks, at most 5.",
                "items": {
                    "type": "object",
                    "properties": {
                        "severity": {"type": "string", "enum": ["high", "medium", "low"]},
                        "description": {"type": "string"},
                    },
                    "required": ["severity", "description"],
                },
                "maxItems": 5,
            },
            "do_not_list": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Things the agent MUST NOT do.",
            },
            "must_do_list": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Things the agent MUST do.",
            },
        },
        "required": ["instruction_summary", "overall_level", "risks", "do_not_list", "must_do_list"],
    },
}

RISK_EVALUATOR_SYSTEM = """You are a RISK EVALUATION agent for an autonomous coding agent. You run BEFORE the main agent starts.

Analyze the INSTRUCTION and PROJECT STATE. Identify the TOP 5 risks that could cause failure or acting against the instruction.

Categories: irreversible actions (deletes, overwrites, destructive commands), instruction misinterpretation, project-specific (files not to modify, deps, config), traps/pitfalls.

Be specific (file names, exact constraints). You MUST call submit_risk_evaluation with:
- instruction_summary: 1-2 sentences
- overall_level: high, medium, or low
- risks: at most 5 items, each with severity (high/medium/low) and description
- do_not_list: 3-5 things the agent MUST NOT do
- must_do_list: 3-5 things the agent MUST do

Do NOT output text. ONLY call submit_risk_evaluation."""

RISK_PATTERN = ["[high]", "[medium]", "[low]"]


def _has_risk_tag(text: str) -> bool:
    lines = [l.strip().lower() for l in text.strip().split("\n") if l.strip()]
    for line in lines[-3:]:
        if any(tag == line or line.endswith(tag) for tag in RISK_PATTERN):
            return True
    return False


def _risk_args_to_markdown(args: Dict[str, Any]) -> str:
    """Format structured risk result as markdown for downstream."""
    lines = ["# Risk Evaluation", ""]
    lines.append("## Instruction Summary")
    lines.append(args.get("instruction_summary", ""))
    lines.append("")
    risks = args.get("risks") or []
    if risks:
        lines.append("## Risks (top 5)")
        for r in risks:
            sev = r.get("severity", "medium")
            desc = r.get("description", "")
            lines.append(f"- [{sev}] {desc}")
        lines.append("")
    do_not = args.get("do_not_list") or []
    if do_not:
        lines.append("## DO NOT")
        for x in do_not:
            lines.append(f"- {x}")
        lines.append("")
    must_do = args.get("must_do_list") or []
    if must_do:
        lines.append("## MUST DO")
        for x in must_do:
            lines.append(f"- {x}")
        lines.append("")
    overall = args.get("overall_level", "medium")
    lines.append(f"[{overall}]")
    return "\n".join(lines)


def evaluate_risk(
    llm: "LLMClient",
    instruction: str,
    workspace_state: str,
    cwd: str = ".",
    *,
    max_tokens: int = 4096,
    max_retries: int = 5,
    write_md: bool = False,
) -> str:
    """
    Run risk evaluation on the task instruction and workspace state.

    Args:
        llm: LLM client.
        instruction: Task instruction.
        workspace_state: Truncated pwd/ls or similar workspace snapshot.
        cwd: Working directory (used only if write_md is True).
        max_tokens: Max tokens for the risk-eval response.
        max_retries: Retries on failure.
        write_md: If True, write report to cwd/risk_evaluation.md.

    Returns:
        Markdown risk report (instruction summary, risks, DO NOT, MUST DO, overall level).
    """
    from src.llm.client import CostLimitExceeded

    user_prompt = (
        f"Analyze this task for risks.\n\n"
        f"## Instruction\n{instruction}\n\n"
        f"## Workspace State\n```\n{workspace_state[:3000]}\n```\n\n"
        f"Call submit_risk_evaluation with your analysis (top 5 risks, do_not_list, must_do_list, overall_level)."
    )
    model = RISK_EVAL_MODELS[0]
    for attempt in range(1, max_retries + 1):
        try:
            _log(f"Risk evaluation attempt {attempt}/{max_retries}...")
            response = llm.chat(
                messages=[
                    {"role": "system", "content": RISK_EVALUATOR_SYSTEM},
                    {"role": "user", "content": user_prompt},
                ],
                tools=[RISK_EVALUATION_TOOL],
                max_tokens=max_tokens,
                model=model,
            )
            if response.function_calls:
                for fc in response.function_calls:
                    if fc.name == "submit_risk_evaluation":
                        args = json.loads(fc.arguments) if isinstance(fc.arguments, str) else fc.arguments
                        result = _risk_args_to_markdown(args)
                        if write_md:
                            out_path = Path(cwd) / "risk_evaluation.md"
                            out_path.parent.mkdir(parents=True, exist_ok=True)
                            out_path.write_text(result, encoding="utf-8")
                            _log(f"Wrote {out_path} ({len(result)} chars)")
                        return result
            # Fallback: no tool call
            text = (response.text or "").strip()
            if text:
                if not _has_risk_tag(text):
                    text += "\n\n[medium]"
                if write_md:
                    out_path = Path(cwd) / "risk_evaluation.md"
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    out_path.write_text(text, encoding="utf-8")
                return text
            raise ValueError("Empty Content")
        except CostLimitExceeded:
            raise
        except Exception as e:
            _log(f"Risk evaluation attempt {attempt}/{max_retries} failed: {e}")

            model_index = RISK_EVAL_MODELS.index(model) if model in RISK_EVAL_MODELS else -1
            model = RISK_EVAL_MODELS[(model_index + 1) % len(RISK_EVAL_MODELS)]

            if attempt < max_retries:
                time.sleep(min(4 * attempt, 60))
            else:
                return f"[Risk evaluation error after {max_retries} attempts: {e}]"
    return "[Risk evaluation error: unknown]"
