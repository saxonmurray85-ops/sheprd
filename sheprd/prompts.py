"""
Centralized Prompt Generation for Sheprd Agents.
Ensures consistent persona, identity, job constraints, and tool invocation
instructions across interactive terminal chat, Telegram bots, Web UI, and
peer-to-peer agent delegation.
"""

from typing import Optional
from .database import AgentRecord


def build_agent_system_prompt(
    agent: AgentRecord,
    caller_name: Optional[str] = None,
    extra_instructions: Optional[str] = None,
) -> str:
    """
    Builds a standardized system prompt embedding identity, personality,
    job responsibilities, and tool calling guidance.
    """
    base = (
        f"You are {agent.name}.\n"
        f"IDENTITY: {agent.identity}\n"
        f"PERSONALITY: {agent.personality}\n"
        f"PRIMARY JOB: {agent.job}\n\n"
    )

    if caller_name:
        base += (
            f"You have received a delegated request from peer agent '{caller_name}'. "
            f"Provide an expert, professional response fulfilling your specific role."
        )
    elif extra_instructions:
        base += extra_instructions.strip()
    else:
        base += (
            "Instructions:\n"
            "- Speak consistently in your specified identity and personality.\n"
            "- Stay dedicated to your assigned job.\n"
            "- Use your available tools whenever real-time data, calculations, or external facts are needed.\n"
            "- Be clear, insightful, and direct."
        )

    return base
