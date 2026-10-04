"""
mcp_layer/servers/moderation_server.py

MCP server exposing one tool: check_safety - classifies whether a user
question is a loaded/inflammatory accusatory question (about any country,
group, or person), or asks about weapons/military capabilities of an armed
group, that shouldn't be answered directly.
"""

import os
import sys
from typing import Annotated

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from mcp.server.fastmcp import FastMCP
from pydantic import Field

from core.llm_client import chat

mcp = FastMCP("moderation-server")

_CLASSIFIER_PROMPT = """You are a content classifier, not a conversational assistant.
Read the question below and answer with EXACTLY ONE WORD: either
"SAFE" or "SENSITIVE".

Mark it SENSITIVE if it does any of these, about ANY country, group,
religion, or person (apply the same standard no matter who is named):
- Asks to confirm/deny an inflammatory label (e.g. "is X a terrorist
  state/organization?", "is X evil?").
- Asks for a one-sided accusatory judgment presented as fact.
- Asks for hateful, defamatory, or harassing content about a group or person.
- Asks about weapons, munitions, arms types, or military capabilities used
  by a specific armed group, militia, or party in an active conflict
  (e.g. "what weapons does group X use", "what bombs/missiles does X have").

Mark it SAFE for ordinary factual, informational, or neutral questions -
including questions that ask about controversies, conflicts, or criticism
in a neutral, informational way (e.g. "what is the controversy about X",
general history of a conflict without focusing on weapons/capabilities).

Question: {question}

Answer with exactly one word: SAFE or SENSITIVE."""


@mcp.tool()
def check_safety(
    question: Annotated[str, Field(description="The user's question to classify")],
) -> str:
    """Classify whether a question is a loaded/inflammatory accusatory question that should be deflected."""
    try:
        response = chat([
            {"role": "user", "content": _CLASSIFIER_PROMPT.format(question=question)}
        ])
        verdict = response.get("message", {}).get("content", "").strip().upper()
        return "SENSITIVE" if "SENSITIVE" in verdict else "SAFE"
    except Exception:
        return "SAFE"


if __name__ == "__main__":
    mcp.run()