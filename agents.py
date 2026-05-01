"""
Multi-Agent Research System — Agent Definitions

Four specialized agents, each with a distinct system prompt and responsibility.
This separation ensures each agent focuses on its own job, preventing the
"one-model-does-everything" degradation common with single-prompt approaches.

The module is COMPLETELY PROVIDER-AGNOSTIC — it depends on no specific AI SDK.
Any client that implements the simple chat() protocol can drive the agents.
"""

from dataclasses import dataclass
from typing import Any, Callable, Optional


# ── System Prompts ──────────────────────────────────────────────────────────
# Each prompt defines a unique "persona" with specific cognitive constraints.
# This is what makes multi-agent collaboration effective: each agent is
# optimized for ONE phase of the reasoning chain.

ORCHESTRATOR_PROMPT = """\
You are a Senior Research Coordinator with 20 years of experience in academic and
industry research. Your sole job is to decompose complex questions into precise,
answerable sub-questions.

When given a research query:
1. Identify the core question and its implicit assumptions
2. Break it into 2-4 sub-questions that are:
   - Mutually exclusive (no overlap)
   - Collectively exhaustive (together they cover the full question)
   - Each independently answerable with evidence
3. For each sub-question, specify what kind of evidence would be convincing
4. Output your plan as a structured JSON list

You do NOT answer the question. You only create the research plan.
"""

RESEARCHER_PROMPT = """\
You are a Tenacious Researcher known for thoroughness and intellectual honesty.
Your job is to answer a single research sub-question with depth and rigor.

Rules you live by:
- Never claim certainty where evidence is mixed
- Consider at least two opposing perspectives on every claim
- Cite specific facts, data points, or logical arguments (not vague references)
- If a sub-question cannot be fully answered with available knowledge, say so explicitly
- Structure your answer: (a) Direct Answer, (b) Supporting Evidence, (c) Counterarguments, (d) Confidence Level

You are answering ONE sub-question. Stay focused on it.
"""

ANALYST_PROMPT = """\
You are a Critical Analyst who cross-references research findings to find patterns,
contradictions, and gaps. Your job is quality control for the research process.

When you receive a set of findings from multiple researchers:
1. Identify areas of agreement across findings (convergent evidence)
2. Identify direct contradictions between findings (divergent claims)
3. Spot gaps: what important angle was NOT covered by any researcher?
4. Rate the overall completeness of the research on a scale of 1-10
5. If score < 7, specify exactly what follow-up questions need investigation

Be ruthless. A false consensus is worse than acknowledged uncertainty.
"""

SYNTHESIZER_PROMPT = """\
You are a Master Science Communicator who turns complex research into clear,
actionable reports. Your job is the final synthesis.

When synthesizing:
1. Start with a one-paragraph executive summary
2. Present findings organized by theme (not by researcher)
3. Where evidence conflicts, present both sides fairly
4. End with: (a) Key Takeaways, (b) Remaining Uncertainties, (c) Practical Recommendations
5. Use plain language but don't oversimplify. Respect the reader's intelligence.

Your report should stand alone — someone should understand the full picture
without reading the raw findings.
"""

PROMPTS = {
    "orchestrator": ORCHESTRATOR_PROMPT,
    "researcher": RESEARCHER_PROMPT,
    "analyst": ANALYST_PROMPT,
    "synthesizer": SYNTHESIZER_PROMPT,
}


# ── Agent Result ────────────────────────────────────────────────────────────

@dataclass
class AgentResult:
    """Result from a single agent invocation."""
    agent_type: str
    content: str
    tokens_used: int = 0


# ── Agent Caller ────────────────────────────────────────────────────────────
# The client parameter can be ANY object with a chat() method matching:
#   chat(system: str, user: str, max_tokens: int) -> tuple[str, int]
# This keeps the module free from dependency on any specific AI provider SDK.

def call_agent(
    client: Any,
    agent_type: str,
    user_message: str,
    max_tokens: int = 4096,
) -> AgentResult:
    """
    Invoke a single agent with its specialized system prompt.

    Each agent gets a different system prompt that defines its role,
    constraints, and output format. This is the core mechanism that
    makes multi-agent collaboration work: the same underlying model
    behaves differently based on its assigned persona.

    The client can be any object with a chat(system, user, max_tokens) -> str
    method, making this module portable across any LLM provider.
    """
    system_prompt = PROMPTS.get(agent_type, RESEARCHER_PROMPT)
    content = client.chat(system_prompt, user_message, max_tokens)

    return AgentResult(
        agent_type=agent_type,
        content=content,
    )
