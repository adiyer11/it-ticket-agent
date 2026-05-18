"""
agent/router.py
LLM-orchestrated agent that routes classified tickets to the correct handler.
Uses LangChain tool-calling so the agent decides which resolution action to invoke.
"""

from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from agent.handlers import (
    handle_password_reset,
    handle_access_request,
    handle_hardware,
    handle_software_install,
    handle_onboarding,
)

CONFIDENCE_THRESHOLD = 0.75

llm = ChatOpenAI(model="gpt-4o", temperature=0)

SYSTEM = """You are an IT support agent. Given a support ticket and its predicted category,
use the appropriate tool to resolve it. Be concise and action-oriented.
If the ticket is ambiguous or the category seems wrong, use the escalate tool."""


@tool
def password_reset(ticket: str) -> str:
    """Handle a password or MFA reset request."""
    return handle_password_reset(ticket)


@tool
def access_request(ticket: str) -> str:
    """Handle a software access or permissions request."""
    return handle_access_request(ticket)


@tool
def hardware_issue(ticket: str) -> str:
    """Handle a hardware or device issue."""
    return handle_hardware(ticket)


@tool
def software_install(ticket: str) -> str:
    """Handle a software installation request."""
    return handle_software_install(ticket)


@tool
def onboarding(ticket: str) -> str:
    """Handle a new employee onboarding request."""
    return handle_onboarding(ticket)


@tool
def escalate(ticket: str, reason: str) -> str:
    """Escalate ticket to a human IT agent."""
    return f"[ESCALATED] Ticket routed to human queue. Reason: {reason}"


TOOLS = [password_reset, access_request, hardware_issue, software_install, onboarding, escalate]

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM),
    ("human", "Ticket: {ticket}\nPredicted category: {category}\nConfidence: {confidence}"),
    ("placeholder", "{agent_scratchpad}"),
])

agent = create_tool_calling_agent(llm, TOOLS, prompt)
executor = AgentExecutor(agent=agent, tools=TOOLS, verbose=False)


def route(ticket: str, category: str, confidence: float) -> dict:
    """Route a ticket through the agent. Escalate if confidence is below threshold."""
    if confidence < CONFIDENCE_THRESHOLD:
        return {
            "resolved": False,
            "escalated": True,
            "resolution": f"Low confidence ({confidence:.2f}) — routed to human queue.",
        }

    result = executor.invoke({
        "ticket": ticket,
        "category": category,
        "confidence": confidence,
    })

    escalated = "[ESCALATED]" in result["output"]
    return {
        "resolved": not escalated,
        "escalated": escalated,
        "resolution": result["output"],
    }
