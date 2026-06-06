from google.adk.agents.llm_agent import Agent
from google.adk.agents.sequential_agent import SequentialAgent

from grounded_doc_agent.agents.tools import (
    get_conflict_report,
    get_ingestion_status,
    query_documents,
)

research_instruction = """
You are the retrieval specialist for GroundedDoc Agent.
Use query_documents to retrieve grounded answers with citations from the indexed corpus.
Always call query_documents before answering factual questions.
If the user asks about document disagreements or data conflicts, call get_conflict_report.
For index health questions, call get_ingestion_status.
Never invent citations or facts not returned by tools.
"""

verifier_instruction = """
You are the citation verifier for GroundedDoc Agent.
Review the prior tool output and ensure:
1) Every factual claim maps to a cited source from the tool response.
2) Conflicts between documents are explicitly surfaced when present.
3) If evidence is insufficient, say so clearly.
Return the verified answer with citations preserved.
"""

research_agent = Agent(
    model="gemini-2.0-flash",
    name="research_agent",
    description="Retrieves grounded answers and conflict reports from the document index.",
    instruction=research_instruction,
    tools=[query_documents, get_conflict_report, get_ingestion_status],
)

verifier_agent = Agent(
    model="gemini-2.0-flash",
    name="verifier_agent",
    description="Verifies citations and conflict handling in grounded answers.",
    instruction=verifier_instruction,
)

root_agent = SequentialAgent(
    name="grounded_doc_agent",
    description=(
        "Conflict-aware document intelligence agent with hierarchical indexing, "
        "adaptive retrieval, and mandatory citations."
    ),
    sub_agents=[research_agent, verifier_agent],
)
