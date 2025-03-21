"""
Knowledge generator node for LangGraph-based PDF RAG system.
Synthesizes retrieved content into coherent knowledge structures.
"""

import logging
import os
import json
from typing import Dict, Any, List
from datetime import datetime
from openai import OpenAI

from app.chat.langgraph.state import GraphState, ResearchState
from app.chat.types import ResearchContext

logger = logging.getLogger(__name__)

# Knowledge synthesis prompt template
KNOWLEDGE_SYNTHESIS_PROMPT = """
You are a knowledge synthesis expert for a technical document retrieval system.
Synthesize the following content retrieved from technical documents into a coherent knowledge structure.

Query: {query}

Retrieved Content:
{content}

Your task:
1. Identify the key facts and insights from the retrieved content
2. Organize them into a coherent knowledge structure
3. Identify relationships between different pieces of information
4. Note any contradictions or gaps in the information
5. Create a synthesis that directly addresses the query

Format your response as a JSON object with these keys:
- summary: A concise summary of the key information (1-2 paragraphs)
- facts: List of key facts extracted from the content (3-5 bullet points)
- insights: List of deeper insights derived from analysis (2-3 bullet points)
- gaps: List of identified information gaps (0-2 bullet points, if any)
- cross_references: List of related concepts worth exploring (0-3 items, if any)
"""

def generate_knowledge(state: GraphState) -> GraphState:
    """
    Generate synthesized knowledge from retrieved content.
    Simplified implementation without async/await.

    Args:
        state: Current graph state

    Returns:
        Updated graph state with knowledge synthesis
    """
    # Check required states
    if not state.retrieval_state or not state.query_state:
        logger.warning("Missing retrieval or query state, skipping knowledge generation")
        state.research_state = ResearchState(query_state=state.query_state) if state.query_state else None
        return state

    # Check if we have retrieval elements
    if not state.retrieval_state.elements:
        logger.warning("No elements retrieved, skipping knowledge generation")
        state.research_state = ResearchState(query_state=state.query_state)
        return state

    query = state.query_state.query
    logger.info(f"Generating knowledge synthesis for query: {query[:50]}...")

    try:
        # Format retrieved content for the prompt
        content_str = ""
        for i, element in enumerate(state.retrieval_state.elements[:10]):  # Limit to avoid token limits
            content_type = element.get("content_type", "text")
            content_str += f"\n--- Content {i+1} (Type: {content_type}) ---\n"
            content_str += element.get("content", "")[:1000] + ("..." if len(element.get("content", "")) > 1000 else "")

        # Initialize OpenAI client
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

        # Prepare the prompt
        prompt = KNOWLEDGE_SYNTHESIS_PROMPT.format(
            query=query,
            content=content_str
        )

        # Call the OpenAI API
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # Using GPT-4 for better synthesis
            messages=[
                {"role": "system", "content": "You are an expert knowledge synthesizer for technical documentation."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            response_format={"type": "json_object"}
        )

        # Parse the response
        try:
            synthesis = json.loads(response.choices[0].message.content)
        except json.JSONDecodeError:
            logger.warning("Failed to parse JSON from OpenAI response, using raw text")
            synthesis = {
                "summary": response.choices[0].message.content[:500],
                "facts": ["Unable to parse structured response"],
                "insights": [],
                "gaps": ["Information structure could not be parsed"],
                "cross_references": []
            }

        # Create research context from synthesis
        research_context = ResearchContext(
            query=query,
            summary=synthesis.get("summary", ""),
            facts=synthesis.get("facts", []),
            insights=synthesis.get("insights", []),
            gaps=synthesis.get("gaps", []),
            cross_references=synthesis.get("cross_references", []),
            sources=[
                {
                    "id": source.get("id", ""),
                    "title": source.get("title", ""),
                    "pdf_id": source.get("pdf_id", "")
                }
                for source in state.retrieval_state.sources if source
            ]
        )

        # Create research state
        state.research_state = ResearchState(
            query_state=state.query_state,
            research_context=research_context,
            cross_references=synthesis.get("cross_references", []),
            insights=synthesis.get("insights", []),
            metadata={
                "synthesis": synthesis,
                "model": "gpt-4o-mini",
                "element_count": len(state.retrieval_state.elements),
                "generated_at": datetime.now().isoformat()
            }
        )

        logger.info(
            f"Knowledge synthesis complete with {len(research_context.facts)} facts "
            f"and {len(research_context.insights)} insights"
        )

        return state

    except Exception as e:
        logger.error(f"Knowledge generation failed: {str(e)}", exc_info=True)

        # Initialize basic research state in case of error
        state.research_state = ResearchState(
            query_state=state.query_state,
            metadata={"error": str(e)}
        )

        return state
