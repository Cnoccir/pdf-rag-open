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
from app.chat.models import ResearchContext

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

def generate_knowledge(state: GraphState) -> dict:
    """
    Generate synthesized knowledge from retrieved content.
    Simplified implementation without async/await.

    Args:
        state: Current graph state

    Returns:
        Dictionary with updated state components
    """
    # Check required states
    if not state.retrieval_state or not state.query_state:
        logger.warning("Missing retrieval or query state, skipping knowledge generation")
        state.research_state = ResearchState(query_state=state.query_state) if state.query_state else None
        return {"research_state": state.research_state}

    # Check if we have retrieval elements
    if not state.retrieval_state.elements:
        logger.warning("No elements retrieved, skipping knowledge generation")
        state.research_state = ResearchState(query_state=state.query_state)
        return {"research_state": state.research_state}

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

        # Format cross_references properly for the research state
        formatted_cross_references = []
        for ref in synthesis.get("cross_references", []):
            if isinstance(ref, str):
                formatted_cross_references.append({"name": ref, "type": "concept"})
            elif isinstance(ref, dict):
                formatted_cross_references.append(ref)

        # Store information in research state - don't create ResearchContext with attributes it doesn't have
        state.research_state = ResearchState(
            query_state=state.query_state,
            cross_references=formatted_cross_references,
            insights=synthesis.get("insights", []),
            metadata={
                "synthesis": synthesis,
                "summary": synthesis.get("summary", ""),
                "facts": synthesis.get("facts", []),  # Store in metadata instead of ResearchContext
                "gaps": synthesis.get("gaps", []),
                "model": "gpt-4o-mini",
                "element_count": len(state.retrieval_state.elements),
                "generated_at": datetime.now().isoformat()
            }
        )

        # Create an appropriate research context that matches its actual definition
        from app.chat.models import ResearchContext
        research_context = ResearchContext(primary_pdf_id=state.query_state.pdf_ids[0] if state.query_state.pdf_ids else None)

        # Add active PDF IDs
        if state.query_state.pdf_ids:
            for pdf_id in state.query_state.pdf_ids:
                research_context.add_document(pdf_id)

        # Set the research context properly
        state.research_state.research_context = research_context

        # Access facts from metadata, not from research_context
        facts_count = len(synthesis.get("facts", []))
        insights_count = len(synthesis.get("insights", []))

        logger.info(
            f"Knowledge synthesis complete with {facts_count} facts "
            f"and {insights_count} insights"
        )

        return {"research_state": state.research_state}

    except Exception as e:
        logger.error(f"Knowledge generation failed: {str(e)}", exc_info=True)

        # Initialize basic research state in case of error
        state.research_state = ResearchState(
            query_state=state.query_state,
            metadata={"error": str(e)},
            cross_references=[]  # Ensure this is a valid empty list
        )

        return {"research_state": state.research_state}
