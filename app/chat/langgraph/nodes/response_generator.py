"""
Enhanced response generator node for LangGraph-based PDF RAG system.
Generates comprehensive responses with improved citations and research insights.
"""

import logging
import os
import asyncio
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
import traceback

from openai import OpenAI
from app.chat.langgraph.state import RetrievalState, GraphState, GenerationState, ResearchState
from app.chat.types import ContentElement

logger = logging.getLogger(__name__)

# Enhanced response generation prompt template for single document
RESPONSE_GENERATION_PROMPT = """
You are an AI assistant specialized in answering queries about technical documents.
Generate a comprehensive response to the query based on the retrieved content.

Query: {query}

Retrieved Content:
{content}

Knowledge Synthesis:
{synthesis}

Your task:
1. Answer the query directly and accurately based on the retrieved content
2. Include relevant technical details and explanations
3. Cite specific sources for key information using [1], [2], etc. notation
4. Present a coherent and well-structured response
5. Acknowledge any limitations or gaps in the information

Use the knowledge synthesis to guide your response, but focus on addressing the query directly.
Make citations specific and relevant - cite the exact source where a piece of information comes from.
"""

# Enhanced response generation prompt template for research mode
RESEARCH_MODE_PROMPT = """
You are an AI assistant specializing in cross-document research and analysis.
Generate a comprehensive response combining insights from multiple documents.

Query: {query}

Retrieved Content:
{content}

Cross-Document Insights:
{insights}

Your task:
1. Answer the query by synthesizing information across all documents
2. Highlight similarities, differences, and complementary information between documents
3. Cite specific sources using [1], [2], etc. notation, mentioning document names
4. Organize your response in a clear, structured manner
5. Identify any gaps or contradictions between the documents

Make sure to explicitly mention when different documents provide different perspectives on the same topic.
For each key point, indicate which document(s) the information comes from.
If documents contradict each other, present both perspectives fairly.
"""

async def generate_response(state: GraphState) -> GraphState:
    """
    Generate the final response based on retrieved content and knowledge synthesis.
    Provides improved citations and handles research mode elegantly.

    Args:
        state: Current graph state

    Returns:
        Updated graph state with generated response
    """
    try:
        if (not state.retrieval_state or not state.query_state):
            logger.error("Retrieval state and query state are required for response generation")
            raise ValueError("Required states are missing")

        # Check if we have enough elements for a response
        if not state.retrieval_state.elements or len(state.retrieval_state.elements) == 0:
            logger.warning("No elements retrieved, generating fallback response")
            # Generate fallback response
            state.generation_state = GenerationState(
                response="I couldn't find specific information to answer your query. Could you please rephrase or provide more details?",
                metadata={"fallback": True}
            )
            return state

        logger.info(f"Generating response for query: {state.query_state.query}")

        # Determine if we're in research mode
        is_research_mode = False
        if state.query_state.pdf_ids and len(state.query_state.pdf_ids) > 1:
            is_research_mode = True
            logger.info("Using research mode for response generation")

        # Format retrieved content for the prompt
        content_str = ""
        citation_map = {}

        # Add document title mapping for research mode
        document_titles = {}

        # Sort elements by score first if available
        sorted_elements = sorted(
            state.retrieval_state.elements,
            key=lambda x: getattr(x.metadata, 'score', 0) if hasattr(x, 'metadata') else 0,
            reverse=True
        )

        # Limit to most relevant elements
        elements_to_use = sorted_elements[:12]

        for i, element in enumerate(elements_to_use):
            # Determine content type
            content_type = element.content_type.value if hasattr(element.content_type, 'value') else str(element.content_type)

            # Create citation ID
            citation_id = f"[{i+1}]"

            # Get PDF ID and page info
            pdf_id = element.pdf_id if hasattr(element, 'pdf_id') else "unknown"
            page = element.page if hasattr(element, 'page') else None

            # Get document title if available
            document_title = None
            if hasattr(element, 'metadata') and hasattr(element.metadata, 'document_title'):
                document_title = element.metadata.document_title
            elif hasattr(element, 'metadata') and isinstance(element.metadata, dict) and 'document_title' in element.metadata:
                document_title = element.metadata['document_title']
            else:
                document_title = f"Document {pdf_id}"

            # Store document title for research mode
            document_titles[pdf_id] = document_title

            # Create citation metadata
            citation_map[citation_id] = {
                "id": element.id if hasattr(element, 'id') else element.element_id if hasattr(element, 'element_id') else f"element_{i}",
                "type": content_type,
                "pdf_id": pdf_id,
                "page": page,
                "document_title": document_title
            }

            # Format the content with citation marker
            content_str += f"\n--- Content {citation_id} (Type: {content_type}, Document: {document_title}) ---\n"
            content_str += element.content[:1000] + ("..." if len(element.content) > 1000 else "")

        # Format research insights if available
        insights_str = ""
        if is_research_mode and hasattr(state, 'research_state') and state.research_state:
            if hasattr(state.research_state, 'insights') and state.research_state.insights:
                insights_str += "Cross-Document Insights:\n"
                for insight in state.research_state.insights:
                    insights_str += f"- {insight}\n"

            if hasattr(state.research_state, 'cross_references') and state.research_state.cross_references:
                insights_str += "\nCommon Concepts Across Documents:\n"
                for i, concept in enumerate(state.research_state.cross_references[:5]):
                    if isinstance(concept, dict) and 'name' in concept:
                        insights_str += f"- {concept['name']}\n"
                    else:
                        insights_str += f"- {concept}\n"

        # Format knowledge synthesis if available
        synthesis_str = ""
        if hasattr(state, 'research_state') and state.research_state and hasattr(state.research_state, 'research_context'):
            context = state.research_state.research_context
            if hasattr(context, 'summary') and context.summary:
                synthesis_str += f"Summary: {context.summary}\n\n"

            if hasattr(context, 'facts') and context.facts:
                synthesis_str += "Key Facts:\n"
                for fact in context.facts[:5]:  # Limit to most important facts
                    synthesis_str += f"- {fact}\n"
                synthesis_str += "\n"

            if hasattr(context, 'insights') and context.insights:
                synthesis_str += "Insights:\n"
                for insight in context.insights[:3]:  # Limit to key insights
                    synthesis_str += f"- {insight}\n"

        # Initialize OpenAI client
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        # Choose prompt based on research mode
        if is_research_mode:
            prompt = RESEARCH_MODE_PROMPT.format(
                query=state.query_state.query,
                content=content_str,
                insights=insights_str or "No specific cross-document insights available."
            )
        else:
            prompt = RESPONSE_GENERATION_PROMPT.format(
                query=state.query_state.query,
                content=content_str,
                synthesis=synthesis_str or "No knowledge synthesis available."
            )

        # Add context about available documents in research mode
        if is_research_mode:
            prompt += "\n\nAvailable Documents:\n"
            for pdf_id, title in document_titles.items():
                prompt += f"- {title} (ID: {pdf_id})\n"

        # Call the OpenAI API
        response = client.chat.completions.create(
            model="gpt-4-0125-preview",
            messages=[
                {"role": "system", "content": "You are an expert assistant for technical documentation."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )

        # Get the response text
        response_text = response.choices[0].message.content

        # Extract citations from the response
        citations = []
        for citation_id in citation_map:
            if citation_id in response_text:
                # Get citation data
                citation_data = citation_map[citation_id].copy()
                citations.append(citation_data)

        # Create generation state
        state.generation_state = GenerationState(
            response=response_text,
            citations=citations,
            metadata={
                "timestamp": datetime.utcnow().isoformat(),
                "model": "gpt-4-0125-preview",
                "token_usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                },
                "research_mode": is_research_mode,
                "document_titles": document_titles,
                "document_count": len(document_titles)
            },
            token_usage={
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            }
        )

        logger.info(
            f"Response generation complete with {len(citations)} citations, "
            f"{response.usage.total_tokens} total tokens, "
            f"research_mode={is_research_mode}"
        )

        return state

    except Exception as e:
        logger.error(f"Response generation failed: {str(e)}")
        logger.error(traceback.format_exc())

        # Generate error response
        error_response = "I'm sorry, but I encountered an error while generating a response. Please try again or rephrase your query."

        # Create error state
        state.generation_state = GenerationState(
            response=error_response,
            metadata={"error": str(e)}
        )

        return state
