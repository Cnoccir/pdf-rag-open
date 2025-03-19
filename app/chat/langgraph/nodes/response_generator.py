"""
Response generator node for LangGraph-based PDF RAG system.
Generates comprehensive responses with citations from retrieved content.
"""

import logging
import os
from typing import Dict, Any, List
from datetime import datetime
from openai import OpenAI

from app.chat.langgraph.state import GraphState, GenerationState

logger = logging.getLogger(__name__)

# Response generation prompt template for single document
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

# Response generation prompt template for research mode
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

def generate_response(state: GraphState) -> GraphState:
    """
    Generate the final response based on retrieved content and knowledge synthesis.

    Args:
        state: Current graph state containing retrieval results and knowledge synthesis

    Returns:
        Updated graph state with generated response
    """
    # Validate required states
    if not state.retrieval_state or not state.query_state:
        logger.warning("Missing retrieval or query state for response generation")
        error_msg = "Missing required information to generate a response"

        state.generation_state = GenerationState(
            response="I'm sorry, but I don't have enough information to provide a response.",
            citations=[],
            metadata={"error": error_msg}
        )
        return state

    # Check if we have elements to work with
    has_elements = state.retrieval_state.elements and len(state.retrieval_state.elements) > 0

    if not has_elements:
        logger.warning("No elements retrieved, generating fallback response")

        # Check for specific database errors
        error_msg = ""
        if state.retrieval_state.metadata and "error" in state.retrieval_state.metadata:
            error_msg = state.retrieval_state.metadata["error"]

        # Generate appropriate fallback response
        if "Neo4j" in error_msg or "neo4j" in error_msg:
            fallback_msg = ("I'm having trouble connecting to the document database. "
                           "This might be due to a temporary issue. "
                           "Please try again in a few moments.")
        else:
            # Generic fallback for when retrieval simply found nothing relevant
            fallback_msg = ("I couldn't find specific information in the document to answer your query. "
                           "Could you please rephrase your question or provide more details about what you're looking for?")

        state.generation_state = GenerationState(
            response=fallback_msg,
            citations=[],
            metadata={"fallback": True, "reason": "no_results"}
        )
        return state

    logger.info(f"Generating response for query: {state.query_state.query[:50]}...")

    try:
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

        # Sort elements by score if available
        sorted_elements = sorted(
            state.retrieval_state.elements,
            key=lambda x: x.get("metadata", {}).get("score", 0) if isinstance(x.get("metadata"), dict) else 0,
            reverse=True
        )

        # Limit to most relevant elements
        elements_to_use = sorted_elements[:12]

        for i, element in enumerate(elements_to_use):
            # Determine content type
            content_type = element.get("content_type", "text")

            # Create citation ID
            citation_id = f"[{i+1}]"

            # Get PDF ID and page info safely
            pdf_id = element.get("pdf_id", "")
            page = element.get("page", 0)

            # Get document title if available
            document_title = None
            if isinstance(element.get("metadata"), dict) and "document_title" in element["metadata"]:
                document_title = element["metadata"]["document_title"]
            else:
                document_title = f"Document {pdf_id}"

            # Store document title for research mode
            document_titles[pdf_id] = document_title

            # Create citation metadata
            citation_map[citation_id] = {
                "id": element.get("id", "") or element.get("element_id", f"element_{i}"),
                "type": content_type,
                "pdf_id": pdf_id,
                "page": page,
                "document_title": document_title
            }

            # Format the content with citation marker
            content_str += f"\n--- Content {citation_id} (Type: {content_type}, Document: {document_title}) ---\n"
            content_str += element.get("content", "")[:1000] + ("..." if len(element.get("content", "")) > 1000 else "")

        # Format research insights if available
        insights_str = ""
        if is_research_mode and state.research_state:
            if state.research_state.insights:
                insights_str += "Cross-Document Insights:\n"
                for insight in state.research_state.insights:
                    insights_str += f"- {insight}\n"

            if state.research_state.cross_references:
                insights_str += "\nCommon Concepts Across Documents:\n"
                for i, concept in enumerate(state.research_state.cross_references[:5]):
                    if isinstance(concept, dict) and "name" in concept:
                        insights_str += f"- {concept['name']}\n"
                    else:
                        insights_str += f"- {concept}\n"

        # Format knowledge synthesis if available
        synthesis_str = ""
        if state.research_state and hasattr(state.research_state, 'research_context'):
            context = state.research_state.research_context
            if context.summary:
                synthesis_str += f"Summary: {context.summary}\n\n"

            if context.facts:
                synthesis_str += "Key Facts:\n"
                for fact in context.facts[:5]:  # Limit to most important facts
                    synthesis_str += f"- {fact}\n"
                synthesis_str += "\n"

            if context.insights:
                synthesis_str += "Insights:\n"
                for insight in context.insights[:3]:  # Limit to key insights
                    synthesis_str += f"- {insight}\n"

        # Initialize OpenAI client
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

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
            model="gpt-4",  # Use the most capable model
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
                "timestamp": datetime.now().isoformat(),
                "model": "gpt-4",
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
            f"{response.usage.total_tokens} total tokens"
        )

        return state

    except Exception as e:
        logger.error(f"Response generation failed: {str(e)}", exc_info=True)

        # Generate error response
        error_response = "I'm sorry, but I encountered an error while generating a response. Please try again or rephrase your query."

        # Create error state
        state.generation_state = GenerationState(
            response=error_response,
            citations=[],
            metadata={"error": str(e)}
        )

        return state
